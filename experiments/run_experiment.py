"""Run one receipt KIE experiment from a YAML config."""

import argparse
import json
import os
import pickle
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import compute_field_metrics, print_metrics_table
from src.ocr import create_ocr_engine
from src.pipeline import EMPTY_FIELDS, run_pipeline


PROCESSED_DIR = os.path.join("data", "processed")
OUTPUTS_DIR = "outputs"


def load_dataset(dataset_name: str, split: str = "val") -> list:
    path = os.path.join(PROCESSED_DIR, dataset_name, f"{split}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run python -m data_preparation.prepare_all first.")
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_classifier(train_dataset_name: str, config: dict):
    model_path = os.path.join(OUTPUTS_DIR, "models", f"classifier_{train_dataset_name}.pkl")
    if os.path.isfile(model_path) and not config.get("force_train_classifier", False):
        with open(model_path, "rb") as file:
            return pickle.load(file)

    print(f"  Training classifier on {train_dataset_name}...")
    from src.classifier import train_classifier_from_dataset

    model = train_classifier_from_dataset(
        train_dataset_name,
        ocr_lang=config.get("ocr_lang", "en"),
        use_ocr_cache=config.get("use_ocr_cache", True),
        force_ocr=config.get("force_ocr", False),
        cache_dir=config.get("ocr_cache_dir", os.path.join(OUTPUTS_DIR, "ocr_cache")),
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"  Saved model -> {model_path}")
    return model


def run_experiment(config: dict, dataset_override: str = None) -> dict:
    exp_name = config["name"]
    dataset_name = dataset_override or config.get("dataset", "mc-ocr")
    records = load_dataset(dataset_name, split="val")
    if not records:
        raise ValueError(f"Dataset {dataset_name} validation split is empty")

    use_preprocess = config.get("use_preprocess", False)
    extractor_type = config.get("extractor_type", "scoring")
    ocr_lang = config.get("ocr_lang", "en")
    use_ocr_cache = config.get("use_ocr_cache", True)
    force_ocr = config.get("force_ocr", False)
    cache_dir = config.get("ocr_cache_dir", os.path.join(OUTPUTS_DIR, "ocr_cache"))
    error_threshold = float(config.get("error_threshold", 0.30))
    limit = config.get("limit")
    if limit:
        records = records[: int(limit)]

    print(f"\nExperiment: {exp_name}")
    print(f"Dataset:    {dataset_name} (val)")
    print(f"Mode:       {extractor_type}, preprocess={use_preprocess}, cache={use_ocr_cache}")
    print(f"Records:    {len(records)}")

    ocr_engine = create_ocr_engine(lang=ocr_lang)
    classifier_model = None
    if extractor_type == "classifier":
        classifier_model = load_classifier(config.get("classifier_train_dataset", dataset_name), config)

    predictions = []
    ground_truths = []
    errors = []
    start = time.time()

    for idx, record in enumerate(records):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(records)}] {time.time() - start:.1f}s")
        try:
            pred = run_pipeline(
                record["image_path"],
                ocr_engine,
                extractor_type=extractor_type,
                use_preprocess=use_preprocess,
                classifier_model=classifier_model,
                dataset_name=dataset_name,
                image_id=record.get("image_id"),
                use_ocr_cache=use_ocr_cache,
                force_ocr=force_ocr,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            errors.append({"image_id": record.get("image_id"), "image_path": record.get("image_path"), "error": str(exc)})
            pred = EMPTY_FIELDS.copy()
        predictions.append({field: pred.get(field, "") for field in EMPTY_FIELDS})
        ground_truths.append(record["gt"])

    elapsed = time.time() - start
    error_rate = len(errors) / len(records)
    print(f"Done {len(records)} images in {elapsed:.1f}s ({elapsed / len(records):.2f}s/image)")
    if errors:
        print(f"Errors: {len(errors)} images ({error_rate:.1%})")
        for err in errors[:10]:
            print(f"  {err['image_id']}: {err['error']}")
    if error_rate > error_threshold:
        _save_results(exp_name, dataset_name, predictions, ground_truths, {}, records, errors)
        raise RuntimeError(f"Experiment failed: error rate {error_rate:.1%} exceeds threshold {error_threshold:.1%}")

    metrics = compute_field_metrics(predictions, ground_truths)
    print_metrics_table(metrics, name=exp_name)
    _save_results(exp_name, dataset_name, predictions, ground_truths, metrics, records, errors)
    return metrics


def _save_results(exp_name, dataset_name, predictions, ground_truths, metrics, records, errors):
    pred_dir = os.path.join(OUTPUTS_DIR, "predictions")
    metric_dir = os.path.join(OUTPUTS_DIR, "metrics")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    pred_path = os.path.join(pred_dir, f"{exp_name}_{dataset_name}.json")
    output_records = []
    for rec, pred, gt in zip(records, predictions, ground_truths):
        output_records.append({
            "image_id": rec.get("image_id"),
            "image_path": rec.get("image_path"),
            "prediction": pred,
            "ground_truth": gt,
        })
    payload = {"records": output_records, "errors": errors}
    with open(pred_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    metric_path = os.path.join(metric_dir, f"{exp_name}_{dataset_name}.json")
    with open(metric_path, "w", encoding="utf-8") as file:
        json.dump({"metrics": metrics, "errors": errors}, file, ensure_ascii=False, indent=2)
    print(f"Predictions -> {pred_path}")
    print(f"Metrics     -> {metric_path}")


def main():
    parser = argparse.ArgumentParser(description="Run KIE experiment")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--dataset", default=None, help="Override dataset: mc-ocr | sroie | all")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    datasets = ["mc-ocr", "sroie"] if args.dataset == "all" else [args.dataset or config.get("dataset", "mc-ocr")]
    for dataset in datasets:
        run_experiment(config, dataset_override=dataset)


if __name__ == "__main__":
    main()
