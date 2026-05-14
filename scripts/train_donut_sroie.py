"""
E3: Fine-tune Donut (tu E2) tren SROIE 2019 — cross-dataset.

Su dung:
    python scripts/train_donut_sroie.py --config configs/donut_sroie.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import load_config, compute_metrics, save_metrics, parse_donut_output

# Reuse training logic tu train_donut.py
from scripts.train_donut import main as train_main


def run_error_analysis(checkpoint_dir, test_dir, output_dir):
    """Phan tich loi cross-lingual sau khi train xong."""
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DonutProcessor.from_pretrained(checkpoint_dir)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    metadata_path = os.path.join(test_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        print(f"[SKIP] Khong tim thay {metadata_path}")
        return

    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    errors = {"total_samples": len(records), "error_counts": {}, "error_types": {}, "examples": []}
    field_errors = {"store_name": 0, "date": 0, "total": 0, "address": 0}

    preds_list, golds_list = [], []

    for rec in records:
        img_path = os.path.join(test_dir, rec["file_name"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated = model.generate(
                pixel_values,
                max_length=model.config.decoder.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_beams=1,
            )

        pred_text = processor.tokenizer.decode(generated[0], skip_special_tokens=False)
        pred = parse_donut_output(pred_text)
        gt = json.loads(rec["ground_truth"]).get("gt_parse", {})

        preds_list.append(pred)
        golds_list.append(gt)

        # Count errors per field
        for field in field_errors:
            p = pred.get(field, "").strip().lower()
            g = gt.get(field, "").strip().lower()
            if g and p != g:
                field_errors[field] += 1
                # Save first 3 examples per field
                if len([e for e in errors["examples"] if e["field"] == field]) < 3:
                    errors["examples"].append({
                        "file": rec["file_name"],
                        "field": field,
                        "predicted": pred.get(field, ""),
                        "expected": gt.get(field, ""),
                    })

    errors["error_counts"] = field_errors
    metrics = compute_metrics(preds_list, golds_list)
    errors["metrics"] = metrics

    output_path = os.path.join(output_dir, "error_analysis.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"[OK] Error analysis -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E3: Cross-dataset SROIE")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # Train (reuse E2 logic)
    print("=" * 50)
    print("E3: Fine-tune Donut tren SROIE")
    print("=" * 50)
    train_main()

    # Error analysis
    config = load_config(args.config)
    ckpt = config["output"]["checkpoint_dir"]
    test_dir = config["data"]["test_dir"]
    output_dir = os.path.dirname(config["output"]["log_file"])

    print("\n" + "=" * 50)
    print("Error Analysis")
    print("=" * 50)
    run_error_analysis(ckpt, test_dir, output_dir)


if __name__ == "__main__":
    main()
