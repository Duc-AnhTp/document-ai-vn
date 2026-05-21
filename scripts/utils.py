"""Shared utilities for Donut-only KIE training and evaluation."""

import json
import os
import re
import unicodedata

from PIL import Image


FIELDS = ["store_name", "date", "total", "address"]


def load_config(yaml_path: str) -> dict:
    """Load a YAML config file."""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Normalize text before field comparison."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text).strip().lower())
    return " ".join(text.split())


def normalize_field_value(field: str, text: str) -> str:
    """Apply field-specific normalization for metric comparison."""
    text = normalize_text(text)

    if field == "date":
        text = re.sub(r"[\.\-]", "/", text)
        text = re.sub(r"\s*/\s*", "/", text)
        return text

    if field == "total":
        text = re.sub(r"(vnd|vnd|dong|d)", "", text)
        text = re.sub(r"[,\.\s]", "", text)
        return text

    if field == "address":
        replacements = {
            "thanh pho": "tp",
            "tp.": "tp",
            "quan ": "q ",
            "q.": "q",
            "phuong ": "p ",
            "p.": "p",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        text = re.sub(r"[^\w\s]", " ", text)
        return " ".join(text.split())

    return text


def compute_metrics(preds: list[dict], golds: list[dict]) -> dict:
    """Compute macro precision, recall and F1 across KIE fields."""
    per_field = {}

    for field in FIELDS:
        tp = fp = fn = 0
        for pred, gold in zip(preds, golds):
            p = normalize_field_value(field, pred.get(field, ""))
            g = normalize_field_value(field, gold.get(field, ""))

            if g and p and p == g:
                tp += 1
            elif p and (not g or p != g):
                fp += 1
            if g and (not p or p != g):
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_field[field] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    macro_p = sum(v["precision"] for v in per_field.values()) / len(FIELDS)
    macro_r = sum(v["recall"] for v in per_field.values()) / len(FIELDS)
    macro_f1 = sum(v["f1"] for v in per_field.values()) / len(FIELDS)

    return {
        "overall": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
        "per_field": per_field,
    }


def save_metrics(metrics: dict, path: str):
    """Save metrics to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {path}")


def parse_donut_output(generated_text: str, task_prompt: str = "") -> dict:
    """Parse Donut decoder text into the four-field KIE schema."""
    result = {}
    for field in FIELDS:
        pattern = rf"<s_{field}>(.*?)</s_{field}>"
        match = re.search(pattern, generated_text)
        result[field] = match.group(1).strip() if match else ""

    return result


def serialize_donut_parse(data) -> str:
    """Serialize ground truth into the decoder target text."""
    if isinstance(data, dict) and set(data.keys()).issubset(set(FIELDS)):
        return "".join(f"<s_{k}>{v}</s_{k}>" for k, v in data.items() if v not in ("", None))
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def visualize_sample(image_path: str, annotation: dict, ax=None):
    """Render one image with its annotation text below the plot."""
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 8))

    ax.imshow(img)
    ax.set_title(os.path.basename(image_path), fontsize=8)

    info_lines = [f"{k}: {v}" for k, v in annotation.items() if v]
    info_text = "\n".join(info_lines)
    ax.set_xlabel(info_text, fontsize=7, ha="left", x=0)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def load_metadata(metadata_path: str) -> list[dict]:
    """Load metadata.jsonl into a list of records."""
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_gt_parse(record: dict) -> dict:
    """Extract gt_parse from one metadata.jsonl record."""
    gt_str = record.get("ground_truth", "{}")
    gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
    return gt.get("gt_parse", {})
