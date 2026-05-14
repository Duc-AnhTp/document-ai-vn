"""
Hàm tiện ích dùng chung cho toàn bộ dự án Document AI KIE.
"""

import json
import os
from pathlib import Path

import yaml
from PIL import Image


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(yaml_path: str) -> dict:
    """Đọc file YAML config → dict."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Metrics ─────────────────────────────────────────────────────────────────

FIELDS = ["store_name", "date", "total", "address"]


def normalize_text(text: str) -> str:
    """Chuẩn hoá text để so sánh: lowercase, bỏ khoảng trắng thừa."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def compute_metrics(preds: list[dict], golds: list[dict]) -> dict:
    """
    Tính F1 / Precision / Recall cho KIE fields.

    Parameters
    ----------
    preds : list[dict]
        Mỗi dict có keys = FIELDS, values = chuỗi trích xuất.
    golds : list[dict]
        Ground truth cùng format.

    Returns
    -------
    dict với overall và per_field metrics.
    """
    per_field = {}

    for field in FIELDS:
        tp = fp = fn = 0
        for pred, gold in zip(preds, golds):
            p = normalize_text(pred.get(field, ""))
            g = normalize_text(gold.get(field, ""))

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

    # Macro average
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
    """Lưu metrics ra JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu metrics → {path}")


# ── Donut output parsing ───────────────────────────────────────────────────

def parse_donut_output(generated_text: str, task_prompt: str = "") -> dict:
    """
    Parse output decoder Donut → dict các trường KIE.

    Donut sinh text dạng:
    <s_mcocr><s_store_name>ABC</s_store_name><s_date>01/01</s_date>...
    """
    import re

    result = {}
    for field in FIELDS:
        pattern = rf"<s_{field}>(.*?)</s_{field}>"
        match = re.search(pattern, generated_text)
        result[field] = match.group(1).strip() if match else ""

    return result


# ── Visualization ──────────────────────────────────────────────────────────

def visualize_sample(image_path: str, annotation: dict, ax=None):
    """Hiển thị ảnh + annotation text overlay."""
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 8))

    ax.imshow(img)
    ax.set_title(os.path.basename(image_path), fontsize=8)

    # Hiển thị annotation bên dưới ảnh
    info_lines = [f"{k}: {v}" for k, v in annotation.items() if v]
    info_text = "\n".join(info_lines)
    ax.set_xlabel(info_text, fontsize=7, ha="left", x=0)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


# ── Data loading ───────────────────────────────────────────────────────────

def load_metadata(metadata_path: str) -> list[dict]:
    """Đọc metadata.jsonl → list of dicts."""
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_gt_parse(record: dict) -> dict:
    """Trích gt_parse từ 1 record metadata.jsonl."""
    gt_str = record.get("ground_truth", "{}")
    gt = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
    return gt.get("gt_parse", {})
