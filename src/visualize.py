"""Visualization helpers for OCR boxes, field highlights, and metrics charts."""

import os
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate import eval_normalize


FIELD_COLORS_BGR = {
    "SELLER": (0, 0, 220),
    "SELLER_ADDRESS": (220, 0, 0),
    "TIMESTAMP": (0, 180, 0),
    "TOTAL_COST": (0, 140, 255),
}


def draw_ocr_boxes(image: np.ndarray, lines: List[Dict]) -> np.ndarray:
    img_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    for line in lines:
        x1, y1, x2, y2 = map(int, line["bbox"])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = line["text"][:25] + "..." if len(line["text"]) > 25 else line["text"]
        cv2.putText(img_bgr, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_field_highlight(image: np.ndarray, lines: List[Dict], predicted_fields: Dict[str, str]) -> np.ndarray:
    img_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    matched_lines = {}

    source_ids = predicted_fields.get("_meta", {}).get("source_line_ids", {})
    for field, source_line_id in source_ids.items():
        if field in FIELD_COLORS_BGR and source_line_id is not None:
            matched_lines[int(source_line_id)] = field

    for field, value in predicted_fields.items():
        if field not in FIELD_COLORS_BGR or not value:
            continue
        value_norm = eval_normalize(field, value)
        if not value_norm:
            continue
        for idx, line in enumerate(lines):
            if idx in matched_lines:
                continue
            text_norm = eval_normalize(field, line["text"])
            if value_norm and text_norm and (value_norm in text_norm or text_norm in value_norm):
                matched_lines[idx] = field
                break

    for idx, line in enumerate(lines):
        field = matched_lines.get(idx)
        if field is None:
            continue
        x1, y1, x2, y2 = map(int, line["bbox"])
        color = FIELD_COLORS_BGR[field]
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0, img_bgr)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_bgr, field, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_prediction_examples(
    records: List[Dict],
    predictions: List[Dict],
    output_dir: str,
    n: int = 5,
    ocr_lines_list: Optional[List[List[Dict]]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = min(n, len(records))
    for idx in range(n):
        rec = records[idx]
        pred = predictions[idx]
        image = cv2.imread(rec["image_path"])
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if ocr_lines_list and idx < len(ocr_lines_list):
            image_rgb = draw_field_highlight(image_rgb, ocr_lines_list[idx], pred)

        fig, axis = plt.subplots(1, 1, figsize=(8, 10))
        axis.imshow(image_rgb)
        axis.axis("off")
        gt = rec.get("gt", {})
        caption_lines = ["PREDICTION vs GT:"]
        for field in FIELD_COLORS_BGR:
            ok = "OK" if eval_normalize(field, pred.get(field, "")) == eval_normalize(field, gt.get(field, "")) else "ERR"
            caption_lines.append(f"{ok} {field}: {pred.get(field, '')[:30]!r} | GT: {gt.get(field, '')[:30]!r}")
        fig.text(0.01, 0.01, "\n".join(caption_lines), fontsize=7, family="monospace", verticalalignment="bottom")
        plt.savefig(os.path.join(output_dir, f"example_{idx + 1:03d}.png"), bbox_inches="tight", dpi=100)
        plt.close(fig)
    print(f"Saved {n} examples -> {output_dir}")


def plot_f1_bar_chart(metrics_dict: Dict[str, Dict], output_path: str) -> None:
    fields = ["SELLER", "SELLER_ADDRESS", "TIMESTAMP", "TOTAL_COST"]
    exp_names = list(metrics_dict.keys())
    if not exp_names:
        raise ValueError("metrics_dict is empty")
    x = np.arange(len(fields))
    width = 0.8 / len(exp_names)
    fig, axis = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for idx, (exp_name, metrics) in enumerate(metrics_dict.items()):
        f1_vals = [metrics.get(field, {}).get("f1", 0) for field in fields]
        offset = (idx - len(exp_names) / 2 + 0.5) * width
        bars = axis.bar(x + offset, f1_vals, width, label=exp_name, color=colors[idx % len(colors)], alpha=0.85)
        for bar, val in zip(bars, f1_vals):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    axis.set_xlabel("Field")
    axis.set_ylabel("F1 Score")
    axis.set_title("F1 Score per Field by Experiment")
    axis.set_xticks(x)
    axis.set_xticklabels(fields)
    axis.set_ylim(0, 1.1)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved chart -> {output_path}")
