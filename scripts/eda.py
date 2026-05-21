"""
Analyze and visualize a Donut-format dataset.

Usage:
    python scripts/eda.py --data-dir data/mc-ocr/donut_format/ --output docs/eda_figures/
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from PIL import Image


FIELDS = ["store_name", "date", "total", "address"]


def load_all_metadata(data_dir):
    """Load metadata.jsonl from every available split."""
    all_records = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, split, "metadata.jsonl")
        if not os.path.exists(path):
            continue
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    gt = json.loads(rec.get("ground_truth", "{}"))
                    rec["gt_parse"] = gt.get("gt_parse", {})
                    rec["split"] = split
                    rec["dir"] = os.path.join(data_dir, split)
                    records.append(rec)
        all_records[split] = records
    return all_records


def plot_split_counts(data, output_dir):
    """Plot image counts by split."""
    splits = list(data.keys())
    counts = [len(data[s]) for s in splits]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(splits, counts, color=["#2196F3", "#FF9800", "#4CAF50"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(count), ha="center", fontweight="bold")
    ax.set_title("Images per split")
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "split_counts.png"), dpi=150)
    plt.close()
    print("[OK] split_counts.png")


def plot_field_distribution(data, output_dir):
    """Plot field coverage across all splits."""
    all_records = [record for records in data.values() for record in records]
    total = len(all_records)

    filled = {}
    for field in FIELDS:
        count = sum(1 for record in all_records if record["gt_parse"].get(field))
        filled[field] = count / total * 100 if total else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(filled.keys(), filled.values(), color="#673AB7")
    for bar, pct in zip(bars, filled.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%", ha="center")
    ax.set_title("Field coverage")
    ax.set_ylabel("% records with field")
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "field_distribution.png"), dpi=150)
    plt.close()
    print("[OK] field_distribution.png")


def plot_text_length(data, output_dir):
    """Plot text length distribution by field."""
    all_records = [record for records in data.values() for record in records]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, field in zip(axes.flat, FIELDS):
        lengths = [len(record["gt_parse"].get(field, "")) for record in all_records if record["gt_parse"].get(field)]
        if lengths:
            ax.hist(lengths, bins=30, color="#009688", edgecolor="white")
        ax.set_title(f"{field} (n={len(lengths)})")
        ax.set_xlabel("Characters")
    fig.suptitle("Text length by field")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "text_lengths.png"), dpi=150)
    plt.close()
    print("[OK] text_lengths.png")


def plot_image_sizes(data, output_dir):
    """Plot image dimensions for a sample of records."""
    all_records = [record for records in data.values() for record in records]
    widths, heights = [], []

    for rec in all_records[:200]:
        img_path = os.path.join(rec["dir"], rec["file_name"])
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
            except Exception:
                pass

    if widths:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(widths, heights, alpha=0.5, s=10, color="#E91E63")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title(f"Image dimensions (n={len(widths)})")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "image_sizes.png"), dpi=150)
        plt.close()
        print("[OK] image_sizes.png")


def visualize_samples(data, output_dir, n=20):
    """Render a grid of sample images and annotations."""
    all_records = [record for records in data.values() for record in records]
    samples = all_records[:n]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5))

    for i, ax in enumerate(axes.flat):
        if i >= len(samples):
            ax.axis("off")
            continue
        rec = samples[i]
        img_path = os.path.join(rec["dir"], rec["file_name"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        ax.set_title(rec["file_name"], fontsize=7)
        info = "\n".join(f"{k}: {v}" for k, v in rec["gt_parse"].items() if v)
        ax.set_xlabel(info, fontsize=6, ha="left", x=0)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Sample images and annotations", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "samples.png"), dpi=100)
    plt.close()
    print("[OK] samples.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze a Donut-format dataset")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="docs/eda_figures/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load_all_metadata(args.data_dir)

    if not data:
        print(f"[ERROR] No metadata.jsonl found in {args.data_dir}")
        return

    print(f"[INFO] Loaded: {', '.join(f'{k}={len(v)}' for k, v in data.items())}")

    plot_split_counts(data, args.output)
    plot_field_distribution(data, args.output)
    plot_text_length(data, args.output)
    plot_image_sizes(data, args.output)
    visualize_samples(data, args.output)

    print(f"\n[DONE] Figures saved to {args.output}")


if __name__ == "__main__":
    main()
