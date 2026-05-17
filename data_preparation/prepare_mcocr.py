"""Convert MC-OCR2021 raw CSV annotations to the common JSON format."""

import csv
import json
import os
import random
import unicodedata

from data_preparation.normalize import normalize_money_gt, normalize_text, normalize_timestamp_gt


RAW_DIR = os.path.join("data", "mc-ocr", "raw")
CSV_PATH = os.path.join(RAW_DIR, "mcocr_train_df.csv")
IMG_DIR = os.path.join(RAW_DIR, "train_images")
OUT_DIR = os.path.join("data", "processed", "mc-ocr")
IMAGE_DIR_CANDIDATES = [
    IMG_DIR,
    os.path.join(IMG_DIR, "train_images"),
    os.path.join(RAW_DIR, "data0.7", "data0.7"),
    os.path.join(RAW_DIR, "kie_data", "kie_data", "images"),
]


def _is_broken_text(text: str) -> bool:
    chars = [char for char in text if not char.isspace()]
    return bool(chars) and all(ord(char) > 127 and unicodedata.category(char) == "Cn" for char in chars)


def _money_candidate_score(text: str) -> tuple:
    normalized = normalize_money_gt(text)
    original = normalize_text(text).lower()
    keyword_penalty = 1 if any(k in original for k in ["tổng", "tong", "total", "amount"]) else 0
    return (1 if normalized else 0, len(normalized), -keyword_penalty)


def _parse_anno(anno_texts: str, anno_labels: str) -> dict | None:
    """Parse paired MC-OCR text/label strings into four normalized field buckets."""
    texts = anno_texts.split("|||")
    labels = anno_labels.split("|||")
    if len(texts) != len(labels):
        return None

    buckets = {"SELLER": [], "SELLER_ADDRESS": [], "TIMESTAMP": [], "TOTAL_COST": []}
    for text, label in zip(texts, labels):
        text = normalize_text(text)
        label = label.strip()
        if label == "ADDRESS":
            label = "SELLER_ADDRESS"
        if label in buckets and text:
            buckets[label].append(text)

    fields = {
        "SELLER": buckets["SELLER"][0] if buckets["SELLER"] else "",
        "SELLER_ADDRESS": " ".join(buckets["SELLER_ADDRESS"]),
        "TIMESTAMP": " ".join(buckets["TIMESTAMP"]),
        "TOTAL_COST": "",
    }

    money_candidates = [text for text in buckets["TOTAL_COST"] if normalize_money_gt(text)]
    if money_candidates:
        fields["TOTAL_COST"] = max(money_candidates, key=_money_candidate_score)
    elif buckets["TOTAL_COST"]:
        fields["TOTAL_COST"] = buckets["TOTAL_COST"][-1]

    return fields


def _resolve_image_path(img_id: str) -> str | None:
    for directory in IMAGE_DIR_CANDIDATES:
        path = os.path.join(directory, img_id)
        if os.path.isfile(path):
            return path
    return None


def _normalize_fields(fields: dict) -> dict:
    return {
        "SELLER": normalize_text(fields.get("SELLER", "")),
        "SELLER_ADDRESS": normalize_text(fields.get("SELLER_ADDRESS", "")),
        "TIMESTAMP": normalize_timestamp_gt(fields.get("TIMESTAMP", "")),
        "TOTAL_COST": normalize_money_gt(fields.get("TOTAL_COST", "")),
    }


def prepare_mcocr(val_ratio: float = 0.2, seed: int = 42) -> tuple:
    """Read MC-OCR train CSV, validate image paths, and create a reproducible split."""
    records = []
    skipped = 0

    with open(CSV_PATH, encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            img_id = row.get("img_id", "").strip()
            anno_texts = row.get("anno_texts", "").strip()
            anno_labels = row.get("anno_labels", "").strip()

            if not img_id or not anno_texts or not anno_labels:
                skipped += 1
                continue

            img_path = _resolve_image_path(img_id)
            if not img_path:
                skipped += 1
                continue

            fields = _parse_anno(anno_texts, anno_labels)
            if fields is None or _is_broken_text(fields.get("SELLER", "")):
                skipped += 1
                continue

            records.append({
                "image_id": img_id.replace(".jpg", ""),
                "image_path": img_path.replace("\\", "/"),
                "gt": _normalize_fields(fields),
            })

    rng = random.Random(seed)
    rng.shuffle(records)
    n_val = int(len(records) * val_ratio)
    return records[n_val:], records[:n_val], skipped


def save(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def main():
    print("Preparing MC-OCR dataset...")
    train, val, skipped = prepare_mcocr()
    train_path = os.path.join(OUT_DIR, "train.json")
    val_path = os.path.join(OUT_DIR, "val.json")
    save(train, train_path)
    save(val, val_path)
    print(f"  Train: {len(train)} records -> {train_path}")
    print(f"  Val:   {len(val)} records -> {val_path}")
    print(f"  Skipped: {skipped} records")
    return {"train": len(train), "val": len(val), "skipped": skipped}


if __name__ == "__main__":
    main()
