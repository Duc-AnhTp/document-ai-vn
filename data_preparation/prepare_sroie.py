"""Convert SROIE2019 raw annotations to the common JSON format."""

import json
import os
import random

from data_preparation.normalize import normalize_money_gt, normalize_text, normalize_timestamp_gt


SROIE_DIR = os.path.join("data", "sroie", "SROIE2019")
OUT_DIR = os.path.join("data", "processed", "sroie")


def _load_split(split_dir: str) -> list:
    entities_dir = os.path.join(split_dir, "entities")
    img_dir = os.path.join(split_dir, "img")
    if not os.path.isdir(entities_dir):
        return []

    records = []
    for fname in sorted(os.listdir(entities_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]
        entity_path = os.path.join(entities_dir, fname)
        img_path = os.path.join(img_dir, stem + ".jpg")
        if not os.path.isfile(img_path):
            continue
        try:
            with open(entity_path, encoding="utf-8", errors="replace") as file:
                data = json.load(file)
        except Exception:
            continue

        records.append({
            "image_id": stem,
            "image_path": img_path.replace("\\", "/"),
            "gt": {
                "SELLER": normalize_text(data.get("company", "")),
                "SELLER_ADDRESS": normalize_text(data.get("address", "")),
                "TIMESTAMP": normalize_timestamp_gt(data.get("date", "")),
                "TOTAL_COST": normalize_money_gt(data.get("total", "")),
            },
        })
    return records


def prepare_sroie(val_ratio: float = 0.2, seed: int = 42) -> tuple:
    all_records = _load_split(os.path.join(SROIE_DIR, "train"))
    rng = random.Random(seed)
    rng.shuffle(all_records)
    n_val = int(len(all_records) * val_ratio)
    return all_records[n_val:], all_records[:n_val]


def save(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def main():
    print("Preparing SROIE dataset...")
    train, val = prepare_sroie()
    train_path = os.path.join(OUT_DIR, "train.json")
    val_path = os.path.join(OUT_DIR, "val.json")
    save(train, train_path)
    save(val, val_path)
    print(f"  Train: {len(train)} records -> {train_path}")
    print(f"  Val:   {len(val)} records -> {val_path}")
    return {"train": len(train), "val": len(val), "skipped": 0}


if __name__ == "__main__":
    main()
