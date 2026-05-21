"""
Convert MC-OCR 2021 annotations to Donut metadata.jsonl format.

Usage:
    python scripts/convert_mcocr.py --input data/mc-ocr/raw/ --output data/mc-ocr/donut_format/ --split-ratio 0.8 0.1 0.1
"""

import argparse
import csv
import json
import os
import shutil
import sys
import unicodedata

import numpy as np


FIELDS = ["store_name", "date", "total", "address"]


def normalize_unicode(text):
    if not text or not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def find_col(columns, candidates):
    if not columns:
        return candidates[0]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for candidate in candidates:
        for col in columns:
            if candidate.lower() in col.lower():
                return col
    return candidates[0]


def parse_csv(csv_path):
    records = []
    column_mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        print(f"[INFO] CSV columns: {cols}")

        if "anno_texts" in cols and "anno_labels" in cols:
            column_mapping = {"mode": "mcocr_kie_format", "image": "img_id"}
            print("[INFO] Detected MC-OCR KIE format: anno_texts + anno_labels")
            label_map = {
                "SELLER": "store_name",
                "TIMESTAMP": "date",
                "TOTAL_COST": "total",
                "ADDRESS": "address",
            }
            for row in reader:
                texts = row["anno_texts"].split("|||")
                labels = row["anno_labels"].split("|||")
                grouped = {field: [] for field in FIELDS}
                for text, label in zip(texts, labels):
                    mapped = label_map.get(label)
                    if mapped:
                        grouped[mapped].append(text)
                records.append(
                    {
                        "file_name": row.get("img_id", ""),
                        "store_name": normalize_unicode(" ".join(grouped["store_name"])),
                        "date": normalize_unicode(" ".join(grouped["date"])),
                        "total": normalize_unicode(" ".join(grouped["total"])),
                        "address": normalize_unicode(" ".join(grouped["address"])),
                    }
                )
        else:
            column_mapping = {
                "store_name": find_col(cols, ["store_name", "seller", "company"]),
                "date": find_col(cols, ["timestamp", "date", "time"]),
                "total": find_col(cols, ["total_cost", "total", "total_amount"]),
                "address": find_col(cols, ["address", "addr"]),
                "image": find_col(cols, ["img_id", "image_id", "filename", "file_name"]),
            }
            print(f"[INFO] Fallback mapping: {column_mapping}")
            for row in reader:
                records.append(
                    {
                        "file_name": row.get(column_mapping["image"], ""),
                        "store_name": normalize_unicode(row.get(column_mapping["store_name"], "")),
                        "date": normalize_unicode(row.get(column_mapping["date"], "")),
                        "total": normalize_unicode(row.get(column_mapping["total"], "")),
                        "address": normalize_unicode(row.get(column_mapping["address"], "")),
                    }
                )
    return records, column_mapping


def split_data(records, ratios=(0.8, 0.1, 0.1), seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(records))
    n1 = int(len(records) * ratios[0])
    n2 = int(len(records) * ratios[1])
    return [records[i] for i in idx[:n1]], [records[i] for i in idx[n1 : n1 + n2]], [records[i] for i in idx[n1 + n2 :]]


def find_image_dir(input_dir):
    for name in ["images", "train_images", "img", "mcocr_public_145K_images"]:
        path = os.path.join(input_dir, name)
        if os.path.isdir(path):
            return path
    return input_dir


def write_split(records, img_dir, output_dir, name):
    split_dir = os.path.join(output_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    count = missing = 0

    img_cache = {}
    for dirpath, _, filenames in os.walk(img_dir):
        for filename in filenames:
            img_cache[filename] = os.path.join(dirpath, filename)

    with open(os.path.join(split_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            img = rec["file_name"]
            src = img_cache.get(img)

            if not src:
                for ext in [".jpg", ".jpeg", ".png"]:
                    if img + ext in img_cache:
                        src = img_cache[img + ext]
                        img = img + ext
                        break

            if not src:
                missing += 1
                continue

            dst = os.path.join(split_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

            gt = {field: rec.get(field, "") for field in FIELDS}
            row = {
                "file_name": img,
                "ground_truth": json.dumps({"gt_parse": gt}, ensure_ascii=False),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"  [{name}] {count} images | {missing} missing")


def main():
    parser = argparse.ArgumentParser(description="Convert MC-OCR to Donut metadata.jsonl format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--force", action="store_true", help="Skip mapping confirmation")
    args = parser.parse_args()

    csv_files = sorted([f for f in os.listdir(args.input) if f.endswith(".csv")])
    if not csv_files:
        print(f"[ERROR] No CSV file found in {args.input}")
        return
    if len(csv_files) > 1:
        print(f"[WARN] Found multiple CSV files: {csv_files}")
        if args.force:
            print("[WARN] Using the first CSV because --force is enabled")
        else:
            print("[INFO] Remove unrelated CSV files if the mapping below is wrong")
    csv_path = os.path.join(args.input, csv_files[0])

    img_dir = find_image_dir(args.input)
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Images: {img_dir}")

    records, mapping = parse_csv(csv_path)
    print(f"[INFO] Total records: {len(records)}")

    if records:
        print("\n--- SAMPLE RECORD ---")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))
        print("---------------------\n")

    if not args.force and sys.stdin.isatty():
        print(f"[INFO] Selected CSV: {os.path.basename(csv_path)}")
        print(f"[INFO] Final mapping: {mapping}")
        ans = input("Continue with this mapping? (y/n): ")
        if ans.lower() != "y":
            print("Cancelled.")
            return
    else:
        print(f"[INFO] Selected CSV: {os.path.basename(csv_path)}")
        print(f"[INFO] Final mapping: {mapping}")
        print("[INFO] Non-interactive environment or --force enabled. Continuing automatically...")

    train, val, test = split_data(records, tuple(args.split_ratio))
    print(f"[INFO] Split: train={len(train)} val={len(val)} test={len(test)}")

    for split_name, split_data_records in [("train", train), ("val", val), ("test", test)]:
        write_split(split_data_records, img_dir, args.output, split_name)

    print("[DONE]")


if __name__ == "__main__":
    main()
