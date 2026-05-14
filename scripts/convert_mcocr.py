"""
Convert annotation MC-OCR 2021 -> format gt_parse cua Donut.

Su dung:
    python scripts/convert_mcocr.py --input data/mc-ocr/raw/ --output data/mc-ocr/donut_format/ --split-ratio 0.8 0.1 0.1
"""

import argparse
import csv
import json
import os
import shutil
import unicodedata

import numpy as np


def normalize_unicode(text):
    if not text or not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def find_col(columns, candidates):
    if not columns:
        return candidates[0]
    for c in candidates:
        if c in columns:
            return c
    for c in candidates:
        for col in columns:
            if c.lower() in col.lower():
                return col
    return candidates[0]


def parse_csv(csv_path):
    records = []
    cm = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        print(f"[INFO] Cot CSV: {cols}")
        
        if "anno_texts" in cols and "anno_labels" in cols:
            cm = {"mode": "mcocr_kie_format", "image": "img_id"}
            print(f"[INFO] Phat hien format MC-OCR KIE (anno_texts, anno_labels)")
            label_map = {"SELLER": "store_name", "TIMESTAMP": "date", "TOTAL_COST": "total", "ADDRESS": "address"}
            for row in reader:
                texts = row["anno_texts"].split("|||")
                labels = row["anno_labels"].split("|||")
                res = {"store_name": [], "date": [], "total": [], "address": []}
                for t, l in zip(texts, labels):
                    mapped = label_map.get(l)
                    if mapped:
                        res[mapped].append(t)
                records.append({
                    "file_name": row.get("img_id", ""),
                    "store_name": normalize_unicode(" ".join(res["store_name"])),
                    "date": normalize_unicode(" ".join(res["date"])),
                    "total": normalize_unicode(" ".join(res["total"])),
                    "address": normalize_unicode(" ".join(res["address"])),
                })
        else:
            cm = {
                "store_name": find_col(cols, ["store_name", "seller", "company"]),
                "date": find_col(cols, ["timestamp", "date", "time"]),
                "total": find_col(cols, ["total_cost", "total", "total_amount"]),
                "address": find_col(cols, ["address", "addr"]),
                "image": find_col(cols, ["img_id", "image_id", "filename", "file_name"]),
            }
            print(f"[INFO] Mapping fallback: {cm}")
            for row in reader:
                records.append({
                    "file_name": row.get(cm["image"], ""),
                    "store_name": normalize_unicode(row.get(cm["store_name"], "")),
                    "date": normalize_unicode(row.get(cm["date"], "")),
                    "total": normalize_unicode(row.get(cm["total"], "")),
                    "address": normalize_unicode(row.get(cm["address"], "")),
                })
    return records, cm


def split_data(records, ratios=(0.8, 0.1, 0.1), seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(records))
    n1 = int(len(records) * ratios[0])
    n2 = int(len(records) * ratios[1])
    return [records[i] for i in idx[:n1]], [records[i] for i in idx[n1:n1+n2]], [records[i] for i in idx[n1+n2:]]


def find_image_dir(input_dir):
    for name in ["images", "train_images", "img", "mcocr_public_145K_images"]:
        p = os.path.join(input_dir, name)
        if os.path.isdir(p):
            return p
    return input_dir


def write_split(records, img_dir, output_dir, name):
    split_dir = os.path.join(output_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    count = missing = 0
    
    img_cache = {}
    for dirpath, _, filenames in os.walk(img_dir):
        for f in filenames:
            img_cache[f] = os.path.join(dirpath, f)
            
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
            gt = {"store_name": rec["store_name"], "date": rec["date"], "total": rec["total"], "address": rec["address"]}
            f.write(json.dumps({"file_name": img, "ground_truth": json.dumps({"gt_parse": gt}, ensure_ascii=False)}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  [{name}] {count} anh | {missing} thieu")


def main():
    parser = argparse.ArgumentParser(description="Convert MC-OCR -> Donut format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--force", action="store_true", help="Bo qua confirm")
    args = parser.parse_args()

    # Tim CSV
    csv_files = sorted([f for f in os.listdir(args.input) if f.endswith(".csv")])
    if not csv_files:
        print(f"[LOI] Khong tim thay CSV trong {args.input}")
        return
    if len(csv_files) > 1:
        print(f"[WARN] Tim thay nhieu CSV: {csv_files}")
        if args.force:
            print("[WARN] Dang dung CSV dau tien vi --force duoc bat")
        else:
            print("[INFO] Hay sap xep/loai bo CSV khong dung neu mapping khong nhu mong doi")
    csv_path = os.path.join(args.input, csv_files[0])

    img_dir = find_image_dir(args.input)
    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Anh: {img_dir}")

    records, mapping = parse_csv(csv_path)
    print(f"[INFO] Tong: {len(records)} records")

    if records:
        print("\n--- SAMPLE RECORD ---")
        try:
            print(json.dumps(records[0], indent=2, ensure_ascii=False))
        except UnicodeEncodeError:
            print(json.dumps(records[0], indent=2))
        print("---------------------\n")
        
    if not args.force:
        print(f"[INFO] Dang su dung CSV: {os.path.basename(csv_path)}")
        print(f"[INFO] Mapping cuoi: {mapping}")
        ans = input("Ban co chac muon tiep tuc voi mapping nay khong? (y/n): ")
        if ans.lower() != 'y':
            print("Huy bo.")
            return

    train, val, test = split_data(records, tuple(args.split_ratio))
    print(f"[INFO] Split: train={len(train)} val={len(val)} test={len(test)}")

    for sname, sdata in [("train", train), ("val", val), ("test", test)]:
        write_split(sdata, img_dir, args.output, sname)

    print("[DONE]")


if __name__ == "__main__":
    main()
