"""
Convert annotation SROIE 2019 -> format gt_parse cua Donut.

SROIE structure expected:
    data/sroie/SROIE2019/train/img/
    data/sroie/SROIE2019/train/entities/
    data/sroie/SROIE2019/test/img/
    data/sroie/SROIE2019/test/entities/

Su dung:
    python scripts/convert_sroie.py --input data/sroie/ --output data/sroie/donut_format/
"""

import argparse
import json
import os
import shutil

import numpy as np


def split_train_val(records, val_ratio=0.1, seed=42):
    """Tach val tu train, giu nguyen thu tu reproducible."""
    np.random.seed(seed)
    idx = np.random.permutation(len(records))
    n_val = int(len(records) * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


def write_split(records, output_dir, name):
    split_dir = os.path.join(output_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    count = missing = 0
    with open(os.path.join(split_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            img = rec["file_name"]
            src = os.path.join(rec["_img_dir"], img)
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(split_dir, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            gt = {
                "store_name": rec.get("company", ""),
                "date": rec.get("date", ""),
                "total": rec.get("total", ""),
                "address": rec.get("address", "")
            }
            f.write(json.dumps({"file_name": img, "ground_truth": json.dumps({"gt_parse": gt}, ensure_ascii=False)}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  [{name}] {count} anh | {missing} thieu")


def main():
    parser = argparse.ArgumentParser(description="Convert SROIE -> Donut format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sroie_root = args.input
    if os.path.exists(os.path.join(sroie_root, "SROIE2019")):
        sroie_root = os.path.join(sroie_root, "SROIE2019")
        
    train_records = []
    test_records = []
    failed_files = []
    
    for split in ["train", "test"]:
        img_dir = os.path.join(sroie_root, split, "img")
        key_dir = os.path.join(sroie_root, split, "entities")
        
        if not os.path.exists(img_dir) or not os.path.exists(key_dir):
            continue
            
        for txt_file in os.listdir(key_dir):
            if not txt_file.endswith(".txt"): continue
            file_path = os.path.join(key_dir, txt_file)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    base_name = os.path.splitext(txt_file)[0]
                    # Try multiple image extensions
                    img_name = None
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidate = base_name + ext
                        if os.path.exists(os.path.join(img_dir, candidate)):
                            img_name = candidate
                            break
                    if img_name is None:
                        img_name = base_name + ".jpg"  # fallback
                    data["file_name"] = img_name
                    data["_img_dir"] = img_dir
                    if split == "train":
                        train_records.append(data)
                    else:
                        test_records.append(data)
                except Exception as e:
                    failed_files.append({"file": file_path, "error": str(e)})

    all_count = len(train_records) + len(test_records)
    if all_count == 0:
        print(f"[LOI] Khong tim thay du lieu trong {sroie_root} (Can cau truc SROIE2019/train/img va entities)")
        return
                
    print(f"[INFO] Tong hop: {all_count} records (official train={len(train_records)}, official test={len(test_records)})")
    if failed_files:
        print(f"[WARN] Bo qua {len(failed_files)} file parse loi")
        for item in failed_files[:10]:
            print(f"  - {item['file']}: {item['error']}")
    
    # Giu nguyen official test set, chi tach val tu official train
    train, val = split_train_val(train_records, val_ratio=0.1)
    test = test_records
    print(f"[INFO] Split: train={len(train)} val={len(val)} test={len(test)} (official test giu nguyen)")

    for sname, sdata in [("train", train), ("val", val), ("test", test)]:
        write_split(sdata, args.output, sname)

    print("[DONE]")


if __name__ == "__main__":
    main()
