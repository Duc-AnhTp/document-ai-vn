"""
E1 Baseline: PaddleOCR + rule-based KIE.

Su dung:
    python scripts/baseline_paddleocr.py --test-dir data/mc-ocr/donut_format/test/ --output results/e1_baseline/
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import compute_metrics, save_metrics, load_metadata, extract_gt_parse


def extract_fields_rule(text_lines):
    """
    Trich xuat 4 truong KIE bang regex/rule tu ket qua OCR.

    Parameters
    ----------
    text_lines : list[str]
        Danh sach dong text OCR (da sap xep top->bottom).

    Returns
    -------
    dict voi keys: store_name, date, total, address
    """
    full_text = "\n".join(text_lines)
    result = {"store_name": "", "date": "", "total": "", "address": ""}

    # --- Ten cua hang: dong dau tien (heuristic) ---
    if text_lines:
        result["store_name"] = text_lines[0].strip()

    # --- Ngay: pattern dd/mm/yyyy hoac dd-mm-yyyy ---
    date_patterns = [
        r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}",
    ]
    for pat in date_patterns:
        match = re.search(pat, full_text)
        if match:
            result["date"] = match.group(0)
            break

    # --- Tong tien: tim so lon nhat hoac dong co "TONG"/"TOTAL" ---
    total_patterns = [
        r"(?:t[oô]ng|total|th[aà]nh ti[eề]n|c[oộ]ng)[\s:]*(\d[\d\.,]*)",
        r"(\d[\d\.,]*)\s*(?:đ|vn[dđ]|dong)",
    ]
    for pat in total_patterns:
        match = re.search(pat, full_text, re.IGNORECASE)
        if match:
            result["total"] = match.group(1).strip()
            break

    # --- Dia chi: dong co "Đ/C", "DC:", hoac chua P./Q./TP. ---
    addr_patterns = [
        r"(?:[đd][/]?c|[đd]ia ch[iỉ])[\s:]+(.+)",
    ]
    for pat in addr_patterns:
        match = re.search(pat, full_text, re.IGNORECASE)
        if match:
            result["address"] = match.group(1).strip()
            break

    # Fallback: tim dong co "P." hoac "Q." hoac "TP."
    if not result["address"]:
        for line in text_lines:
            if re.search(r"(?:P\.|Q\.|TP\.|phuong|quan|thanh pho)", line, re.IGNORECASE):
                result["address"] = line.strip()
                break

    return result


def run_paddleocr_on_image(ocr, image_path):
    """Chay PaddleOCR tren 1 anh, tra ve danh sach dong text sap xep."""
    result = ocr.ocr(image_path, cls=True)

    if not result or not result[0]:
        return []

    # Sap xep theo toa do y (top -> bottom)
    boxes_text = []
    for line in result[0]:
        box = line[0]
        text = line[1][0]
        y_center = (box[0][1] + box[2][1]) / 2
        boxes_text.append((y_center, text))

    boxes_text.sort(key=lambda x: x[0])
    return [t for _, t in boxes_text]


def main():
    parser = argparse.ArgumentParser(description="E1: PaddleOCR + rule-based KIE baseline")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", default="results/e1_baseline/")
    args = parser.parse_args()

    # Load ground truth
    metadata_path = os.path.join(args.test_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        print(f"[LOI] Khong tim thay {metadata_path}")
        return

    records = load_metadata(metadata_path)
    print(f"[INFO] {len(records)} anh test")

    # Init PaddleOCR
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)
    except ImportError:
        print("[LOI] Can cai: pip install paddleocr paddlepaddle")
        return

    preds = []
    golds = []
    predictions = []

    for i, rec in enumerate(records):
        img_path = os.path.join(args.test_dir, rec["file_name"])
        gt = extract_gt_parse(rec)
        golds.append(gt)

        if not os.path.exists(img_path):
            preds.append({"store_name": "", "date": "", "total": "", "address": ""})
            continue

        # OCR
        text_lines = run_paddleocr_on_image(ocr, img_path)

        # Rule-based extraction
        pred = extract_fields_rule(text_lines)
        preds.append(pred)

        predictions.append({
            "file_name": rec["file_name"],
            "ocr_text": text_lines,
            "prediction": pred,
            "ground_truth": gt,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(records)}]")

    # Compute metrics
    metrics = compute_metrics(preds, golds)
    metrics["experiment"] = "E1_PaddleOCR_Baseline"
    metrics["num_samples"] = len(records)

    os.makedirs(args.output, exist_ok=True)
    save_metrics(metrics, os.path.join(args.output, "metrics.json"))

    # Save predictions
    with open(os.path.join(args.output, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"[OK] predictions.json -> {args.output}")

    # Print results
    print(f"\n{'='*40}")
    print("E1 BASELINE RESULTS")
    print(f"{'='*40}")
    print(f"Overall F1:        {metrics['overall']['f1']}")
    print(f"Overall Precision: {metrics['overall']['precision']}")
    print(f"Overall Recall:    {metrics['overall']['recall']}")
    for field, m in metrics["per_field"].items():
        print(f"  {field:15s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")


if __name__ == "__main__":
    main()
