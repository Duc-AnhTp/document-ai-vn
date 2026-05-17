"""TF-IDF + Logistic Regression line classifier for receipt KIE."""

import json
import os
import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from src.postprocess import normalize_money, normalize_text, normalize_timestamp


LABELS = ["SELLER", "SELLER_ADDRESS", "TIMESTAMP", "TOTAL_COST", "OTHER"]
FIELD_LABELS = LABELS[:4]


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", normalize_text(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def _overlap_ratio(a: str, b: str) -> float:
    a = _normalize_for_match(a)
    b = _normalize_for_match(b)
    if not a or not b:
        return 0.0
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words & b_words) / len(a_words) if a_words else 0.0


def _field_match_score(field: str, line_text: str, gt_val: str) -> float:
    if not line_text or not gt_val:
        return 0.0
    if field == "TOTAL_COST":
        return 1.0 if normalize_money(line_text) and normalize_money(line_text) == normalize_money(gt_val) else 0.0
    if field == "TIMESTAMP":
        line_ts = normalize_timestamp(line_text)
        gt_ts = normalize_timestamp(gt_val)
        if line_ts and gt_ts and line_ts == gt_ts:
            return 1.0
    line_norm = _normalize_for_match(line_text)
    gt_norm = _normalize_for_match(gt_val)
    if line_norm and gt_norm and (line_norm in gt_norm or gt_norm in line_norm):
        return 1.0
    return _overlap_ratio(line_text, gt_val)


def assign_line_labels(ocr_lines: List[Dict], gt_fields: Dict[str, str]) -> List[str]:
    """Assign one KIE label to each OCR line using field-aware matching."""
    labels = []
    for line in ocr_lines:
        text = line.get("text", "")
        best_label = "OTHER"
        best_score = 0.5
        for field in FIELD_LABELS:
            score = _field_match_score(field, text, gt_fields.get(field, ""))
            if score > best_score:
                best_score = score
                best_label = field
        labels.append(best_label)
    return labels


def build_feature_matrix(lines: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    layout = []
    texts = []
    for line in lines:
        x1, y1, x2, y2 = line.get("bbox", [0, 0, 100, 20])
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        x_center = line.get("x_center", (x1 + x2) / 2)
        layout.append([
            line.get("relative_y", 0.5),
            x_center / 1000.0,
            width / 1000.0,
            height / 100.0,
        ])
        texts.append(line.get("text_lower", line.get("text", "").lower()))
    return np.array(layout, dtype=np.float32), texts


def train_classifier(all_lines: List[List[Dict]], all_labels: List[List[str]]) -> dict:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    flat_lines = [line for lines in all_lines for line in lines]
    flat_labels = [label for labels in all_labels for label in labels]
    if not flat_lines:
        raise ValueError("No OCR lines available for classifier training")
    label_counts = Counter(flat_labels)
    missing = [label for label in FIELD_LABELS if label_counts.get(label, 0) == 0]
    if missing:
        raise ValueError(f"Cannot train classifier; missing labels: {', '.join(missing)}")

    layout_feats, texts = build_feature_matrix(flat_lines)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000, sublinear_tf=True)
    tfidf_feats = vectorizer.fit_transform(texts)
    scaler = StandardScaler()
    layout_scaled = scaler.fit_transform(layout_feats)
    x_train = hstack([tfidf_feats, csr_matrix(layout_scaled)])

    clf = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(x_train, flat_labels)
    return {"vectorizer": vectorizer, "scaler": scaler, "clf": clf, "labels": LABELS, "label_counts": dict(label_counts)}


def train_classifier_from_dataset(
    dataset_name: str,
    ocr_lang: str = "en",
    use_ocr_cache: bool = True,
    force_ocr: bool = False,
    cache_dir: str = os.path.join("outputs", "ocr_cache"),
) -> dict:
    from src.line_processing import add_line_features
    from src.ocr import create_ocr_engine, run_ocr_cached

    data_path = os.path.join("data", "processed", dataset_name, "train.json")
    with open(data_path, encoding="utf-8") as file:
        records = json.load(file)

    ocr_engine = create_ocr_engine(lang=ocr_lang)
    all_lines = []
    all_labels = []
    errors = []

    print(f"  Running OCR on {len(records)} train images...")
    for idx, rec in enumerate(records):
        if (idx + 1) % 100 == 0:
            print(f"    [{idx + 1}/{len(records)}]")
        try:
            lines = run_ocr_cached(
                rec["image_path"],
                ocr_engine,
                dataset_name=dataset_name,
                image_id=rec.get("image_id"),
                cache_dir=cache_dir,
                use_cache=use_ocr_cache,
                force_ocr=force_ocr,
            )
            if not lines:
                errors.append((rec.get("image_id", f"idx_{idx}"), "OCR returned 0 lines"))
                continue
            import cv2

            image = cv2.imread(rec["image_path"])
            if image is None:
                errors.append((rec.get("image_id", f"idx_{idx}"), f"cv2.imread failed: {rec['image_path']}"))
                continue
            lines = add_line_features(lines, image.shape[0])
            all_lines.append(lines)
            all_labels.append(assign_line_labels(lines, rec["gt"]))
        except Exception as exc:
            errors.append((rec.get("image_id", f"idx_{idx}"), str(exc)))

    success = len(records) - len(errors)
    if errors:
        print(f"  Skipped: {len(errors)}/{len(records)} images")
        for image_id, reason in errors[:20]:
            print(f"    Skip {image_id}: {reason}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more errors")
    if records and success < len(records) * 0.5:
        raise RuntimeError(f"Too many classifier training images failed: {len(errors)}/{len(records)}")

    print(f"  Successful: {success}/{len(records)} images")
    model = train_classifier(all_lines, all_labels)
    model["metadata"] = {
        "dataset": dataset_name,
        "train_records": len(records),
        "successful_records": success,
        "ocr_lang": ocr_lang,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    return model


def predict_fields(lines: List[Dict], model_bundle: dict, include_meta: bool = False) -> Dict[str, str]:
    from scipy.sparse import csr_matrix, hstack

    layout_feats, texts = build_feature_matrix(lines)
    tfidf_feats = model_bundle["vectorizer"].transform(texts)
    layout_scaled = model_bundle["scaler"].transform(layout_feats)
    x_test = hstack([tfidf_feats, csr_matrix(layout_scaled)])

    proba = model_bundle["clf"].predict_proba(x_test)
    class_order = list(model_bundle["clf"].classes_)
    fields = {"SELLER": "", "SELLER_ADDRESS": "", "TIMESTAMP": "", "TOTAL_COST": ""}
    source_line_ids = {}

    for field in FIELD_LABELS:
        if field not in class_order:
            continue
        col = class_order.index(field)
        best_idx = int(np.argmax(proba[:, col]))
        if proba[best_idx, col] > 0.3:
            fields[field] = lines[best_idx]["text"]
            source_line_ids[field] = lines[best_idx].get("line_id")

    fields["SELLER"] = normalize_text(fields["SELLER"])
    fields["SELLER_ADDRESS"] = normalize_text(fields["SELLER_ADDRESS"])
    fields["TIMESTAMP"] = normalize_timestamp(fields["TIMESTAMP"])
    fields["TOTAL_COST"] = normalize_money(fields["TOTAL_COST"])
    if include_meta:
        fields["_meta"] = {"source_line_ids": source_line_ids}
    return fields
