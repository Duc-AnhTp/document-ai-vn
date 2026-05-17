"""Evaluation metrics for receipt KIE."""

import re
import unicodedata
from typing import Dict, List

from src.postprocess import normalize_money, normalize_text, normalize_timestamp


FIELDS = ["SELLER", "SELLER_ADDRESS", "TIMESTAMP", "TOTAL_COST"]


def eval_normalize(field: str, text: str) -> str:
    """Field-aware normalization before exact match and CER."""
    if not text:
        return ""
    if field == "TOTAL_COST":
        return normalize_money(text)
    if field == "TIMESTAMP":
        return normalize_timestamp(text).lower()
    text = normalize_text(text)
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def compute_cer(pred: str, gt: str, field: str = "") -> float:
    p = eval_normalize(field, pred) if field else normalize_text(pred).lower()
    g = eval_normalize(field, gt) if field else normalize_text(gt).lower()
    if not g:
        return 0.0 if not p else 1.0
    return _edit_distance(p, g) / len(g)


def compute_exact_match(pred: str, gt: str, field: str = "") -> bool:
    return (eval_normalize(field, pred) if field else normalize_text(pred).lower()) == (
        eval_normalize(field, gt) if field else normalize_text(gt).lower()
    )


def compute_field_metrics(predictions: List[Dict[str, str]], ground_truths: List[Dict[str, str]]) -> Dict:
    assert len(predictions) == len(ground_truths)
    results = {}
    all_f1 = []
    all_cer = []

    for field in FIELDS:
        tp = fp = fn = 0
        cers = []
        for pred, gt in zip(predictions, ground_truths):
            p_val = pred.get(field, "")
            g_val = gt.get(field, "")
            p_norm = eval_normalize(field, p_val)
            g_norm = eval_normalize(field, g_val)
            cers.append(compute_cer(p_val, g_val, field=field))

            if g_norm and p_norm == g_norm:
                tp += 1
            elif p_norm and p_norm != g_norm:
                fp += 1
                if g_norm:
                    fn += 1
            elif g_norm and not p_norm:
                fn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        avg_cer = sum(cers) / len(cers) if cers else 0.0
        results[field] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "cer": round(avg_cer, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        all_f1.append(f1)
        all_cer.append(avg_cer)

    results["macro_f1"] = round(sum(all_f1) / len(all_f1), 4)
    results["avg_cer"] = round(sum(all_cer) / len(all_cer), 4)
    return results


def print_metrics_table(metrics: Dict, name: str = "") -> None:
    if name:
        print(f"\n{'=' * 60}")
        print(f"  Experiment: {name}")
    print(f"{'=' * 60}")
    print(f"  {'Field':<20} {'P':>7} {'R':>7} {'F1':>7} {'CER':>7}")
    print(f"  {'-' * 50}")
    for field in FIELDS:
        m = metrics[field]
        print(f"  {field:<20} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {m['cer']:>7.4f}")
    print(f"  {'-' * 50}")
    print(f"  {'Macro-F1':<20} {metrics['macro_f1']:>7.4f}")
    print(f"  {'Avg CER':<20} {metrics['avg_cer']:>7.4f}")
    print(f"{'=' * 60}")
