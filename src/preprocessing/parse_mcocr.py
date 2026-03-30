import ast
from typing import Any, Dict, List

import pandas as pd


def safe_split(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split('|||') if str(item).strip()]


def safe_literal_eval(value: Any):
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return []


def parse_mcocr_csv(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    samples: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        sample = {
            'image_id': row.get('img_id', ''),
            'texts': safe_split(row.get('anno_texts', '')),
            'labels': safe_split(row.get('anno_labels', '')),
            'polygons': safe_literal_eval(row.get('anno_polygons', '')),
            'anno_num': int(row.get('anno_num', 0)) if not pd.isna(row.get('anno_num', 0)) else 0,
            'image_quality': row.get('anno_image_quality', None),
        }
        samples.append(sample)

    return samples
