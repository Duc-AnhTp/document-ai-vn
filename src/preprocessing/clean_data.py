import re
from copy import deepcopy
from typing import Dict, List

# Các entity label hợp lệ trong MC-OCR
VALID_ENTITY_LABELS = {
    'SELLER', 'ADDRESS', 'TIMESTAMP', 'TOTAL_COST', 'OTHER',
}

# Bảng chuẩn hóa: map các biến thể label về dạng chuẩn
LABEL_NORMALIZATION_MAP = {
    'SELLER': 'SELLER',
    'STORE': 'SELLER',
    'ADDRESS': 'ADDRESS',
    'ADDR': 'ADDRESS',
    'TIMESTAMP': 'TIMESTAMP',
    'DATE': 'TIMESTAMP',
    'TIME': 'TIMESTAMP',
    'TOTAL_COST': 'TOTAL_COST',
    'TOTAL': 'TOTAL_COST',
    'COST': 'TOTAL_COST',
    'OTHER': 'OTHER',
    'OTHERS': 'OTHER',
    'NONE': 'OTHER',
}

SPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = SPACE_PATTERN.sub(' ', text)
    return text



def normalize_label(label: str) -> str:
    label = str(label).strip().upper()
    return LABEL_NORMALIZATION_MAP.get(label, label)



def align_modalities(sample: Dict) -> Dict:
    sample = deepcopy(sample)
    texts = sample.get('texts', [])
    labels = sample.get('labels', [])
    polygons = sample.get('polygons', [])

    target_len = min(len(texts), len(labels))
    if polygons:
        target_len = min(target_len, len(polygons))

    sample['texts'] = texts[:target_len]
    sample['labels'] = labels[:target_len]
    sample['polygons'] = polygons[:target_len] if polygons else []
    return sample



def clean_sample(sample: Dict) -> Dict:
    sample = align_modalities(sample)

    cleaned_texts: List[str] = []
    cleaned_labels: List[str] = []
    cleaned_polygons: List = []

    polygons = sample.get('polygons', [])
    for idx, (text, label) in enumerate(zip(sample.get('texts', []), sample.get('labels', []))):
        text = clean_text(text)
        label = normalize_label(label)

        if not text:
            continue
        if label not in VALID_ENTITY_LABELS:
            continue

        cleaned_texts.append(text)
        cleaned_labels.append(label)
        if polygons:
            cleaned_polygons.append(polygons[idx])

    sample['texts'] = cleaned_texts
    sample['labels'] = cleaned_labels
    sample['polygons'] = cleaned_polygons
    return sample
