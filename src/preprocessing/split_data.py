from collections import Counter
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


def build_stratify_key(sample: Dict) -> str:
    labels = sorted(set(sample.get('labels', [])))
    quality = sample.get('image_quality', None)

    if quality is None:
        quality_bin = 'unknown'
    else:
        try:
            q = float(quality)
            if q < 0.33:
                quality_bin = 'low'
            elif q < 0.66:
                quality_bin = 'medium'
            else:
                quality_bin = 'high'
        except Exception:
            quality_bin = 'unknown'

    field_key = '+'.join(labels) if labels else 'no_label'
    return f'{field_key}__{quality_bin}'



def safe_train_val_split(samples: List[Dict], val_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
    stratify_labels = [build_stratify_key(sample) for sample in samples]
    counts = Counter(stratify_labels)

    if any(v < 2 for v in counts.values()):
        train_samples, val_samples = train_test_split(samples, test_size=val_size, random_state=random_state)
    else:
        train_samples, val_samples = train_test_split(
            samples,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    return train_samples, val_samples

