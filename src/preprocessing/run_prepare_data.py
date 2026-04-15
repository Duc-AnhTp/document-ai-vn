"""Pipeline tiền xử lý: parse CSV → clean → BIO labeling → split → lưu JSONL."""

import logging
from pathlib import Path

from PIL import Image

from configs.paths import CSV_PATH, IMAGE_DIR, PROCESSED_DIR
from src.preprocessing.clean_data import clean_sample
from src.preprocessing.parse_mcocr import parse_mcocr_csv
from src.preprocessing.split_data import safe_train_val_split
from src.preprocessing.utils import normalize_bbox, polygon_to_bbox, save_jsonl

logger = logging.getLogger(__name__)


def convert_sample(sample):
    """Chuyển đổi 1 sample raw thành định dạng JSONL với BIO labels.

    Returns:
        Dict chứa image_path, words, boxes, labels. None nếu ảnh không tồn tại.
    """
    image_path = IMAGE_DIR / sample['image_id']
    if not image_path.exists():
        return None

    with Image.open(image_path) as img:
        width, height = img.size

    words = sample.get('texts', [])
    entities = sample.get('labels', [])
    polygons = sample.get('polygons', [])
    boxes = []
    bio_labels = []

    prev_entity = None
    for idx, entity in enumerate(entities):
        if polygons:
            bbox = polygon_to_bbox(polygons[idx])
            bbox = normalize_bbox(bbox, width, height)
        else:
            bbox = [0, 0, 0, 0]
        boxes.append(bbox)

        # Entity 'OTHER' là background → label 'O' (không dùng BIO prefix)
        if entity == 'OTHER':
            bio_labels.append('O')
            prev_entity = None
        else:
            prefix = 'B' if entity != prev_entity else 'I'
            bio_labels.append(f'{prefix}-{entity}')
            prev_entity = entity

    return {
        'image_path': str(image_path),
        'image_id': sample['image_id'],
        'words': words,
        'boxes': boxes,
        'labels': bio_labels,
        'image_quality': sample.get('image_quality', None),
    }


def main():
    """Chạy toàn bộ pipeline tiền xử lý dữ liệu MC-OCR."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    samples = parse_mcocr_csv(str(CSV_PATH))
    samples = [clean_sample(sample) for sample in samples]
    samples = [sample for sample in samples if sample.get('texts')]

    converted = [convert_sample(sample) for sample in samples]
    converted = [sample for sample in converted if sample is not None]

    train_samples, val_samples = safe_train_val_split(converted)

    save_jsonl(converted, PROCESSED_DIR / 'all.jsonl')
    save_jsonl(train_samples, PROCESSED_DIR / 'train.jsonl')
    save_jsonl(val_samples, PROCESSED_DIR / 'val.jsonl')

    logger.info('Tong so mau hop le: %d', len(converted))
    logger.info('So mau train: %d', len(train_samples))
    logger.info('So mau val: %d', len(val_samples))
    logger.info('Da luu vao: %s', PROCESSED_DIR)


if __name__ == '__main__':
    main()

