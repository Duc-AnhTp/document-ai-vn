from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from configs.layoutlmv3_config import ID2LABEL
from configs.paths import CHECKPOINT_DIR
from src.ocr.run_paddleocr import extract_text_and_boxes
from src.utils.bbox import normalize_bbox


MODEL_DIR = CHECKPOINT_DIR / 'best_model'
processor = LayoutLMv3Processor.from_pretrained(str(MODEL_DIR) if MODEL_DIR.exists() else 'microsoft/layoutlmv3-base', apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_DIR) if MODEL_DIR.exists() else 'microsoft/layoutlmv3-base')
model.eval()



def group_entities(words: List[str], labels: List[str]) -> Dict[str, str]:
    result: Dict[str, List[str]] = {}
    current_label = None
    for word, label in zip(words, labels):
        if label == 'O':
            current_label = None
            continue

        tag, entity = label.split('-', 1)
        if tag == 'B' or entity != current_label:
            result.setdefault(entity, []).append(word)
            current_label = entity
        else:
            result[entity][-1] += ' ' + word
    return {k: ' | '.join(v) for k, v in result.items()}



def predict_from_image(image_path: str):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    words, raw_boxes, _ = extract_text_and_boxes(image_path)
    if not words:
        return {}

    boxes = [normalize_bbox(box, width, height) for box in raw_boxes]
    encoding = processor(image, words, boxes=boxes, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    with torch.no_grad():
        outputs = model(**encoding)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

    word_ids = encoding.word_ids(batch_index=0)
    decoded_labels = []
    kept_words = []
    prev_word_id = None
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            continue
        label_id = predictions[token_idx]
        decoded_labels.append(ID2LABEL.get(label_id, 'O'))
        kept_words.append(words[word_id])
        prev_word_id = word_id

    return group_entities(kept_words, decoded_labels)


if __name__ == '__main__':
    sample_path = input('Nhap duong dan anh: ').strip()
    if sample_path:
        print(predict_from_image(sample_path))
