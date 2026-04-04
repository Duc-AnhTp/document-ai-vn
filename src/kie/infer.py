"""Inference: nhận ảnh đầu vào → chạy OCR → LayoutLMv3 → trả về dict thực thể."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from configs.layoutlmv3_config import ID2LABEL, MODEL_NAME
from configs.paths import CHECKPOINT_DIR
from src.ocr.run_paddleocr import extract_text_and_boxes
from src.utils.bbox import normalize_bbox


def load_model(
    model_dir: Optional[Path] = None,
) -> Tuple[LayoutLMv3Processor, LayoutLMv3ForTokenClassification]:
    """Tải processor và model từ checkpoint hoặc pretrained hub.

    Args:
        model_dir: Đường dẫn thư mục checkpoint. Nếu None, dùng
                   CHECKPOINT_DIR / 'best_model'. Nếu không tồn tại,
                   fallback về pretrained hub.

    Returns:
        Tuple (processor, model) đã sẵn sàng để inference.
    """
    model_path = model_dir or (CHECKPOINT_DIR / 'best_model')
    source = str(model_path) if model_path.exists() else MODEL_NAME

    processor = LayoutLMv3Processor.from_pretrained(source, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(source)
    model.eval()
    return processor, model


def group_entities(words: List[str], labels: List[str]) -> Dict[str, str]:
    """Nhóm các từ liên tiếp có cùng entity type thành chuỗi hoàn chỉnh.

    Args:
        words: Danh sách từ sau khi decode.
        labels: Danh sách nhãn BIO tương ứng.

    Returns:
        Dict ánh xạ entity_type → chuỗi value (nhiều span cách nhau bởi ' | ').
    """
    result: Dict[str, List[str]] = {}
    current_label: Optional[str] = None

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

    return {key: ' | '.join(spans) for key, spans in result.items()}


def predict_from_image(
    image_path: str,
    processor: LayoutLMv3Processor,
    model: LayoutLMv3ForTokenClassification,
) -> Dict[str, str]:
    """Trích xuất thực thể từ một ảnh hóa đơn.

    Args:
        image_path: Đường dẫn ảnh.
        processor: LayoutLMv3Processor đã khởi tạo.
        model: LayoutLMv3ForTokenClassification đã load.

    Returns:
        Dict ánh xạ entity_type → text được trích xuất.
        Trả về dict rỗng nếu OCR không nhận diện được text.
    """
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    words, raw_boxes, _ = extract_text_and_boxes(image_path)
    if not words:
        return {}

    boxes = [normalize_bbox(box, width, height) for box in raw_boxes]
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = encoding.word_ids(batch_index=0)

    decoded_labels: List[str] = []
    kept_words: List[str] = []
    prev_word_id: Optional[int] = None

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            continue
        decoded_labels.append(ID2LABEL.get(predictions[token_idx], 'O'))
        kept_words.append(words[word_id])
        prev_word_id = word_id

    return group_entities(kept_words, decoded_labels)


if __name__ == '__main__':
    sample_path = input('Nhập đường dẫn ảnh: ').strip()
    if sample_path:
        _processor, _model = load_model()
        result = predict_from_image(sample_path, _processor, _model)
        for entity, value in result.items():
            print(f'{entity}: {value}')
