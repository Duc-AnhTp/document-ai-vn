"""HuggingFace Dataset builder và encoder cho LayoutLMv3."""

from typing import Callable, Dict, List

from datasets import Dataset
from transformers import LayoutLMv3Processor

from configs.layoutlmv3_config import LABEL2ID, MODEL_NAME
from src.utils.io import load_jsonl


def get_processor() -> LayoutLMv3Processor:
    """Tải LayoutLMv3Processor từ pretrained hub.

    Returns:
        Processor với apply_ocr=False (OCR do PaddleOCR đảm nhận).
    """
    return LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)


def load_records(jsonl_path: str) -> List[Dict]:
    """Đọc file JSONL và trả về danh sách record.

    Args:
        jsonl_path: Đường dẫn tới file .jsonl.

    Returns:
        Danh sách dict, mỗi dict là một sample.
    """
    return load_jsonl(jsonl_path)


def build_hf_dataset(jsonl_path: str) -> Dataset:
    """Xây dựng HuggingFace Dataset từ file JSONL.

    Args:
        jsonl_path: Đường dẫn tới file .jsonl.

    Returns:
        HuggingFace Dataset object.
    """
    records = load_records(jsonl_path)
    return Dataset.from_list(records)


def build_encode_fn(processor: LayoutLMv3Processor) -> Callable:
    """Tạo hàm encode sử dụng processor đã cho (closure pattern).

    Tránh khởi tạo processor ở module-level để không gây side-effect
    khi import và không bị lỗi serialization trong multiprocessing.

    Args:
        processor: LayoutLMv3Processor đã được khởi tạo từ bên ngoài.

    Returns:
        Hàm encode nhận một example dict và trả về encoding tensor.
    """
    def encode_example(example: Dict) -> Dict:
        from PIL import Image

        image = Image.open(example['image_path']).convert('RGB')
        word_labels = [LABEL2ID[label] for label in example['labels']]

        return processor(
            image,
            example['words'],
            boxes=example['boxes'],
            word_labels=word_labels,
            truncation=True,
            padding='max_length',
            max_length=512,
        )

    return encode_example
