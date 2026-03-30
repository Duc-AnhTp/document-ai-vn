from typing import Dict, List

from datasets import Dataset
from transformers import LayoutLMv3Processor

from configs.layoutlmv3_config import LABEL2ID, MODEL_NAME
from src.utils.io import load_jsonl


processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)



def load_records(jsonl_path: str) -> List[Dict]:
    return load_jsonl(jsonl_path)



def build_hf_dataset(jsonl_path: str) -> Dataset:
    records = load_records(jsonl_path)
    return Dataset.from_list(records)



def encode_example(example):
    from PIL import Image

    image = Image.open(example['image_path']).convert('RGB')
    words = example['words']
    boxes = example['boxes']
    word_labels = [LABEL2ID[label] for label in example['labels']]

    encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding='max_length',
        max_length=512,
    )
    return encoding
