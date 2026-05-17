"""End-to-end orchestration for receipt KIE."""

import os
import tempfile
from typing import Dict, Optional

import cv2

from src.extractor import extract_fields
from src.ocr import run_ocr, run_ocr_cached
from src.preprocessing import preprocess_image


EMPTY_FIELDS = {"SELLER": "", "SELLER_ADDRESS": "", "TIMESTAMP": "", "TOTAL_COST": ""}


def _image_height(image_path: str) -> int:
    img = cv2.imread(image_path)
    return img.shape[0] if img is not None else 1024


def run_pipeline(
    image_path: str,
    ocr_engine,
    extractor_type: str = "scoring",
    use_preprocess: bool = False,
    classifier_model=None,
    image_height: Optional[int] = None,
    dataset_name: str = "default",
    image_id: Optional[str] = None,
    use_ocr_cache: bool = False,
    force_ocr: bool = False,
    cache_dir: str = os.path.join("outputs", "ocr_cache"),
    include_meta: bool = False,
) -> Dict[str, str]:
    """Run OCR, line processing, extraction/classification, and postprocess."""
    tmp_path = None
    ocr_input = image_path
    h = image_height

    try:
        if use_preprocess:
            image = preprocess_image(image_path)
            h = image.shape[0]
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            ocr_input = tmp_path
            if use_ocr_cache:
                cache_image_id = f"{image_id or os.path.splitext(os.path.basename(image_path))[0]}_preprocessed"
                lines = run_ocr_cached(
                    ocr_input,
                    ocr_engine,
                    dataset_name=dataset_name,
                    image_id=cache_image_id,
                    cache_dir=cache_dir,
                    use_cache=True,
                    force_ocr=force_ocr,
                )
            else:
                lines = run_ocr(ocr_input, ocr_engine)
        else:
            h = h or _image_height(image_path)
            if use_ocr_cache:
                lines = run_ocr_cached(
                    ocr_input,
                    ocr_engine,
                    dataset_name=dataset_name,
                    image_id=image_id,
                    cache_dir=cache_dir,
                    use_cache=True,
                    force_ocr=force_ocr,
                )
            else:
                lines = run_ocr(ocr_input, ocr_engine)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not lines:
        return EMPTY_FIELDS.copy()

    if extractor_type == "classifier":
        if classifier_model is None:
            raise ValueError("extractor_type='classifier' requires a trained classifier_model")
        from src.classifier import predict_fields
        from src.line_processing import add_line_features

        lines = add_line_features(lines, h or 1024)
        return predict_fields(lines, classifier_model, include_meta=include_meta)

    mode = "simple_rule" if extractor_type == "simple_rule" else "scoring"
    return extract_fields(lines, h or 1024, mode=mode, include_meta=include_meta)
