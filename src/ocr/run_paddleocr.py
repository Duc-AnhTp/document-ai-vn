"""PaddleOCR wrapper: phát hiện và nhận dạng text tiếng Việt từ ảnh."""

import logging
import os
from typing import List, Tuple

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

# Lazy initialization: chỉ khởi tạo khi thực sự cần dùng
_ocr_instance: PaddleOCR | None = None


def _get_ocr() -> PaddleOCR:
    """Lấy singleton PaddleOCR instance (lazy init tránh side-effect khi import)."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang='vi')
    return _ocr_instance


def extract_text_and_boxes(
    image_path: str,
) -> Tuple[List[str], List[List[int]], List[float]]:
    """Chạy OCR trên ảnh và trả về text, bounding boxes, confidence scores.

    Args:
        image_path: Đường dẫn tới file ảnh.

    Returns:
        Tuple gồm (texts, boxes, scores). Trả về 3 list rỗng nếu có lỗi.
    """
    empty_result: Tuple[List[str], List[List[int]], List[float]] = ([], [], [])

    if not os.path.exists(image_path):
        logger.error("Không tìm thấy file ảnh: %s", image_path)
        return empty_result

    try:
        result = _get_ocr().ocr(image_path)
    except Exception:
        logger.exception("PaddleOCR gặp lỗi khi xử lý ảnh: %s", image_path)
        return empty_result

    if not result or not result[0]:
        return empty_result

    texts: List[str] = []
    boxes: List[List[int]] = []
    scores: List[float] = []

    for item in result[0]:
        polygon, (text, score) = item
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

        texts.append(text)
        boxes.append(box)
        scores.append(float(score))

    return texts, boxes, scores

