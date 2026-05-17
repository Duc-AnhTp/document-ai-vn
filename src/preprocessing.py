"""
preprocessing.py — Tiền xử lý ảnh đầu vào trước khi chạy OCR.

Giữ đơn giản để không làm mất dấu tiếng Việt.
"""
import os

import cv2
import numpy as np


def preprocess_image(image_path: str, max_side: int = 1600) -> np.ndarray:
    """
    Đọc ảnh, convert sang RGB, resize nếu cạnh lớn nhất vượt max_side.

    Args:
        image_path: Đường dẫn ảnh (.jpg, .jpeg, .png).
        max_side: Kích thước cạnh tối đa sau resize.

    Returns:
        Ảnh dạng numpy array RGB (H, W, 3).

    Raises:
        FileNotFoundError: Nếu ảnh không tồn tại.
        ValueError: Nếu không đọc được ảnh.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize nếu cần
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image
