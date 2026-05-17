"""
line_processing.py — Sắp xếp OCR lines và thêm layout features.
"""
from typing import List, Dict


def sort_lines(lines: List[Dict]) -> List[Dict]:
    """
    Sắp xếp danh sách OCR lines theo thứ tự đọc tự nhiên:
    từ trên xuống dưới (y), nếu y gần nhau thì từ trái sang phải (x).

    Dùng ngưỡng y_thresh để nhóm các dòng cùng hàng ngang.
    """
    if not lines:
        return []

    # Tính chiều cao trung bình để làm ngưỡng nhóm hàng
    avg_h = sum(l["bbox"][3] - l["bbox"][1] for l in lines) / len(lines)
    y_thresh = avg_h * 0.5

    # Sort sơ bộ theo y_center rồi x1
    lines = sorted(lines, key=lambda l: (
        round((l["bbox"][1] + l["bbox"][3]) / 2 / y_thresh),
        l["bbox"][0]
    ))
    return lines


def add_line_features(lines: List[Dict], image_height: int) -> List[Dict]:
    """
    Thêm layout features vào mỗi line sau khi đã sort.

    Features thêm vào:
        line_id     — index thứ tự dòng (0-based)
        x_center    — tọa độ x trung tâm bbox
        y_center    — tọa độ y trung tâm bbox
        width       — chiều rộng bbox
        height      — chiều cao bbox
        relative_y  — y_center / image_height (0.0 = đầu, 1.0 = cuối)
        text_lower  — text chuyển về lowercase

    Args:
        lines: List OCR lines (đã sort).
        image_height: Chiều cao ảnh (pixels).

    Returns:
        Danh sách lines đã thêm features.
    """
    lines = sort_lines(lines)

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line["bbox"]
        y_center = (y1 + y2) / 2

        line["line_id"] = i
        line["x_center"] = (x1 + x2) / 2
        line["y_center"] = y_center
        line["width"] = x2 - x1
        line["height"] = y2 - y1
        line["relative_y"] = y_center / image_height if image_height > 0 else 0.0
        line["text_lower"] = line["text"].lower()

    return lines
