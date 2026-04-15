"""Utility functions cho preprocessing: bbox, I/O."""

import json
from pathlib import Path
from typing import Dict, List


def polygon_to_bbox(polygon: List) -> List[int]:
    """Chuyển polygon (list tọa độ) thành bounding box [x_min, y_min, x_max, y_max].

    Hỗ trợ cả 2 dạng:
      - List phẳng: [x1, y1, x2, y2, ...]
      - List of pairs: [[x1, y1], [x2, y2], ...]
    """
    if not polygon:
        return [0, 0, 0, 0]

    if isinstance(polygon[0], (list, tuple)):
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
    else:
        x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] for i in range(1, len(polygon), 2)]

    return [
        int(min(x_coords)), int(min(y_coords)),
        int(max(x_coords)), int(max(y_coords)),
    ]


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """Normalize bbox về khoảng [0, 1000] (chuẩn LayoutLM).

    Args:
        bbox: [x_min, y_min, x_max, y_max] tọa độ pixel.
        width: Chiều rộng ảnh gốc.
        height: Chiều cao ảnh gốc.

    Returns:
        Bbox đã normalize về [0, 1000].
    """
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]


def save_jsonl(data: List[Dict], output_path: Path) -> None:
    """Lưu list of dict thành file JSONL (mỗi dòng = 1 JSON object).

    Args:
        data: Danh sách dict cần lưu.
        output_path: Đường dẫn file output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
