"""Utility chuyển đổi polygon → bounding box và chuẩn hóa tọa độ cho LayoutLMv3."""

from typing import List

# LayoutLMv3 sử dụng hệ tọa độ chuẩn hóa [0, 1000]
LAYOUTLM_BBOX_SCALE = 1000


def polygon_to_bbox(polygon: List) -> List[int]:
    """Chuyển polygon (danh sách tọa độ) → bbox [x_min, y_min, x_max, y_max].

    Polygon có thể ở dạng:
    - [[x1,y1], [x2,y2], ...] (list of points)
    - [x1, y1, x2, y2, ...] (flat list)
    """
    if not polygon:
        return [0, 0, 0, 0]

    # Tách tọa độ x, y tùy theo định dạng polygon
    if isinstance(polygon[0], (list, tuple)):
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
    else:
        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i] for i in range(1, len(polygon), 2)]

    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def normalize_bbox(
    bbox: List[int],
    width: int,
    height: int,
) -> List[int]:
    """Chuẩn hóa bbox về hệ [0, 1000] cho LayoutLMv3.

    Args:
        bbox: [x_min, y_min, x_max, y_max] pixel coordinates
        width: chiều rộng ảnh gốc
        height: chiều cao ảnh gốc

    Returns:
        bbox chuẩn hóa [0, 1000]
    """
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]

    x_min, y_min, x_max, y_max = bbox

    normalized = [
        int(LAYOUTLM_BBOX_SCALE * x_min / width),
        int(LAYOUTLM_BBOX_SCALE * y_min / height),
        int(LAYOUTLM_BBOX_SCALE * x_max / width),
        int(LAYOUTLM_BBOX_SCALE * y_max / height),
    ]

    # Clamp giá trị trong khoảng [0, 1000]
    return [max(0, min(LAYOUTLM_BBOX_SCALE, val)) for val in normalized]
