"""Nhận diện bảng biểu trong ảnh hóa đơn bằng PP-Structure."""

import argparse
import logging
import os
from typing import List

from paddleocr import PPStructure

logger = logging.getLogger(__name__)

# Khởi tạo PP-Structure chuyên biệt cho nhận diện bảng biểu
table_engine = PPStructure(table=True, lang='vi', show_log=False)


def extract_table(image_path: str) -> List[str]:
    """Trích xuất bảng biểu từ ảnh biên lai, hóa đơn.

    Sử dụng PP-Structure để phân tích kết cấu và trả về danh sách các bảng dạng HTML.

    Args:
        image_path: Đường dẫn tới ảnh cần xử lý.

    Returns:
        Danh sách các chuỗi HTML tương ứng với các bảng biểu tìm thấy.
    """
    if not os.path.exists(image_path):
        logger.error("Không tìm thấy file ảnh: %s", image_path)
        return []

    try:
        result = table_engine(image_path)
    except Exception:
        logger.exception("Lỗi khi xử lý ảnh bằng PP-Structure: %s", image_path)
        return []

    if not result:
        return []

    html_tables = []
    # PP-Structure trả về list các region (text, table, figure...)
    for region in result:
        if region['type'] == 'table':
            # region['res'] chứa thông tin trả về tuỳ loại, với 'table' nó chứa 'html'
            html = region['res'].get('html')
            if html:
                html_tables.append(html)

    return html_tables


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments cho table detection script."""
    parser = argparse.ArgumentParser(
        description='Nhận diện bảng biểu trong ảnh hóa đơn bằng PP-Structure.',
    )
    parser.add_argument(
        '--image', required=True, type=str,
        help='Đường dẫn tới ảnh cần kiểm tra bảng biểu.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = parse_args()
    tables = extract_table(args.image)
    logger.info("Tìm thấy %d bảng.", len(tables))
    for idx, html in enumerate(tables):
        print(f"--- Bảng {idx + 1} ---")
        print(html)

