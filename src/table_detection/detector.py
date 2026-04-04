import os
from typing import List

from paddleocr import PPStructure

# Khởi tạo PP-Structure chuyên biệt cho nhận diện bảng biểu
table_engine = PPStructure(table=True, lang='vi', show_log=False)

def extract_table(image_path: str) -> List[str]:
    """
    Trích xuất bảng biểu từ ảnh biên lai, hóa đơn.
    Sử dụng PP-Structure để phân tích kết cấu và trả về danh sách các bảng dạng HTML.
    
    Args:
        image_path (str): Đường dẫn tới ảnh cần xử lý.
        
    Returns:
        List[str]: Danh sách các chuỗi HTML tương ứng với các bảng biểu tìm thấy.
    """
    if not os.path.exists(image_path):
        return []

    result = table_engine(image_path)
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

if __name__ == '__main__':
    sample_path = input('Nhập đường dẫn ảnh kiểm tra table: ').strip()
    if sample_path:
        tables = extract_table(sample_path)
        print(f"Tìm thấy {len(tables)} bảng.")
        for i, html in enumerate(tables):
            print(f"--- Bảng {i+1} ---")
            print(html)
