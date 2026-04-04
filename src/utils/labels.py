"""Định nghĩa các nhãn thực thể và mapping chuẩn hóa cho MC-OCR dataset."""

# Các nhãn thực thể hợp lệ trong dataset
VALID_ENTITY_LABELS = {
    'SELLER',
    'SELLER_ADDRESS',
    'TIMESTAMP',
    'TOTAL_COST',
    'OTHER',
}

# Mapping chuẩn hóa: sửa các nhãn bị gõ sai / không đồng nhất
LABEL_NORMALIZATION_MAP = {
    'TIMESTAMPS': 'TIMESTAMP',
    'TIME_STAMP': 'TIMESTAMP',
    'SELLER_ADDR': 'SELLER_ADDRESS',
    'SELLERADDRESS': 'SELLER_ADDRESS',
    'TOTAL': 'TOTAL_COST',
    'TOTALCOST': 'TOTAL_COST',
    'TOTAL_COSTS': 'TOTAL_COST',
    'SELLERS': 'SELLER',
    'STORE': 'SELLER',
    'ADDRESS': 'SELLER_ADDRESS',
}
