# Format dữ liệu Donut — gt_parse

## Format metadata.jsonl

Mỗi split (train/val/test) chứa file `metadata.jsonl`. Mỗi dòng là 1 JSON object:

```json
{
  "file_name": "image_001.jpg",
  "ground_truth": "{\"gt_parse\": {\"store_name\": \"CỬA HÀNG ABC\", \"date\": \"01/01/2021\", \"total\": \"150.000\", \"address\": \"123 Nguyễn Huệ, Q.1, TP.HCM\"}}"
}
```

### Các trường KIE

| Trường | Mô tả | Ví dụ |
|--------|--------|-------|
| `store_name` | Tên cửa hàng / đơn vị bán | CỬA HÀNG ABC |
| `date` | Ngày trên biên lai | 01/01/2021 |
| `total` | Tổng tiền | 150.000 |
| `address` | Địa chỉ cửa hàng | 123 Nguyễn Huệ, Q.1, TP.HCM |

### Lưu ý

- `ground_truth` là **chuỗi JSON lồng** (string chứa JSON), không phải object trực tiếp.
- Text đã được normalize unicode NFC.
- Trường rỗng vẫn giữ key với value `""`.

## Cấu trúc thư mục

```
data/mc-ocr/donut_format/
├── train/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── metadata.jsonl
├── val/
│   ├── ...
│   └── metadata.jsonl
└── test/
    ├── ...
    └── metadata.jsonl
```

## Script convert

```bash
python scripts/convert_mcocr.py --input data/mc-ocr/raw/ --output data/mc-ocr/donut_format/ --split-ratio 0.8 0.1 0.1
```
