# Code Review hien trang

Review nay phan anh trang thai sau khi don `main` thanh nhanh Donut-only.

## Findings

Khong thay blocker trong scope hien tai sau dot don dep. Repo da thong nhat ve mot pipeline chinh:

```text
CORD v2 warm-up -> MC-OCR fine-tune -> MC-OCR test evaluation
```

## Rủi ro con lai

1. Chua chay train/inference thuc te trong dot sua nay, nen chua xac nhan chat luong checkpoint.
2. `.pytest_cache` trong workspace hien bi permission denied, gay warning khi chay pytest nhung khong lam test fail.
3. Data trong `data/` khong commit vao git; nguoi chay lai can tai MC-OCR/CORD va rebuild `metadata.jsonl` theo README.

## Trang thai kiem tra

- Unit tests `tests/test_utils.py`: pass.
- CLI help cho `convert_mcocr.py`, `train_donut.py`, `evaluate.py`, `download_data.py`, `eda.py`: pass.
- Grep ngoai scope trong code/docs chinh: clean.
