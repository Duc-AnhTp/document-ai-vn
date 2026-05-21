# Code Review — main branch (Donut-only)

Review cap nhat sau dot don dep nhanh `main`.

## Trang thai

Nhanh `main` da duoc don thanh pipeline Donut-only duy nhat:

```text
CORD v2 warm-up -> MC-OCR fine-tune -> MC-OCR test evaluation
```

## Nhung gi da lam

1. Xoa cac script ngoai scope: `baseline_paddleocr.py`, `convert_sroie.py`, `train_donut_sroie.py`, `visualize_attention.py`.
2. Xoa config ngoai scope: `donut_sroie.yaml`.
3. Thu gon `PROJECT.md`, `README.md`, `requirements.txt`, `docs/` chi giu noi dung Donut-only.
4. Don dep data local: xoa `data/processed/` va `data/sroie/` (khong dung tren main).
5. Xoa file log trong `results/e2_donut/cord_warmup_log.csv` (se duoc tao lai khi train).

## Kiem tra

- Unit tests `tests/test_utils.py`: pass.
- CLI help cho cac script chinh: pass.
- Khong con tham chieu den E1/E3/SROIE/PaddleOCR trong code.

## Rui ro con lai

1. Chua chay train/inference thuc te, nen chua xac nhan chat luong checkpoint.
2. Data `data/` khong commit vao git; nguoi chay lai can tai MC-OCR/CORD va rebuild theo README.
