# Ghi chú Thí nghiệm

## E1 — PaddleOCR Baseline

- **Ngày chạy:** _chưa chạy_
- **Hyperparameters:** N/A (rule-based)
- **Ghi chú:** ...
- **Kết quả:** Xem `results/e1_baseline/metrics.json`

---

## E2 — Donut Fine-tune (Mô hình chính)

### Warm-up CORD
- **Config:** `configs/donut_cord.yaml`
- **Epochs:** 3
- **Ghi chú:** ...

### Fine-tune MC-OCR
- **Config:** `configs/donut_mcocr.yaml`
- **Epochs:** 30 (early stopping patience=5)
- **Best epoch:** _chưa chạy_
- **Ghi chú:** ...
- **Kết quả:** Xem `results/e2_donut/metrics.json`

---

## E3 — Cross-dataset SROIE

- **Config:** `configs/donut_sroie.yaml`
- **Ghi chú:** ...
- **Error analysis:** Xem `results/e3_cross/error_analysis.json`

---

## Lỗi thường gặp & cách fix

| Lỗi | Nguyên nhân | Cách fix |
|-----|-------------|----------|
| ... | ... | ... |
