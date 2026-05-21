# Experiment notes

Nhanh `main` chi ghi chu cac thi nghiem Donut-only.

## Donut warm-up tren CORD v2

- Config: `configs/donut_cord.yaml`
- Pretrained: `naver-clova-ix/donut-base`
- Dataset: `naver-clova-ix/cord-v2`
- Selection metric: `val_loss`
- Output checkpoint: `results/e2_donut/checkpoints/cord_warmup`
- Log: `results/e2_donut/cord_warmup_log.csv`
- Trang thai: _chua cap nhat ket qua chinh thuc_

## Donut fine-tune tren MC-OCR

- Config: `configs/donut_mcocr.yaml`
- Pretrained: `results/e2_donut/checkpoints/cord_warmup`
- Dataset: `data/mc-ocr/donut_format/`
- Selection metric: `f1`
- Output checkpoint: `results/e2_donut/checkpoints/mcocr`
- Log: `results/e2_donut/training_log.csv`
- Metrics: `results/e2_donut/metrics.json`
- Trang thai: _chua cap nhat ket qua chinh thuc_

## Checklist truoc khi train

- `data/cord-v2/` da co HuggingFace dataset cache.
- `data/mc-ocr/donut_format/train/metadata.jsonl` ton tai.
- `data/mc-ocr/donut_format/val/metadata.jsonl` ton tai.
- `data/mc-ocr/donut_format/test/metadata.jsonl` ton tai.
- Da chay EDA va kiem tra sample annotation hop ly.
