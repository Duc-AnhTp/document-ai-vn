# Document AI VN - Donut-only baseline

## Tong quan

Du an xay dung baseline Donut-only cho bai toan Key Information Extraction tren bien lai tieng Viet. Nhanh `main` chi phuc vu mot luong chinh:

```text
CORD v2 warm-up -> MC-OCR fine-tune -> MC-OCR test evaluation
```

Ket qua tren nhanh nay duoc dung lam diem tham chieu khi so sanh voi cac pipeline khac tren nhanh `new-document-ai`.

## Muc tieu va metric

| Hang muc | Muc tieu |
| --- | --- |
| Model | Donut OCR-free document understanding |
| Dataset chinh | MC-OCR 2021 |
| Warm-up | CORD v2 |
| Task | Trich xuat 4 field KIE |
| Metric | Precision, Recall, F1 macro theo field |
| Target F1 | >= 0.80 sau fine-tune MC-OCR |

Bon field duoc dung xuyen suot codebase:

- `store_name`
- `date`
- `total`
- `address`

## Dataset

### CORD v2

CORD v2 la dataset receipt tieng Anh duoc dung de warm-up Donut tu checkpoint `naver-clova-ix/donut-base`. Dataset duoc tai bang HuggingFace `datasets` va cache tai `data/cord-v2/`.

### MC-OCR 2021

MC-OCR 2021 la dataset chinh cho bien lai tieng Viet. Du lieu raw nam tai `data/mc-ocr/raw/`, sau do duoc convert sang Donut format tai `data/mc-ocr/donut_format/`.

Annotation sau convert co schema:

```json
{
  "gt_parse": {
    "store_name": "...",
    "date": "...",
    "total": "...",
    "address": "..."
  }
}
```

Script convert giu day du 4 key, ke ca khi gia tri rong. Text duoc normalize Unicode NFC de on dinh hon khi train va evaluate.

## Pipeline train

1. Tai CORD v2 va MC-OCR.
2. Convert MC-OCR sang `metadata.jsonl`.
3. Chay EDA de kiem tra split, field coverage va mau annotation.
4. Train warm-up voi `configs/donut_cord.yaml`.
5. Fine-tune MC-OCR voi `configs/donut_mcocr.yaml`.
6. Evaluate checkpoint MC-OCR tren test split.

Lenh chinh:

```powershell
python scripts/train_donut.py --config configs/donut_cord.yaml
python scripts/train_donut.py --config configs/donut_mcocr.yaml
python scripts/evaluate.py --checkpoint results/e2_donut/checkpoints/mcocr --test-dir data/mc-ocr/donut_format/test --output results/e2_donut/metrics.json --task-prompt "<s_mcocr>"
```

## Cau truc thu muc

```text
document-ai-vn/
  configs/
    donut_cord.yaml
    donut_mcocr.yaml
  data/
    cord-v2/
    mc-ocr/
      raw/
      donut_format/
        train/
        val/
        test/
  scripts/
    convert_mcocr.py
    download_data.py
    eda.py
    evaluate.py
    train_donut.py
    utils.py
  docs/
    data_format.md
    experiment_notes.md
  results/
    e2_donut/
```

`data/` va checkpoint nang khong commit vao git. `results/e2_donut/metrics.json` va log nho co the commit neu can luu ket qua bao cao.

## Bang ket qua

| Run | Train data | Test data | Precision | Recall | F1 | Avg inference |
| --- | --- | --- | --- | --- | --- | --- |
| Donut CORD warm-up + MC-OCR | CORD v2 + MC-OCR train | MC-OCR test | _chua chay_ | _chua chay_ | _chua chay_ | _chua chay_ |

## Ngoai scope cua main

Nhanh `main` chi giu pipeline Donut-only. Cac huong khac neu can so sanh se nam o nhanh rieng, uu tien `new-document-ai`.
