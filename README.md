# Document AI VN - Donut Fine-tuning cho bien lai tieng Viet

> Do an thi giac may tinh: fine-tune mo hinh Donut de trich xuat thong tin tu anh bien lai tieng Viet trong MC-OCR 2021.

Nhanh gon: nhanh `main` chi giu baseline Donut-only. Cac huong pipeline khac va so sanh voi nhanh moi duoc thuc hien tren `new-document-ai`.

## Muc tieu

- Warm-up Donut tren CORD v2.
- Fine-tune tiep tren MC-OCR 2021.
- Evaluate tren MC-OCR test split voi 4 truong KIE:
  `store_name`, `date`, `total`, `address`.

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu dung GPU, hay cai ban PyTorch phu hop voi CUDA tren may truoc khi cai cac goi con lai.

## Chuan bi du lieu

`data/` khong commit vao git. Co the dung script de tai lai va build lai processed data.

### 1. Tai MC-OCR va CORD

```powershell
python scripts/download_data.py --dataset mcocr --output data/
python scripts/download_data.py --dataset cord --output data/
```

MC-OCR tai tu Kaggle, can cai `kaggle` CLI va dat credential tai `C:\Users\<user>\.kaggle\kaggle.json`.

### 2. Convert MC-OCR sang Donut format

```powershell
python scripts/convert_mcocr.py --input data/mc-ocr/raw/ --output data/mc-ocr/donut_format/ --split-ratio 0.8 0.1 0.1
```

Output chuan:

```text
data/mc-ocr/donut_format/
  train/
    metadata.jsonl
    *.jpg
  val/
    metadata.jsonl
    *.jpg
  test/
    metadata.jsonl
    *.jpg
```

Moi dong `metadata.jsonl` co `file_name` va `ground_truth`, trong do `ground_truth` la chuoi JSON long chua `gt_parse`.

### 3. Kiem tra nhanh du lieu

```powershell
python scripts/eda.py --data-dir data/mc-ocr/donut_format/ --output docs/eda_figures/
```

Lenh nay tao cac hinh thong ke split, field coverage, do dai text, kich thuoc anh va mau annotation.

## Train Donut

### 1. Warm-up CORD v2

```powershell
python scripts/train_donut.py --config configs/donut_cord.yaml
```

Checkpoint duoc luu tai:

```text
results/e2_donut/checkpoints/cord_warmup
```

### 2. Fine-tune MC-OCR

```powershell
python scripts/train_donut.py --config configs/donut_mcocr.yaml
```

Checkpoint duoc luu tai:

```text
results/e2_donut/checkpoints/mcocr
```

### Train tren Kaggle T4 x2

Kaggle nen dung 2 config rieng vi output phai nam trong `/kaggle/working` va data thuong nam trong `/kaggle/input`.

1. Bat accelerator `GPU T4 x2`.
2. Upload repo nay vao Kaggle Notebook, hoac clone repo trong notebook.
3. Tao Kaggle Dataset chua MC-OCR Donut format theo cau truc:

```text
/kaggle/input/mcocr-donut-format/donut_format/
  train/metadata.jsonl
  val/metadata.jsonl
  test/metadata.jsonl
```

Neu dataset slug khac `mcocr-donut-format`, sua cac duong dan `data.*_dir` trong `configs/donut_mcocr_kaggle.yaml`.

Chay warm-up CORD:

```bash
python scripts/train_donut.py --config configs/donut_cord_kaggle.yaml
```

Chay fine-tune MC-OCR:

```bash
python scripts/train_donut.py --config configs/donut_mcocr_kaggle.yaml
```

Neu session bi ngat, chay lai dung lenh cu. Script se resume tu:

```text
/kaggle/working/results/e2_donut/checkpoints/<experiment>/last
```

Co the override nhanh so epoch hoac batch size:

```bash
python scripts/train_donut.py --config configs/donut_mcocr_kaggle.yaml --epochs 15 --batch-size 4
```

Script tu dong bat mixed precision `fp16`, `pin_memory`, `num_workers=2`, va `DataParallel` khi Kaggle cap 2 GPU.

## Evaluate

```powershell
python scripts/evaluate.py --checkpoint results/e2_donut/checkpoints/mcocr --test-dir data/mc-ocr/donut_format/test --output results/e2_donut/metrics.json --task-prompt "<s_mcocr>"
```

Ket qua gom:

- `overall.precision`
- `overall.recall`
- `overall.f1`
- `per_field`
- `avg_inference_ms`

## Cau truc repo

```text
document-ai-vn/
  configs/
    donut_cord.yaml
    donut_mcocr.yaml
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
    eda_figures/
  notebooks/
  results/
  PROJECT.md
  README.md
  requirements.txt
```

## Ghi chu ve pham vi

- `main` chi giu pipeline Donut-only.
- Cac pipeline khac dung de so sanh se nam tren nhanh rieng, uu tien `new-document-ai`.

## Tham khao

- [Donut: OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664)
- [Donut GitHub](https://github.com/clovaai/donut)
- [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
- [MC-OCR 2021](https://kaggle.com/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021)
