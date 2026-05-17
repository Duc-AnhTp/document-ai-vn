# Huong Dan Chay Tien Xu Ly Va Train

Tai lieu nay huong dan chay pipeline tren 3 moi truong: Kaggle Notebook, Google Colab, va GPU server rieng.

## 1. Tong Quan Lenh Chinh

Chay tu thu muc goc repo:

```bash
python -m data_preparation.prepare_all
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml
python experiments/run_experiment.py --config experiments/configs/exp02_preprocess_scoring.yaml
python experiments/run_experiment.py --config experiments/configs/exp03_classifier.yaml
```

Y nghia:

```text
Exp 1: OCR + simple rule
Exp 2: OCR + preprocessing + rule scoring
Exp 3: OCR + TF-IDF + Logistic Regression classifier
```

Output sinh ra trong:

```text
data/processed/          JSON sau tien xu ly
outputs/ocr_cache/       OCR cache theo anh
outputs/predictions/     Prediction JSON
outputs/metrics/         Metric JSON
outputs/models/          Classifier .pkl
```

## 2. Chuan Bi Data

Repo ky vong raw data nam theo layout:

```text
data/
├── mc-ocr/raw/
│   ├── mcocr_train_df.csv
│   └── train_images/...
└── sroie/SROIE2019/train/
    ├── entities/
    └── img/
```

Neu chi chay MC-OCR, SROIE co the thieu; khi do chi can chay experiment voi `dataset: mc-ocr`.

Chay tien xu ly:

```bash
python -m data_preparation.prepare_all
```

Ket qua mong doi:

```text
data/processed/mc-ocr/train.json
data/processed/mc-ocr/val.json
data/processed/sroie/train.json
data/processed/sroie/val.json
```

## 3. Chay Tren Kaggle Notebook

### 3.1. Bat GPU

Trong Kaggle Notebook:

```text
Settings -> Accelerator -> GPU
```

### 3.2. Lay Source Code

Neu repo da upload vao Kaggle Dataset, copy vao working dir:

```bash
cp -r /kaggle/input/document-ai-vn/document-ai-vn /kaggle/working/document-ai-vn
cd /kaggle/working/document-ai-vn
```

Neu dung git:

```bash
cd /kaggle/working
git clone <YOUR_REPO_URL> document-ai-vn
cd document-ai-vn
```

### 3.3. Cai Dependencies

Ban CPU:

```bash
pip install -r requirements.txt
```

Neu PaddleOCR/PaddlePaddle loi GPU, cai PaddlePaddle theo wheel phu hop voi CUDA cua Kaggle. Tham khao trang cai dat PaddlePaddle chinh thuc, sau do cai lai:

```bash
pip uninstall -y paddlepaddle paddlepaddle-gpu
pip install paddleocr
```

### 3.4. Gan Data Kaggle

Neu data duoc add vao notebook tu Kaggle Dataset, tao thu muc va copy:

```bash
mkdir -p data
cp -r /kaggle/input/<YOUR_DATASET_NAME>/mc-ocr data/mc-ocr
cp -r /kaggle/input/<YOUR_DATASET_NAME>/sroie data/sroie
```

Neu chi co file zip:

```bash
mkdir -p data/mc-ocr/raw
unzip /kaggle/input/<YOUR_DATASET_NAME>/mc-ocr.zip -d data/mc-ocr/raw
```

### 3.5. Tien Xu Ly Va Chay Experiment

```bash
python -m data_preparation.prepare_all
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml
python experiments/run_experiment.py --config experiments/configs/exp02_preprocess_scoring.yaml
python experiments/run_experiment.py --config experiments/configs/exp03_classifier.yaml
```

Luu ket qua de download:

```bash
zip -r outputs_kaggle.zip outputs data/processed
```

## 4. Chay Tren Google Colab

### 4.1. Bat GPU

```text
Runtime -> Change runtime type -> GPU
```

Kiem tra GPU:

```bash
nvidia-smi
```

### 4.2. Lay Source Code

Tu Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
cp -r /content/drive/MyDrive/document-ai-vn ./document-ai-vn
cd document-ai-vn
```

Hoac clone repo:

```bash
cd /content
git clone <YOUR_REPO_URL> document-ai-vn
cd document-ai-vn
```

### 4.3. Cai Dependencies

```bash
pip install -r requirements.txt
```

Neu PaddlePaddle can GPU wheel rieng, cai theo CUDA version cua Colab:

```bash
nvidia-smi
pip uninstall -y paddlepaddle paddlepaddle-gpu
# Cai paddlepaddle-gpu theo huong dan chinh thuc neu can.
pip install paddleocr
```

### 4.4. Gan Data

Neu data nam tren Drive:

```bash
mkdir -p data
cp -r /content/drive/MyDrive/data/mc-ocr data/mc-ocr
cp -r /content/drive/MyDrive/data/sroie data/sroie
```

Neu upload zip:

```bash
mkdir -p data/mc-ocr/raw
unzip /content/mc-ocr.zip -d data/mc-ocr/raw
```

### 4.5. Chay Va Luu Ket Qua

```bash
python -m data_preparation.prepare_all
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml
python experiments/run_experiment.py --config experiments/configs/exp02_preprocess_scoring.yaml
python experiments/run_experiment.py --config experiments/configs/exp03_classifier.yaml
```

Luu ve Drive:

```bash
cp -r outputs /content/drive/MyDrive/document-ai-vn-outputs
cp -r data/processed /content/drive/MyDrive/document-ai-vn-processed
```

## 5. Chay Tren GPU Server Ngoai

### 5.1. Tao Moi Truong

```bash
git clone <YOUR_REPO_URL> document-ai-vn
cd document-ai-vn
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Kiem tra GPU:

```bash
nvidia-smi
python - <<'PY'
try:
    import paddle
    print("Paddle compiled with CUDA:", paddle.is_compiled_with_cuda())
except Exception as exc:
    print("Paddle check failed:", exc)
PY
```

### 5.2. Dong Bo Data

Vi du dung `rsync`:

```bash
mkdir -p data
rsync -av /path/to/mc-ocr data/
rsync -av /path/to/sroie data/
```

Hoac giai nen:

```bash
mkdir -p data/mc-ocr/raw
unzip mc-ocr.zip -d data/mc-ocr/raw
```

### 5.3. Chay Bang Terminal

```bash
python -m data_preparation.prepare_all
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml
python experiments/run_experiment.py --config experiments/configs/exp02_preprocess_scoring.yaml
python experiments/run_experiment.py --config experiments/configs/exp03_classifier.yaml
```

Chay rieng SROIE:

```bash
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml --dataset sroie
```

Chay ca hai dataset:

```bash
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml --dataset all
```

### 5.4. Chay Nen Bang `tmux`

```bash
tmux new -s receipt-kie
source .venv/bin/activate
python experiments/run_experiment.py --config experiments/configs/exp03_classifier.yaml
```

Thoat tam thoi:

```text
Ctrl+B, sau do bam D
```

Quay lai:

```bash
tmux attach -t receipt-kie
```

## 6. Cache, Train Lai, Va Reset Ket Qua

Mac dinh config da bat:

```yaml
use_ocr_cache: true
force_ocr: false
error_threshold: 0.30
```

Muon OCR lai tu dau, sua config:

```yaml
force_ocr: true
```

Hoac xoa cache:

```bash
rm -rf outputs/ocr_cache
```

Muon train lai classifier:

```bash
rm -f outputs/models/classifier_mc-ocr.pkl
python experiments/run_experiment.py --config experiments/configs/exp03_classifier.yaml
```

## 7. Chay Demo

Local/server co GUI:

```bash
python -m src.demo.app
```

Mo:

```text
http://localhost:7860
```

Tren GPU server remote, forward port:

```bash
ssh -L 7860:localhost:7860 user@server
python -m src.demo.app
```

Sau do mo tren may local:

```text
http://localhost:7860
```

## 8. Loi Thuong Gap

### Thieu `cv2`, `paddleocr`, `sklearn`, `gradio`

```bash
pip install -r requirements.txt
```

### PaddleOCR tu tai model lau

Lan dau chay OCR, PaddleOCR co the tai model. Hay dam bao notebook/server co internet hoac model da duoc cache san.

### MC-OCR prepare ra 0 records

Kiem tra raw data:

```bash
ls data/mc-ocr/raw
find data/mc-ocr/raw -name "mcocr_train_df.csv"
find data/mc-ocr/raw -name "*.jpg" | head
```

Can co `mcocr_train_df.csv` va anh trong mot trong cac thu muc ma script ho tro:

```text
data/mc-ocr/raw/train_images/
data/mc-ocr/raw/train_images/train_images/
data/mc-ocr/raw/data0.7/data0.7/
data/mc-ocr/raw/kie_data/kie_data/images/
```

### Experiment dung do error rate cao

Runner se fail neu hon `error_threshold` anh loi. Xem:

```text
outputs/predictions/<experiment>_<dataset>.json
outputs/metrics/<experiment>_<dataset>.json
```

Tang nguong tam thoi trong config chi de debug:

```yaml
error_threshold: 0.80
```

## 9. Thu Tu Khuyen Nghi Cho Bao Cao

1. Chay `python -m data_preparation.prepare_all`.
2. Chay Exp 1 tren MC-OCR.
3. Chay Exp 2 tren MC-OCR.
4. Chay Exp 3 tren MC-OCR.
5. Neu con thoi gian, chay Exp 1/2 tren SROIE de so sanh generalization.
6. Luu `outputs/metrics`, `outputs/predictions`, va mot so anh visualization cho bao cao.
