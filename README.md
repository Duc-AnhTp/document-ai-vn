# Vietnamese Receipt KIE Pipeline

OCR-based pipeline for extracting four receipt fields:

```text
SELLER | SELLER_ADDRESS | TIMESTAMP | TOTAL_COST
```

The main dataset is MC-OCR2021. SROIE2019 is supported as a secondary dataset for generalization checks.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare Data

```bash
python data_preparation/download_mcocr.py
python -m data_preparation.prepare_all
```

Processed files are written to `data/processed/`. The data directory is gitignored.

## Run Experiments

```bash
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml
python experiments/run_experiment.py --config experiments/configs/exp02_preprocess_scoring.yaml
python experiments/run_experiment.py --config experiments/configs/exp03_classifier.yaml
```

Useful options:

```bash
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml --dataset sroie
python experiments/run_experiment.py --config experiments/configs/exp01_simple_rule.yaml --dataset all
```

OCR cache, predictions, metrics, and classifier models are stored under `outputs/`, which is gitignored.

## Run Demo

```bash
python -m src.demo.app
```

Open `http://localhost:7860`.

## Project Layout

```text
data_preparation/      Data conversion and normalization
src/                   OCR, extraction, classifier, evaluation, demo
experiments/configs/   YAML experiment configs
outputs/               Runtime outputs and OCR cache
tests/                 Lightweight regression tests
```
