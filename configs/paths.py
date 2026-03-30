from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Chỉnh hai đường dẫn này khi chạy trên Colab hoặc máy cá nhân
DATA_ROOT = Path('/content/drive/MyDrive/mcocr')
IMAGE_DIR = DATA_ROOT / 'train_images'
CSV_PATH = DATA_ROOT / 'train_df.csv'

OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
PROCESSED_DIR = OUTPUT_ROOT / 'processed'
FIGURE_DIR = OUTPUT_ROOT / 'figures'
LOG_DIR = OUTPUT_ROOT / 'logs'
PREDICTION_DIR = OUTPUT_ROOT / 'predictions'
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'

for path in [OUTPUT_ROOT, PROCESSED_DIR, FIGURE_DIR, LOG_DIR, PREDICTION_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
