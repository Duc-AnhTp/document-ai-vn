from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Chỉnh đường dẫn gốc thành cục bộ thuộc project (thay vì phụ thuộc Google Drive)
DATA_ROOT = PROJECT_ROOT / 'data' / 'mcocr'
IMAGE_DIR = DATA_ROOT / 'train_images'
CSV_PATH = DATA_ROOT / 'train_df.csv'

# Tự động tạo thư mục rỗng nếu ai đó vừa clone repo về chưa kịp tải data
DATA_ROOT.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
PROCESSED_DIR = OUTPUT_ROOT / 'processed'
FIGURE_DIR = OUTPUT_ROOT / 'figures'
LOG_DIR = OUTPUT_ROOT / 'logs'
PREDICTION_DIR = OUTPUT_ROOT / 'predictions'
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'

for path in [OUTPUT_ROOT, PROCESSED_DIR, FIGURE_DIR, LOG_DIR, PREDICTION_DIR, CHECKPOINT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
