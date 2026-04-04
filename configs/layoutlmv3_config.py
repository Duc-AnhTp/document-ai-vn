"""Cấu hình hyperparameter và label schema cho LayoutLMv3."""

# --- Model ---
MODEL_NAME = 'microsoft/layoutlmv3-base'
MAX_SEQ_LEN = 512
IMAGE_SIZE = 224

# --- Training ---
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

# --- Label Schema (BIO format) ---
# Thứ tự quan trọng: 'O' phải là index 0
LABEL_LIST = [
    'O',
    'B-SELLER',       'I-SELLER',
    'B-SELLER_ADDRESS', 'I-SELLER_ADDRESS',
    'B-TIMESTAMP',    'I-TIMESTAMP',
    'B-TOTAL_COST',   'I-TOTAL_COST',
    'B-SIGNATURE',    'I-SIGNATURE',
]

NUM_LABELS = len(LABEL_LIST)
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
