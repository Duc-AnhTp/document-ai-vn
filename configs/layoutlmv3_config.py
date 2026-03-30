MODEL_NAME = 'microsoft/layoutlmv3-base'
MAX_SEQ_LEN = 512
IMAGE_SIZE = 224
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

LABEL_LIST = [
    'O',
    'B-SELLER', 'I-SELLER',
    'B-SELLER_ADDRESS', 'I-SELLER_ADDRESS',
    'B-TIMESTAMP', 'I-TIMESTAMP',
    'B-TOTAL_COST', 'I-TOTAL_COST'
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
