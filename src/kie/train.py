"""Script huấn luyện LayoutLMv3 cho bài toán KIE (Key Information Extraction)."""

from transformers import (
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)

from configs.layoutlmv3_config import (
    EVAL_BATCH_SIZE,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MODEL_NAME,
    NUM_EPOCHS,
    NUM_LABELS,
    TRAIN_BATCH_SIZE,
    WEIGHT_DECAY,
)
from configs.paths import CHECKPOINT_DIR, PROCESSED_DIR
from src.kie.dataset import build_encode_fn, build_hf_dataset, get_processor
from src.kie.evaluate import compute_metrics


def main() -> None:
    """Chạy toàn bộ pipeline huấn luyện LayoutLMv3."""
    processor = get_processor()
    encode_fn = build_encode_fn(processor)

    train_ds = build_hf_dataset(str(PROCESSED_DIR / 'train.jsonl')).map(encode_fn)
    val_ds = build_hf_dataset(str(PROCESSED_DIR / 'val.jsonl')).map(encode_fn)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True,          # Tăng tốc + tiết kiệm VRAM trên GPU T4/A100
        report_to='none',   # Tắt wandb
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR / 'best_model'))
    print(f'Best model saved to: {CHECKPOINT_DIR / "best_model"}')


if __name__ == '__main__':
    main()
