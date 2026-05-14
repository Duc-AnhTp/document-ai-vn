"""
E2: Fine-tune Donut tren CORD (warm-up) va MC-OCR (main).

Su dung:
    python scripts/train_donut.py --config configs/donut_cord.yaml
    python scripts/train_donut.py --config configs/donut_mcocr.yaml
"""

import argparse
import csv
import json
import os
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import (
    load_config,
    compute_metrics,
    parse_donut_output,
    serialize_donut_parse,
)


# ── Dataset ────────────────────────────────────────────────────────────────

class DonutLocalDataset(Dataset):
    """Dataset doc tu metadata.jsonl (MC-OCR, SROIE converted)."""

    def __init__(self, data_dir, processor, max_length=768, task_prompt="<s_mcocr>"):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.task_prompt = task_prompt

        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        self.records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.data_dir, rec["file_name"])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        gt = json.loads(rec["ground_truth"])
        gt_parse = gt.get("gt_parse", gt)
        target_text = f"{self.task_prompt}{serialize_donut_parse(gt_parse)}{self.processor.tokenizer.eos_token}"

        tokenized = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenized.input_ids.squeeze()
        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


class DonutHFDataset(Dataset):
    """Dataset doc tu HuggingFace (CORD v2)."""

    def __init__(self, hf_dataset, processor, max_length=768, task_prompt="<s_cord-v2>"):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
        self.task_prompt = task_prompt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        gt = json.loads(sample["ground_truth"])
        gt_parse = gt.get("gt_parse", gt) if isinstance(gt, dict) else gt
        target_text = f"{self.task_prompt}{serialize_donut_parse(gt_parse)}{self.processor.tokenizer.eos_token}"

        tokenized = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenized.input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Training ───────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, device, grad_accum=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Train")):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

    if len(dataloader) % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, processor, dataloader, device, task_prompt="", metric_mode="kie_f1"):
    model.eval()
    total_loss = 0
    preds, golds = [], []
    
    prompt_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    for batch in tqdm(dataloader, desc="Val"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

        batch_size = pixel_values.shape[0]
        decoder_input_ids = prompt_ids.repeat(batch_size, 1)

        # Generate
        generated = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.config.decoder.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=1,
        )

        if metric_mode == "kie_f1":
            for gen, lab in zip(generated, labels):
                pred_text = processor.tokenizer.decode(gen, skip_special_tokens=False)
                pred_dict = parse_donut_output(pred_text, task_prompt)
                preds.append(pred_dict)

                lab_ids = lab[lab != -100]
                gold_text = processor.tokenizer.decode(lab_ids, skip_special_tokens=False)
                gold_dict = parse_donut_output(gold_text, task_prompt)
                golds.append(gold_dict)

    avg_loss = total_loss / len(dataloader)
    if metric_mode == "kie_f1":
        metrics = compute_metrics(preds, golds)
    else:
        metrics = {"overall": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "per_field": {}}
    return avg_loss, metrics


def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Experiment: {config['experiment']}")

    # Load model
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    pretrained = config["model"]["pretrained"]
    print(f"[INFO] Loading model: {pretrained}")
    processor = DonutProcessor.from_pretrained(pretrained)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained)

    # Add tokens
    added_tokens = config["model"].get("added_tokens", [])
    if added_tokens:
        processor.tokenizer.add_tokens(added_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        print(f"[INFO] Added {len(added_tokens)} tokens")

    model.to(device)

    # Load dataset
    task_prompt = config["model"]["task_prompt"]
    max_length = config["model"]["max_length"]

    if "dataset_name" in config["data"]:
        # HuggingFace dataset (CORD)
        from datasets import load_dataset
        ds = load_dataset(config["data"]["dataset_name"])
        train_ds = DonutHFDataset(ds["train"], processor, max_length, task_prompt)
        val_ds = DonutHFDataset(ds["validation"], processor, max_length, task_prompt)
    else:
        # Local dataset (MC-OCR)
        train_ds = DonutLocalDataset(config["data"]["train_dir"], processor, max_length, task_prompt)
        val_ds = DonutLocalDataset(config["data"]["val_dir"], processor, max_length, task_prompt)

    bs = config["training"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    grad_accum = config["training"].get("gradient_accumulation", 1)
    patience = config["training"].get("early_stopping_patience", 999)
    selection_metric = config["training"].get("selection_metric", "f1")
    metric_mode = config["training"].get("metric_mode", "kie_f1")

    # Training loop
    ckpt_dir = config["output"]["checkpoint_dir"]
    log_file = config["output"]["log_file"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    best_score = None
    no_improve = 0

    with open(log_file, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_f1", "val_precision", "val_recall"])

        for epoch in range(1, config["training"]["epochs"] + 1):
            print(f"\n--- Epoch {epoch}/{config['training']['epochs']} ---")

            train_loss = train_one_epoch(model, train_dl, optimizer, device, grad_accum)
            val_loss, val_metrics = validate(model, processor, val_dl, device, task_prompt, metric_mode)

            f1 = val_metrics["overall"]["f1"]
            p = val_metrics["overall"]["precision"]
            r = val_metrics["overall"]["recall"]
            score = f1 if selection_metric == "f1" else -val_loss

            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}"])
            csvf.flush()

            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

            if best_score is None or score > best_score:
                best_score = score
                no_improve = 0
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                if selection_metric == "f1":
                    print(f"  [SAVE] Best model F1={f1:.4f} -> {ckpt_dir}")
                else:
                    print(f"  [SAVE] Best model val_loss={val_loss:.4f} -> {ckpt_dir}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  [STOP] Early stopping (patience={patience})")
                    break

    if selection_metric == "f1":
        print(f"\n[DONE] Best F1={best_score:.4f} | Log: {log_file} | Checkpoint: {ckpt_dir}")
    else:
        print(f"\n[DONE] Best val_loss={-best_score:.4f} | Log: {log_file} | Checkpoint: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Donut (E2)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
