"""
Train Donut for the main branch baseline.

Usage:
    python scripts/train_donut.py --config configs/donut_cord.yaml
    python scripts/train_donut.py --config configs/donut_mcocr.yaml
"""

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import (
    compute_metrics,
    load_config,
    parse_donut_output,
    serialize_donut_parse,
)


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def unwrap_model(model):
    """Return the underlying HF model when wrapped by DataParallel."""
    return model.module if hasattr(model, "module") else model


def resolve_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), 0
    return torch.device("cuda"), torch.cuda.device_count()


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def checkpoint_paths(ckpt_dir):
    last_dir = os.path.join(ckpt_dir, "last")
    state_path = os.path.join(last_dir, "trainer_state.pt")
    return last_dir, state_path


def save_model_bundle(model, processor, path):
    os.makedirs(path, exist_ok=True)
    base_model = unwrap_model(model)
    base_model.save_pretrained(path)
    processor.save_pretrained(path)


def save_training_state(model, processor, optimizer, scaler, path, epoch, best_score, no_improve):
    save_model_bundle(model, processor, path)
    state = {
        "epoch": epoch,
        "best_score": best_score,
        "no_improve": no_improve,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, os.path.join(path, "trainer_state.pt"))


def load_training_state(path, optimizer, scaler, device):
    state_path = os.path.join(path, "trainer_state.pt")
    if not os.path.isfile(state_path):
        return 0, None, 0
    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", 0)), state.get("best_score"), int(state.get("no_improve", 0))


class DonutLocalDataset(Dataset):
    """Dataset loaded from local Donut metadata.jsonl files."""

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
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


class DonutHFDataset(Dataset):
    """Dataset loaded from HuggingFace, used for CORD v2 warm-up."""

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


def train_one_epoch(model, dataloader, optimizer, device, grad_accum=1, precision="fp32", scaler=None):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(progress(dataloader, desc="Train")):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast_context(device, precision):
            outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

    if len(dataloader) % grad_accum != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, processor, dataloader, device, task_prompt="", metric_mode="kie_f1", precision="fp32"):
    model.eval()
    total_loss = 0
    preds, golds = [], []
    base_model = unwrap_model(model)

    prompt_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    for batch in progress(dataloader, desc="Val"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast_context(device, precision):
            outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

        if metric_mode == "kie_f1":
            batch_size = pixel_values.shape[0]
            decoder_input_ids = prompt_ids.repeat(batch_size, 1)

            generated = base_model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=base_model.config.decoder.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_beams=1,
            )

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
    device, gpu_count = resolve_device()
    print(f"[INFO] Device: {device} | GPUs: {gpu_count}")
    print(f"[INFO] Experiment: {config['experiment']}")

    from transformers import DonutProcessor, VisionEncoderDecoderModel

    pretrained = config["model"]["pretrained"]
    ckpt_dir = config["output"]["checkpoint_dir"]
    last_dir, _ = checkpoint_paths(ckpt_dir)
    resume = config["training"].get("resume", False)
    resume_from = config["training"].get("resume_from") or (last_dir if resume else "")
    load_path = resume_from if resume_from and os.path.isdir(resume_from) else pretrained
    print(f"[INFO] Loading model: {load_path}")
    processor = DonutProcessor.from_pretrained(load_path)
    model = VisionEncoderDecoderModel.from_pretrained(load_path)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("[INFO] Enabled gradient checkpointing")

    added_tokens = config["model"].get("added_tokens", [])
    if added_tokens:
        processor.tokenizer.add_tokens(added_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        print(f"[INFO] Added {len(added_tokens)} tokens")

    model.to(device)
    use_multi_gpu = bool(config["training"].get("multi_gpu", True))
    if device.type == "cuda" and gpu_count > 1 and use_multi_gpu:
        model = torch.nn.DataParallel(model)
        print(f"[INFO] DataParallel enabled on {gpu_count} GPUs")

    task_prompt = config["model"]["task_prompt"]
    max_length = config["model"]["max_length"]

    if "dataset_name" in config["data"]:
        from datasets import load_dataset, load_from_disk

        local_path = config["data"].get("local_path", "")
        if local_path and os.path.exists(local_path):
            print(f"[INFO] Loading dataset from local cache: {local_path}")
            ds = load_from_disk(local_path)
        else:
            print(f"[INFO] Loading dataset from HuggingFace: {config['data']['dataset_name']}")
            ds = load_dataset(config["data"]["dataset_name"])
        train_ds = DonutHFDataset(ds["train"], processor, max_length, task_prompt)
        val_ds = DonutHFDataset(ds["validation"], processor, max_length, task_prompt)
    else:
        train_ds = DonutLocalDataset(config["data"]["train_dir"], processor, max_length, task_prompt)
        val_ds = DonutLocalDataset(config["data"]["val_dir"], processor, max_length, task_prompt)

    bs = config["training"]["batch_size"]
    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = bool(config["training"].get("pin_memory", device.type == "cuda"))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")

    lr = float(config["training"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    grad_accum = config["training"].get("gradient_accumulation", 1)
    patience = config["training"].get("early_stopping_patience", 999)
    selection_metric = config["training"].get("selection_metric", "f1")
    metric_mode = config["training"].get("metric_mode", "kie_f1")
    precision = config["training"].get("precision", "fp32").lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("training.precision must be one of: fp32, fp16, bf16")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and precision == "fp16"))
    if precision != "fp32":
        print(f"[INFO] Mixed precision: {precision}")

    log_file = config["output"]["log_file"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    best_score = None
    no_improve = 0
    start_epoch = 1
    if resume_from and os.path.isdir(resume_from):
        loaded_epoch, best_score, no_improve = load_training_state(resume_from, optimizer, scaler, device)
        start_epoch = loaded_epoch + 1
        print(f"[INFO] Resume state: epoch={loaded_epoch}, next_epoch={start_epoch}")

    append_log = start_epoch > 1 and os.path.isfile(log_file)
    with open(log_file, "a" if append_log else "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        if not append_log:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_f1", "val_precision", "val_recall"])

        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            print(f"\n--- Epoch {epoch}/{config['training']['epochs']} ---")

            train_loss = train_one_epoch(model, train_dl, optimizer, device, grad_accum, precision, scaler)
            val_loss, val_metrics = validate(model, processor, val_dl, device, task_prompt, metric_mode, precision)

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
                save_model_bundle(model, processor, ckpt_dir)
                if selection_metric == "f1":
                    print(f"  [SAVE] Best model F1={f1:.4f} -> {ckpt_dir}")
                else:
                    print(f"  [SAVE] Best model val_loss={val_loss:.4f} -> {ckpt_dir}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    save_training_state(model, processor, optimizer, scaler, last_dir, epoch, best_score, no_improve)
                    print(f"  [STOP] Early stopping (patience={patience})")
                    break
            save_training_state(model, processor, optimizer, scaler, last_dir, epoch, best_score, no_improve)
            print(f"  [SAVE] Last training state -> {last_dir}")

    if selection_metric == "f1":
        print(f"\n[DONE] Best F1={best_score:.4f} | Log: {log_file} | Checkpoint: {ckpt_dir}")
    else:
        print(f"\n[DONE] Best val_loss={-best_score:.4f} | Log: {log_file} | Checkpoint: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Donut")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from output.checkpoint_dir/last")
    parser.add_argument("--resume-from", default=None, help="Resume from a specific checkpoint directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config["training"]["resume"] = True
    if args.resume_from:
        config["training"]["resume_from"] = args.resume_from
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    run_training(config)


if __name__ == "__main__":
    main()
