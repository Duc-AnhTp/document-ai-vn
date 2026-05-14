"""
Evaluate: Chay inference Donut tren test set bat ky, tinh F1/P/R.

Su dung:
    python scripts/evaluate.py --checkpoint results/e2_donut/checkpoints/mcocr --test-dir data/mc-ocr/donut_format/test/ --output results/e2_donut/metrics.json
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import compute_metrics, save_metrics, load_metadata, extract_gt_parse, parse_donut_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate Donut model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-prompt", default="<s_mcocr>")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load model
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    processor = DonutProcessor.from_pretrained(args.checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint).to(device)
    model.eval()

    # Load test data
    records = load_metadata(os.path.join(args.test_dir, "metadata.jsonl"))
    print(f"[INFO] {len(records)} test samples")

    preds, golds = [], []
    inference_times_ms = []
    prompt_ids = processor.tokenizer(args.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    for rec in tqdm(records, desc="Inference"):
        img_path = os.path.join(args.test_dir, rec["file_name"])
        gt = extract_gt_parse(rec)
        golds.append(gt)

        if not os.path.exists(img_path):
            preds.append({"store_name": "", "date": "", "total": "", "address": ""})
            continue

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            decoder_input_ids = prompt_ids.repeat(pixel_values.shape[0], 1)
            start = time.perf_counter()
            generated = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.config.decoder.max_position_embeddings,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_beams=1,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            inference_times_ms.append(elapsed_ms)

        pred_text = processor.tokenizer.decode(generated[0], skip_special_tokens=False)
        pred = parse_donut_output(pred_text, args.task_prompt)
        preds.append(pred)

    # Compute metrics
    metrics = compute_metrics(preds, golds)
    metrics["checkpoint"] = args.checkpoint
    metrics["num_samples"] = len(records)
    metrics["avg_inference_ms"] = round(sum(inference_times_ms) / len(inference_times_ms), 2) if inference_times_ms else None

    save_metrics(metrics, args.output)

    # Print
    print(f"\n{'='*40}")
    print(f"RESULTS ({args.checkpoint})")
    print(f"{'='*40}")
    print(f"F1:        {metrics['overall']['f1']}")
    print(f"Precision: {metrics['overall']['precision']}")
    print(f"Recall:    {metrics['overall']['recall']}")
    if metrics["avg_inference_ms"] is not None:
        print(f"Inference: {metrics['avg_inference_ms']} ms/sample")
    for field, m in metrics["per_field"].items():
        print(f"  {field:15s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")


if __name__ == "__main__":
    main()
