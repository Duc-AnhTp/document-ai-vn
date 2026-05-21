"""
Visualize Decoder Cross-Attention for Donut Model.

Usage:
    python scripts/visualize_attention.py \
        --checkpoint results/e2_donut/checkpoints/mcocr \
        --image data/mc-ocr/donut_format/test/some_image.jpg \
        --output-dir results/attention_viz/
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from transformers import DonutProcessor, VisionEncoderDecoderModel


def main():
    parser = argparse.ArgumentParser(description="Visualize Donut Cross-Attention")
    parser.add_argument("--checkpoint", default="results/e2_donut/checkpoints/mcocr", help="Path to checkpoint directory")
    parser.add_argument("--image", required=True, help="Path to input receipt image")
    parser.add_argument("--output-dir", default="results/attention_viz", help="Output directory to save attention maps")
    parser.add_argument("--task-prompt", default="<s_mcocr>", help="Prompt used to trigger generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load processor and model
    print(f"[INFO] Loading model and processor from {args.checkpoint}...")
    processor = DonutProcessor.from_pretrained(args.checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint).to(device)
    model.eval()

    # Load and process image
    print(f"[INFO] Loading image from {args.image}...")
    image = Image.open(args.image).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate sequence with cross attentions enabled
    print(f"[INFO] Running generation...")
    prompt_ids = processor.tokenizer(args.task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    decoder_input_ids = prompt_ids.repeat(pixel_values.shape[0], 1)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.config.decoder.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    prompt_len = decoder_input_ids.shape[1]
    
    # cross_attentions: tuple of steps. Each step: tuple of layers.
    # shape of each layer tensor: (batch_size, num_heads, seq_len_decoder, seq_len_encoder_pixels)
    cross_attentions = outputs.cross_attentions

    if cross_attentions is None:
        print("[ERROR] Model did not return cross attentions. Make sure the model configuration supports attentions.")
        return

    # Suffix tokens generated after the prompt
    generated_steps = len(cross_attentions)
    print(f"[INFO] Generated {generated_steps} tokens.")

    # Dictionary to store step attention vectors per KIE field
    field_attentions = {
        "store_name": [],
        "date": [],
        "total": [],
        "address": []
    }
    
    current_field = None

    for step_idx in range(generated_steps):
        token_id = generated_ids[prompt_len + step_idx].item()
        token_str = processor.tokenizer.decode([token_id]).strip()

        # Handle field tags
        if token_str in ["<s_store_name>", "<s_date>", "<s_total>", "<s_address>"]:
            current_field = token_str[3:-1]  # Extract field name
            continue
        elif token_str in ["</s_store_name>", "</s_date>", "</s_total>", "</s_address>"]:
            current_field = None
            continue

        if current_field is not None:
            # Average attention across all heads and all layers
            # Shape of layer tensor: (1, num_heads, 1, S)
            layers_attn = []
            for layer in range(len(cross_attentions[step_idx])):
                tensor = cross_attentions[step_idx][layer]
                # Average over heads (dim 1) and squeeze batch (dim 0) & decoder seq len (dim 2)
                layer_mean = tensor.mean(dim=1).squeeze(0).squeeze(0).cpu().numpy()
                layers_attn.append(layer_mean)
            
            step_attn_mean = np.mean(layers_attn, axis=0) # shape (S,)
            field_attentions[current_field].append(step_attn_mean)

    # Compute grid size from actual attention tensor shape
    _, _, h_pixel, w_pixel = pixel_values.shape
    S = cross_attentions[0][0].shape[-1]  # encoder sequence length
    h_grid = int(np.sqrt(S * h_pixel / w_pixel))
    w_grid = S // h_grid
    print(f"[INFO] Image resolution: {w_pixel}x{h_pixel} | Encoder tokens: {S} | Grid size: {w_grid}x{h_grid}")

    # Generate overlay for each field
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    img_np = np.array(image).astype(np.float32) / 255.0

    visualized_count = 0

    for field, attns in field_attentions.items():
        if not attns:
            print(f"[INFO] Field '{field}' not found or has no content tokens. Skipping visualization.")
            continue
        
        # Average attention over all content tokens of this field
        field_attn = np.mean(attns, axis=0)  # shape (S,)
        
        # Reshape to grid
        try:
            attn_map = field_attn.reshape(h_grid, w_grid)
        except ValueError as e:
            print(f"[ERROR] Could not reshape attention of size {field_attn.size} to {h_grid}x{w_grid}: {e}")
            continue

        # Resize back to original image dimensions
        attn_resized = cv2.resize(attn_map, (image.width, image.height), interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

        # Apply colormap
        cmap = plt.colormaps.get_cmap("jet")
        heatmap = cmap(attn_norm)[:, :, :3]  # shape (H, W, 3) in [0, 1]

        # Blend original image and heatmap
        alpha = 0.45
        overlay = img_np * (1 - alpha) + heatmap * alpha
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        # Save output image
        output_file = os.path.join(args.output_dir, f"{base_name}_attn_{field}.png")
        Image.fromarray(overlay_uint8).save(output_file)
        print(f"[SAVED] {output_file}")
        visualized_count += 1

    if visualized_count > 0:
        print(f"[OK] Successfully visualized {visualized_count} fields in {args.output_dir}")
    else:
        print("[WARN] No fields were visualized. Verify that the task prompt and model output contain the target tags.")


if __name__ == "__main__":
    main()
