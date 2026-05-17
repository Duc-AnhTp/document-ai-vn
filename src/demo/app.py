"""Gradio demo for the Vietnamese Receipt KIE pipeline."""

import json
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.extractor import extract_fields, format_mcocr_output
from src.line_processing import add_line_features
from src.ocr import create_ocr_engine, run_ocr


_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = create_ocr_engine(lang="en")
    return _ocr_engine


def _public_fields(fields: dict) -> dict:
    return {key: fields.get(key, "") for key in ["SELLER", "SELLER_ADDRESS", "TIMESTAMP", "TOTAL_COST"]}


def process_image(image: np.ndarray, pipeline_mode: str):
    if image is None:
        return None, None, "{}", ""

    mode_map = {"Simple Rule": "simple_rule", "Rule Scoring": "scoring", "Classifier": "classifier"}
    extractor_type = mode_map.get(pipeline_mode, "scoring")

    import cv2
    from src.visualize import draw_field_highlight, draw_ocr_boxes

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    try:
        lines = run_ocr(tmp_path, _get_ocr_engine())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not lines:
        warning = {"warning": "No text detected in image"}
        return image, image, json.dumps(warning, ensure_ascii=False, indent=2), ""

    image_height = image.shape[0]
    lines_with_features = add_line_features(lines, image_height)
    note = None

    if extractor_type == "classifier":
        model_path = os.path.join("outputs", "models", "classifier_mc-ocr.pkl")
        if os.path.isfile(model_path):
            from src.classifier import predict_fields

            with open(model_path, "rb") as file:
                model = pickle.load(file)
            fields = predict_fields(lines_with_features, model, include_meta=True)
        else:
            fields = extract_fields(lines, image_height, mode="scoring", include_meta=True)
            note = "Classifier model not found; showing rule-scoring fallback."
    else:
        fields = extract_fields(lines, image_height, mode=extractor_type, include_meta=True)

    output_fields = _public_fields(fields)
    if note:
        output_fields["_note"] = note

    ocr_image = draw_ocr_boxes(image, lines)
    highlight_image = draw_field_highlight(image, lines_with_features, fields)
    return (
        ocr_image,
        highlight_image,
        json.dumps(output_fields, ensure_ascii=False, indent=2),
        format_mcocr_output(output_fields),
    )


def build_demo():
    import gradio as gr

    description = """
    ## Vietnamese Receipt KIE Demo

    Upload a receipt image to extract SELLER, SELLER_ADDRESS, TIMESTAMP, and TOTAL_COST.
    """
    with gr.Blocks(title="Receipt KIE Demo") as demo:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="numpy", label="Upload receipt image", sources=["upload", "clipboard"])
                pipeline_mode = gr.Dropdown(
                    choices=["Simple Rule", "Rule Scoring", "Classifier"],
                    value="Rule Scoring",
                    label="Pipeline mode",
                )
                run_btn = gr.Button("Extract", variant="primary")
            with gr.Column(scale=2):
                with gr.Tab("OCR Boxes"):
                    ocr_output = gr.Image(label="OCR Bounding Boxes", type="numpy")
                with gr.Tab("Field Highlight"):
                    highlight_output = gr.Image(label="Field Highlight", type="numpy")
        with gr.Row():
            fields_output = gr.Textbox(label="JSON result", lines=8, show_copy_button=True)
            mcocr_output = gr.Textbox(label="MC-OCR format", lines=3, show_copy_button=True)
        run_btn.click(process_image, [input_image, pipeline_mode], [ocr_output, highlight_output, fields_output, mcocr_output])
        input_image.upload(process_image, [input_image, pipeline_mode], [ocr_output, highlight_output, fields_output, mcocr_output])
    return demo


def main():
    build_demo().launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
