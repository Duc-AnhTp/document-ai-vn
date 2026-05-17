"""PaddleOCR wrapper plus optional JSON cache helpers."""

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional


def create_ocr_engine(lang: str = "en", use_angle_cls: bool = True):
    """Create a PaddleOCR instance."""
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)


def run_ocr(image_path: str, ocr_engine) -> List[Dict]:
    """Run OCR and return normalized line dictionaries."""
    result = ocr_engine.ocr(image_path, cls=True)
    lines = []
    if not result:
        return lines

    for page in result:
        if not page:
            continue
        for item in page:
            box = item[0]
            text = item[1][0]
            score = item[1][1]
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            lines.append({
                "text": text,
                "conf": float(score),
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            })
    return lines


def make_cache_key(image_path: str, image_id: Optional[str] = None) -> str:
    """Build a stable cache key for one image."""
    if image_id:
        return image_id.replace("/", "_").replace("\\", "_")
    normalized = os.path.abspath(image_path).replace("\\", "/").lower()
    stem = os.path.splitext(os.path.basename(image_path))[0] or "image"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def cache_path(cache_dir: str, dataset_name: str, image_path: str, image_id: Optional[str] = None) -> str:
    return os.path.join(cache_dir, dataset_name, f"{make_cache_key(image_path, image_id)}.json")


def run_ocr_cached(
    image_path: str,
    ocr_engine,
    dataset_name: str = "default",
    image_id: Optional[str] = None,
    cache_dir: str = os.path.join("outputs", "ocr_cache"),
    use_cache: bool = True,
    force_ocr: bool = False,
) -> List[Dict]:
    """Run OCR using a per-image JSON cache when enabled."""
    path = cache_path(cache_dir, dataset_name, image_path, image_id)
    if use_cache and not force_ocr and os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    lines = run_ocr(image_path, ocr_engine)
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(lines, file, ensure_ascii=False, indent=2)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Run OCR on one image")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Output JSON path; stdout if omitted")
    parser.add_argument("--lang", default="en", help="OCR language")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    engine = create_ocr_engine(lang=args.lang)
    lines = run_ocr(args.image, engine)
    output = json.dumps(lines, ensure_ascii=False, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(output)
        print(f"Saved {len(lines)} OCR lines -> {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
