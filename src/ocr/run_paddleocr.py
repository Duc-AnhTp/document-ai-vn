from typing import List, Tuple

from paddleocr import PaddleOCR


ocr = PaddleOCR(use_angle_cls=True, lang='vi')



def extract_text_and_boxes(image_path: str) -> Tuple[List[str], List[List[int]], List[float]]:
    result = ocr.ocr(image_path)
    texts = []
    boxes = []
    scores = []

    if not result or not result[0]:
        return texts, boxes, scores

    for item in result[0]:
        polygon, (text, score) = item
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

        texts.append(text)
        boxes.append(box)
        scores.append(float(score))

    return texts, boxes, scores
