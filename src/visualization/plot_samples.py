"""Trực quan hóa mẫu dữ liệu: vẽ bounding box, biểu đồ phân bố nhãn và chất lượng ảnh."""

from collections import Counter
from typing import Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

# Bảng màu cho từng loại entity
ENTITY_COLORS = {
    'SELLER':         '#FF6B6B',
    'SELLER_ADDRESS': '#4ECDC4',
    'TIMESTAMP':      '#45B7D1',
    'TOTAL_COST':     '#FFA07A',
    'SIGNATURE':      '#A29BFE',
    'OTHER':          '#C0C0C0',
    'O':              '#E0E0E0',
}



def plot_sample_with_boxes(
    image_path: str,
    boxes: List[List[int]],
    labels: List[str],
    title: Optional[str] = None,
) -> None:
    """Vẽ ảnh kèm bounding box có nhãn, mỗi entity type một màu.

    Args:
        image_path: đường dẫn ảnh
        boxes: danh sách bbox [x_min, y_min, x_max, y_max]
        labels: danh sách nhãn tương ứng (BIO hoặc raw entity)
        title: tiêu đề biểu đồ
    """
    image = Image.open(image_path).convert('RGB')

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label in zip(boxes, labels):
        # Lấy entity name từ BIO label (VD: B-SELLER → SELLER)
        entity = label.split('-', 1)[-1] if '-' in label else label
        color = ENTITY_COLORS.get(entity, '#888888')

        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 4, label,
            fontsize=7, color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8),
        )

    ax.set_title(title or image_path, fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_label_distribution(samples: List[Dict]) -> None:
    """Vẽ histogram phân bố nhãn entity trên toàn dataset.

    Args:
        samples: danh sách sample, mỗi sample chứa key 'labels'
    """
    all_labels = []
    for sample in samples:
        for label in sample.get('labels', []):
            all_labels.append(label)

    counts = Counter(all_labels)
    labels_sorted = sorted(counts.keys())
    values = [counts[label] for label in labels_sorted]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels_sorted, values, color='#4ECDC4', edgecolor='#2C3E50')
    plt.xlabel('Entity Label')
    plt.ylabel('Số lượng')
    plt.title('Phân bố nhãn entity trong dataset')
    plt.xticks(rotation=45, ha='right')

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_quality_distribution(samples: List[Dict]) -> None:
    """Vẽ histogram phân bố chất lượng ảnh (image_quality: 0–1).

    Args:
        samples: danh sách sample, mỗi sample chứa key 'image_quality'
    """
    qualities = []
    for sample in samples:
        quality = sample.get('image_quality')
        if quality is not None:
            try:
                qualities.append(float(quality))
            except (ValueError, TypeError):
                continue

    if not qualities:
        print('Không có dữ liệu image_quality để vẽ.')
        return

    plt.figure(figsize=(10, 5))
    plt.hist(qualities, bins=20, color='#45B7D1', edgecolor='#2C3E50', alpha=0.8)
    plt.xlabel('Image Quality Score')
    plt.ylabel('Số lượng ảnh')
    plt.title('Phân bố chất lượng ảnh trong dataset')
    plt.axvline(x=0.33, color='red', linestyle='--', alpha=0.7, label='Low threshold')
    plt.axvline(x=0.66, color='orange', linestyle='--', alpha=0.7, label='High threshold')
    plt.legend()
    plt.tight_layout()
    plt.show()
