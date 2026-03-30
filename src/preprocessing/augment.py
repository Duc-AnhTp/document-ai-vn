from typing import Optional

import albumentations as A
import cv2



def build_augmentation() -> A.Compose:
    return A.Compose([
        A.Rotate(limit=3, border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.Perspective(scale=(0.02, 0.05), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ImageCompression(quality_range=(70, 95), p=0.2),
    ])



def augment_image(image, transform: Optional[A.Compose] = None):
    transform = transform or build_augmentation()
    return transform(image=image)['image']
