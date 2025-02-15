import logging
import os
from typing import Tuple

import numpy as np
from tqdm import tqdm

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset


class ImageDatasetForSegmentation(Dataset):
    def __init__(
        self,
        root: str,
        is_train: bool = True,
        height: int = 512,
        width: int = 512,
        mean: Tuple[float] = (0.485, 0.456, 0.406),
        std: Tuple[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.resize = A.Resize(height, width)
        self.transform = A.Compose(
            (
                [
                    A.HorizontalFlip(),
                    A.CoarseDropout(),
                    A.Rotate(limit=30.0, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ]
                if is_train
                else []
            )
            + [A.Normalize(mean=mean, std=std), ToTensorV2()]
        )
        self.denormalize = A.Normalize(
            mean=[-mean_ / std_ for mean_, std_ in zip(mean, std)],
            std=[1 / std_ for std_ in std],
            max_pixel_value=1.0,
        )

        self.images, self.masks = [], []

        for filename in tqdm(os.listdir(os.path.join(root, 'images'))):
            if not os.path.exists(os.path.join(root, 'masks', filename)):
                logging.warning(f'No corresponding mask found for image {filename}.')

            image = self.load_image(os.path.join(root, 'images', filename))
            mask = self.load_mask(os.path.join(root, 'masks', filename))

            resized = self.resize(image=image, mask=mask)
            self.images.append(resized['image']), self.masks.append(resized['mask'])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(image=self.images[index], mask=self.masks[index])

    def __len__(self) -> int:
        return len(self.images)

    def load_image(self, filepath: str) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    def load_mask(self, filepath: str) -> np.ndarray:
        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) / 255

    def restore_image(self, image: torch.Tensor) -> np.ndarray:
        return self.denormalize(image=image.permute(1, 2, 0).numpy())['image']
