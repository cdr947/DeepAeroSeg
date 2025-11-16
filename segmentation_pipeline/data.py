import os
from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np
from patchify import patchify
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

from .config import DatasetConfig, TrainingConfig
from .labels import CLASS_NAMES, COLOR_MAP


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.scaler = MinMaxScaler()

    def build(self, training_cfg: TrainingConfig) -> DatasetSplit:
        """Return normalized images and one-hot masks split into train/test."""
        images = self._load_image_patches()
        masks = self._load_mask_patches()
        labels = self._rgb_masks_to_labels(masks)
        labels_cat = to_categorical(labels, num_classes=len(CLASS_NAMES))

        X_train, X_test, y_train, y_test = train_test_split(
            images,
            labels_cat,
            test_size=training_cfg.test_split,
            random_state=training_cfg.random_state,
            shuffle=True,
        )
        return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    def _load_image_patches(self) -> np.ndarray:
        images_dir = self._ensure_dir("images")
        patches: List[np.ndarray] = []
        for path in self._iter_files(images_dir, self.config.image_exts):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {path}")
            patches.extend(self._split_into_patches(image, normalize=True))
        if not patches:
            raise RuntimeError(f"No images loaded from {images_dir}")
        return np.asarray(patches, dtype=np.float32)

    def _load_mask_patches(self) -> np.ndarray:
        masks_dir = self._ensure_dir("masks")
        patches: List[np.ndarray] = []
        for path in self._iter_files(masks_dir, self.config.mask_exts):
            mask = cv2.imread(path, cv2.IMREAD_COLOR)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {path}")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            patches.extend(self._split_into_patches(mask, normalize=False))
        if not patches:
            raise RuntimeError(f"No masks loaded from {masks_dir}")
        return np.asarray(patches, dtype=np.uint8)

    def _split_into_patches(self, array: np.ndarray, normalize: bool) -> List[np.ndarray]:
        """Crop to the nearest patch grid and return flattened patch list."""
        ps = self.config.patch_size
        height = (array.shape[0] // ps) * ps
        width = (array.shape[1] // ps) * ps
        if width == 0 or height == 0:
            raise ValueError("Image dimensions smaller than patch size")

        cropped = np.array(Image.fromarray(array).crop((0, 0, width, height)))
        patch_grid = patchify(cropped, (ps, ps, 3), step=ps).reshape(-1, ps, ps, 3)

        if normalize:
            return [self._normalize_patch(patch) for patch in patch_grid]
        return [patch.astype(np.uint8) for patch in patch_grid]

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        # Flatten -> scale to 0-1 -> reshape to original patch shape.
        scaled = self.scaler.fit_transform(patch.reshape(-1, patch.shape[-1]))
        return scaled.reshape(patch.shape).astype(np.float32)

    def _iter_files(self, directory: str, extensions: Iterable[str]) -> Iterable[str]:
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(tuple(extensions)):
                yield os.path.join(directory, name)

    def _ensure_dir(self, dirname: str) -> str:
        path = os.path.join(self.config.root_dir, dirname)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        return path

    @staticmethod
    def _rgb_masks_to_labels(masks: np.ndarray) -> np.ndarray:
        # Map RGB colors to integer class ids for each pixel.
        label_maps = []
        for mask in masks:
            label = np.zeros(mask.shape[:2], dtype=np.uint8)
            for idx, color in enumerate(COLOR_MAP):
                matches = np.all(mask == color, axis=-1)
                label[matches] = idx
            label_maps.append(label)
        labels = np.asarray(label_maps, dtype=np.uint8)
        return np.expand_dims(labels, axis=-1)
