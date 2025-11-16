import os
import random
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model

from simple_multi_unet_model import jacard_coef

from .config import DatasetConfig, EvaluationConfig, TrainingConfig
from .data import DatasetBuilder
from .labels import CLASS_NAMES


@dataclass
class EvaluationResult:
    mean_iou: float
    accuracy: float
    per_class_iou: Dict[str, float]
    samples: int


class EvaluationRunner:
    def __init__(self, dataset_cfg: DatasetConfig, training_cfg: TrainingConfig, eval_cfg: EvaluationConfig):
        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg
        self.eval_cfg = eval_cfg
        self.dataset_builder = DatasetBuilder(dataset_cfg)

    def evaluate(self, model_path: str) -> EvaluationResult:
        split = self.dataset_builder.build(self.training_cfg)
        model = load_model(model_path, custom_objects={'jacard_coef': jacard_coef}, compile=False)
        y_pred = model.predict(split.X_test, batch_size=self.eval_cfg.batch_size, verbose=1)
        y_pred_argmax = np.argmax(y_pred, axis=-1)
        y_true_argmax = np.argmax(split.y_test, axis=-1)

        mean_iou, accuracy, per_class = self._compute_metrics(y_true_argmax, y_pred_argmax)

        if self.eval_cfg.sample_plot:
            self._save_random_sample(split.X_test, y_true_argmax, y_pred_argmax)

        return EvaluationResult(
            mean_iou=mean_iou,
            accuracy=accuracy,
            per_class_iou=per_class,
            samples=split.X_test.shape[0],
        )

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
        n_classes = len(CLASS_NAMES)
        metric = MeanIoU(num_classes=n_classes)
        metric.update_state(y_true, y_pred)
        mean_iou = float(metric.result().numpy())
        conf_matrix = metric.get_weights()[0].reshape(n_classes, n_classes)
        per_class = {}
        for idx, name in enumerate(CLASS_NAMES):
            intersection = conf_matrix[idx, idx]
            union = conf_matrix[idx, :].sum() + conf_matrix[:, idx].sum() - intersection
            per_class[name] = float(intersection / (union + 1e-7))
        accuracy = float((y_true == y_pred).mean())
        return mean_iou, accuracy, per_class

    def _save_random_sample(self, X_test, y_true, y_pred):
        idx = random.randint(0, X_test.shape[0] - 1)
        sample_img = X_test[idx]
        sample_gt = y_true[idx]
        sample_pred = y_pred[idx]

        os.makedirs(self.eval_cfg.output_dir, exist_ok=True)
        out_path = os.path.join(self.eval_cfg.output_dir, "sample_result.png")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title('Test Image')
        plt.imshow(sample_img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Ground Truth')
        plt.imshow(sample_gt, cmap='nipy_spectral')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Prediction')
        plt.imshow(sample_pred, cmap='nipy_spectral')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
