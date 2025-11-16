from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetConfig:
    root_dir: str = "Dataset"
    patch_size: int = 256
    image_exts: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    mask_exts: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 100
    test_split: float = 0.2
    random_state: int = 42
    backbone: str = "resnet34"
    monitor_metric: str = "val_jacard_coef"
    monitor_mode: str = "max"
    class_weights: List[float] = field(default_factory=lambda: [0.1666] * 6)


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    sample_plot: bool = True
    output_dir: str = "outputs/evaluation"
