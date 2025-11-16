import os
from typing import List

from tensorflow.keras.callbacks import Callback, ModelCheckpoint


def build_checkpoint_callbacks(run_name: str, monitor: str = "val_jacard_coef", mode: str = "max") -> List[Callback]:
    run_dir = os.path.join("models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    best_path = os.path.join(run_dir, "best_model.h5")
    latest_path = os.path.join(run_dir, "latest_model.h5")

    best_ckpt = ModelCheckpoint(
        filepath=best_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    latest_ckpt = ModelCheckpoint(
        filepath=latest_path,
        monitor=monitor,
        mode=mode,
        save_best_only=False,
        save_weights_only=False,
        verbose=0,
    )

    return [best_ckpt, latest_ckpt]
