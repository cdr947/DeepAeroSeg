import os
from typing import Dict, Optional

import numpy as np
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam

from simple_multi_unet_model import jacard_coef, multi_unet_model

from .callbacks import build_checkpoint_callbacks
from .config import DatasetConfig, TrainingConfig
from .data import DatasetBuilder, DatasetSplit
from .visualization import save_training_curves


class TrainingPipeline:
    def __init__(self, dataset_cfg: DatasetConfig, training_cfg: TrainingConfig):
        self.dataset_cfg = dataset_cfg
        self.training_cfg = training_cfg
        self.dataset_builder = DatasetBuilder(dataset_cfg)
        self.losses = self._build_losses()
        self.metrics = ["accuracy", jacard_coef]

    def run(self, train_standard: bool = True, train_resnet: bool = True) -> Dict[str, Dict]:
        split = self.dataset_builder.build(self.training_cfg)
        histories: Dict[str, Dict] = {}

        if train_standard:
            histories["standard_unet"] = self._train_standard_unet(split)
        if train_resnet:
            histories["resnet34_unet"] = self._train_resnet_unet(split)
        return histories

    def _build_losses(self):
        dice_loss = sm.losses.DiceLoss(class_weights=self.training_cfg.class_weights)
        focal_loss = sm.losses.CategoricalFocalLoss()
        return dice_loss + focal_loss

    def _train_standard_unet(self, split: DatasetSplit) -> Dict:
        IMG_HEIGHT = split.X_train.shape[1]
        IMG_WIDTH = split.X_train.shape[2]
        IMG_CHANNELS = split.X_train.shape[3]

        model = multi_unet_model(
            n_classes=len(self.training_cfg.class_weights),
            IMG_HEIGHT=IMG_HEIGHT,
            IMG_WIDTH=IMG_WIDTH,
            IMG_CHANNELS=IMG_CHANNELS,
        )
        model.compile(optimizer=Adam(), loss=self.losses, metrics=self.metrics)
        callbacks = build_checkpoint_callbacks("standard_unet", self.training_cfg.monitor_metric, self.training_cfg.monitor_mode)

        history = model.fit(
            split.X_train,
            split.y_train,
            validation_data=(split.X_test, split.y_test),
            batch_size=self.training_cfg.batch_size,
            epochs=self.training_cfg.epochs,
            shuffle=False,
            verbose=1,
            callbacks=callbacks,
        )

        self._persist_model(model, "standard_unet")
        save_training_curves(history.history, "standard_unet")
        return {"history": history.history, "split": split}

    def _train_resnet_unet(self, split: DatasetSplit) -> Dict:
        preprocess_input = sm.get_preprocessing(self.training_cfg.backbone)
        X_train_prepr = preprocess_input(split.X_train)
        X_test_prepr = preprocess_input(split.X_test)

        model = sm.Unet(
            self.training_cfg.backbone,
            encoder_weights="imagenet",
            classes=len(self.training_cfg.class_weights),
            activation="softmax",
        )
        model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=self.metrics)
        callbacks = build_checkpoint_callbacks("resnet34_unet", self.training_cfg.monitor_metric, self.training_cfg.monitor_mode)

        history = model.fit(
            X_train_prepr,
            split.y_train,
            validation_data=(X_test_prepr, split.y_test),
            batch_size=self.training_cfg.batch_size,
            epochs=self.training_cfg.epochs,
            shuffle=False,
            verbose=1,
            callbacks=callbacks,
        )

        self._persist_model(model, "resnet34_unet")
        save_training_curves(history.history, "resnet34_unet")
        return {"history": history.history}

    @staticmethod
    def _persist_model(model, run_name: str):
        out_dir = os.path.join("models", run_name)
        os.makedirs(out_dir, exist_ok=True)
        model.save(os.path.join(out_dir, "final_model.h5"))
