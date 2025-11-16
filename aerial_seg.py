#!/usr/bin/env python3
"""Training entrypoint for aerial semantic segmentation models."""

import argparse
import os
from typing import List

os.environ.setdefault('SM_FRAMEWORK', 'tf.keras')

from segmentation_pipeline import DatasetConfig, TrainingConfig, TrainingPipeline


def _parse_class_weights(raw: str, expected: int) -> List[float]:
    values = [float(val) for val in raw.split(',')]
    if len(values) != expected:
        raise ValueError(f'Expected {expected} class weights, got {len(values)}')
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train aerial segmentation models.')
    parser.add_argument('--data-root', default='Dataset', help='Root folder with images/ and masks/.')
    parser.add_argument('--patch-size', type=int, default=256, help='Patch size for patchify step.')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test split ratio.')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for split.')
    parser.add_argument('--backbone', default='resnet34', help='Encoder backbone for segmentation_models Unet.')
    parser.add_argument('--monitor', default='val_jacard_coef', help='Metric monitored for checkpoints.')
    parser.add_argument('--monitor-mode', default='max', choices=['min', 'max'], help='Monitor mode for checkpoints.')
    parser.add_argument('--class-weights', default='0.1666,0.1666,0.1666,0.1666,0.1666,0.1666',
                        help='Comma separated class weights for Dice loss.')
    parser.add_argument('--skip-standard', action='store_true', help='Skip training the standard attention U-Net.')
    parser.add_argument('--skip-resnet', action='store_true', help='Skip training the ResNet backbone U-Net.')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_cfg = DatasetConfig(root_dir=args.data_root, patch_size=args.patch_size)
    class_weights = _parse_class_weights(args.class_weights, expected=6)
    training_cfg = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_split=args.test_split,
        random_state=args.random_state,
        backbone=args.backbone,
        monitor_metric=args.monitor,
        monitor_mode=args.monitor_mode,
        class_weights=class_weights,
    )

    pipeline = TrainingPipeline(dataset_cfg, training_cfg)
    histories = pipeline.run(train_standard=not args.skip_standard, train_resnet=not args.skip_resnet)

    for run_name, details in histories.items():
        history = details.get('history', {})
        final_loss = history.get('loss', ['NA'])[-1]
        final_val_loss = history.get('val_loss', ['NA'])[-1]
        final_iou = history.get('jacard_coef', ['NA'])[-1]
        final_val_iou = history.get('val_jacard_coef', ['NA'])[-1]
        print(f"{run_name} → loss: {final_loss}, val_loss: {final_val_loss}, IoU: {final_iou}, val_IoU: {final_val_iou}")


if __name__ == '__main__':
    main()

