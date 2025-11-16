#!/usr/bin/env python3
"""Evaluate a trained aerial segmentation model."""

import argparse
import os

os.environ.setdefault('SM_FRAMEWORK', 'tf.keras')

from segmentation_pipeline import DatasetConfig, EvaluationConfig, TrainingConfig
from segmentation_pipeline.evaluation import EvaluationRunner


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model.')
	parser.add_argument('--model-path', required=True, help='Path to the trained Keras model (.h5 or SavedModel).')
	parser.add_argument('--data-root', default='Dataset', help='Root folder with images/ and masks/.')
	parser.add_argument('--patch-size', type=int, default=256, help='Patch size used during training.')
	parser.add_argument('--batch-size', type=int, default=8, help='Batch size for inference.')
	parser.add_argument('--test-split', type=float, default=0.2, help='Test split ratio (must match training).')
	parser.add_argument('--random-state', type=int, default=42, help='Random seed for split.')
	parser.add_argument('--sample-plot', action='store_true', help='Save a random test sample visualization.')
	return parser.parse_args()


def main():
	args = parse_args()

	if not os.path.exists(args.model_path):
		raise FileNotFoundError(f'Model not found at {args.model_path}')

	dataset_cfg = DatasetConfig(root_dir=args.data_root, patch_size=args.patch_size)
	training_cfg = TrainingConfig(test_split=args.test_split, random_state=args.random_state)
	eval_cfg = EvaluationConfig(batch_size=args.batch_size, sample_plot=args.sample_plot)

	runner = EvaluationRunner(dataset_cfg, training_cfg, eval_cfg)
	result = runner.evaluate(args.model_path)

	print(f"Model: {args.model_path}")
	print(f"Test samples: {result.samples}")
	print(f"Mean IoU: {result.mean_iou:.4f}")
	print(f"Pixel accuracy: {result.accuracy:.4f}")
	print('Per-class IoU:')
	for name, score in result.per_class_iou.items():
		print(f"  {name:<12} {score:.4f}")


if __name__ == '__main__':
	main()
