# Aerial Semantic Segmentation: A Production-Ready U-Net Pipeline

This project tackles **multi-class semantic segmentation of satellite/aerial imagery**—classifying each pixel into one of six land-cover categories: Building, Land, Road, Vegetation, Water, and Unlabeled. It implements two deep learning architectures (standard U-Net with attention gates and a ResNet-34-backed encoder) and wraps them in a modular, production-ready pipeline for data loading, training with checkpoint callbacks, and metric-driven evaluation.

---

## Problem Statement & Approach

Aerial imagery segmentation is crucial for urban planning, environmental monitoring, and disaster response. Manually annotating every pixel is impractical; instead, we train a convolutional neural network to learn the spatial patterns that distinguish buildings, vegetation, roads, etc.

**Why U-Net?**  
U-Net is the go-to architecture for pixel-level segmentation because it preserves spatial information through skip connections—early low-level features (edges, textures) are concatenated with deep semantic features (objects, materials) in the decoder path. This asymmetry (encoder contracts, decoder expands) is especially suited to dense predictions where output resolution matches input.

**Why two models?**  
1. **Standard Attention-Gated U-Net**: A custom 5-level encoder-decoder with learned spatial attention gates on skip connections and multi-scale feature fusion. This baseline tests architectural innovations (attention, multi-scale) without relying on pre-trained weights.
2. **ResNet-34 U-Net**: Swaps the encoder for a ResNet-34 backbone pre-trained on ImageNet. Transfer learning drastically accelerates convergence and improves generalization, especially when training data is limited.

**Why patches?**  
Large satellite images often exceed GPU memory. We split them into 256×256 tiles (patchify), process each independently, and stitch predictions back together. This also regularizes the model to local spatial patterns.

---

## System Architecture

### Data Flow
```
Raw Images (Dataset/images/*.jpg)  →  Patchify (256x256)  →  MinMax Normalize  →  Model Input
           ↓
Raw Masks (Dataset/masks/*.png)    →  Patchify (256x256)  →  RGB→Class ID     →  Model Target (One-Hot)
           ↓
         [Train/Test Split] (80/20 by default)
           ↓
      Keras Model (Dice + Focal Loss)
           ↓
      Checkpoints (best_model.h5, latest_model.h5)  +  Training Curves (loss, IoU)
```

### Module Organization

**`segmentation_pipeline/`** — Reusable core abstractions
- **`config.py`**: Dataclasses (`DatasetConfig`, `TrainingConfig`, `EvaluationConfig`) that bundle all hyperparameters and paths. Immutable configuration objects make pipelines reproducible and testable.
- **`labels.py`**: Class names, RGB color palette, and color-to-class mappings. Centralized so the model, preprocessing, and visualization all agree on semantics.
- **`data.py`**: `DatasetBuilder` class that owns image/mask I/O, patchifying, normalization (MinMaxScaler), and RGB→label conversion. Reduces custom preprocessing logic and forces reproducibility between training and inference.
- **`training.py`**: `TrainingPipeline` orchestrates dataset loading, loss/metric setup, model compilation, and fitting. Separates training orchestration from data/architecture concerns.
- **`callbacks.py`**: `build_checkpoint_callbacks` factory that returns Keras `ModelCheckpoint` callbacks for persisting best and latest models during training.
- **`evaluation.py`**: `EvaluationRunner` computes per-class IoU, pixel accuracy, and optionally saves sample visualizations. Reuses the same data preprocessing as training to ensure fair evaluation.
- **`visualization.py`**: Utility to save training curves (loss, IoU) to disk for analysis.

**`simple_multi_unet_model.py`** — Custom U-Net architecture
- Implements the attention-gated U-Net: 5 encoder levels (downsampling) feeding into 5 decoder levels (upsampling) with spatial attention gates on skip connections.
- Multi-scale feature fusion at the final decoder stage ensures the model reasons about features at different scales.
- Defines `jacard_coef` (Jaccard/IoU) as a custom metric.

**`aerial_seg.py`** — Training CLI
- Thin command-line interface that accepts configuration (data path, epochs, batch size, class weights, etc.) and orchestrates `TrainingPipeline` to train one or both models.
- Prints final metrics (loss, IoU) to console.
- Automatically saves training curves to `outputs/training/{run_name}/`.

**`evaluate_model.py`** — Evaluation CLI
- Loads a saved model and computes metrics on a test split using the same preprocessing as training.
- Optionally saves a random sample prediction to visualize correctness.

---

## Key Design Decisions

### 1. Patchify + MinMax Normalization
- **Why MinMax (0–1) over standardization?** MinMax is simpler, avoids negative values (which can confuse some activation functions), and is deterministic across train/test.
- **Why fit on each patch?** Each patch is normalized independently to account for local brightness/contrast variations in aerial data. Naive global normalization across the entire dataset can wash out local details.

### 2. Loss Function: Dice + Focal Loss
- **Dice Loss** (soft Jaccard) handles class imbalance by penalizing false positives and false negatives symmetrically. Class weights let you emphasize rare classes (e.g., Water).
- **Focal Loss** (categorical) down-weights easy examples (high confidence, correct) and focuses on hard negatives. Combined with Dice, this dual loss encourages the model to learn difficult boundaries.

### 3. Attention Gating
- Standard skip connections indiscriminately concatenate all features. Attention gates re-weight skip features based on the gating signal from the deeper decoder level, suppressing irrelevant spatial locations. This reduces noise and speeds convergence.

### 4. Transfer Learning (ResNet Backbone)
- ImageNet pre-trained ResNet-34 provides a strong spatial feature extractor. Fine-tuning on aerial data requires fewer samples and less training time than training from scratch.
- The `segmentation_models` library abstracts this; we just swap in the backbone name.

### 5. Checkpoint Strategy
- **Best model**: Persisted based on the highest validation Jaccard coefficient. This is the model you'd deploy for production inference.
- **Latest model**: Saved every epoch, useful for debugging training dynamics or recovering from accidental crashes.

### 6. Modular Preprocessing (DatasetBuilder)
- Same preprocessing logic is shared between training and inference. This eliminates common bugs where inference preprocessing differs from training, causing domain shift.
- Encapsulation also makes it easy to swap in augmentation, re-sampling, or different normalization schemes without touching the model training code.

---

## Technical Stack

**Core Libraries:**
- **TensorFlow 2.15** + **Keras**: Deep learning framework.
- **segmentation-models 1.0.1**: Pre-built U-Net and encoder backbones.
- **OpenCV**: Efficient image I/O and color conversion.
- **scikit-learn**: Data splitting, MinMax scaling.
- **Patchify**: Grid-based image tiling.

**Azure Compatibility:**
- Pinned **protobuf 3.20.3** to match Azure's glibc/cuDNN stack (avoids symbol resolution errors).
- Tested on Azure ML compute with CPU-only inference; GPU warnings are benign.

---

## Data Expectations

### Input Format
- **Images**: RGB JPEGs in `Dataset/images/`, any size (automatically cropped to patch grid).
- **Masks**: RGB PNGs in `Dataset/masks/`, exact spatial alignment with corresponding image.
- **Color Palette** (6 classes):
  - Building: `#3C1098` (RGB 60, 16, 152)
  - Land: `#8429F6` (RGB 132, 41, 246)
  - Road: `#6EC1E4` (RGB 110, 193, 228)
  - Vegetation: `#FEDD3A` (RGB 254, 221, 58)
  - Water: `#E2A929` (RGB 226, 169, 41)
  - Unlabeled: `#9B9B9B` (RGB 155, 155, 155)

### Processing Pipeline
1. Read image/mask pairs.
2. Crop to nearest multiple of 256×256 patch size.
3. Extract non-overlapping 256×256 patches.
4. **Image patches**: Normalize each with MinMaxScaler (0–1 range).
5. **Mask patches**: Convert RGB to integer class ID (0–5) using color map.
6. Expand mask class IDs to one-hot encoding (0–5 → 6-dimensional binary vector).
7. Shuffle and split 80/20 for train/test.

---

## Metrics & Evaluation

### Jaccard Coefficient (IoU)
$$
\text{IoU} = \frac{\text{Intersection}}{\text{Union}} = \frac{TP}{TP + FP + FN}
$$
Per-class and mean IoU are reported during and after training.

### Pixel Accuracy
Simple fraction of correctly classified pixels. Less sensitive to class imbalance than IoU.

### Monitoring During Training
- `val_jacard_coef`: Average IoU on validation set every epoch. Used to select best model.
- `val_loss`: Combined Dice + Focal loss on validation set.

---

## Inference & Deployment

For **live demo or production**, reuse the preprocessing from `segmentation_pipeline/data.py`:

1. Load the saved model: `tf.keras.models.load_model('models/standard_unet/best_model.h5', custom_objects={'jacard_coef': jacard_coef})`
2. Preprocess incoming image: use `DatasetBuilder._split_into_patches` and `_normalize_patch` to maintain consistency with training.
3. Run inference on each patch: `model.predict(patch_batch)` → argmax → class ID per pixel.
4. Stitch patches back into full-image prediction.
5. Optionally overlay on original image or save as PNG with color palette.

Common deployment targets:
- **FastAPI** (Python REST service) → run inference server-side.
- **TensorFlow Serving** (Docker microservice) → optimized C++ inference engine.
- **TensorFlow.js** (browser) → only if model is small and you port preprocessing to JavaScript.

---

## Quick Start (For Those Who Want to Run It)

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
python aerial_seg.py --data-root Dataset --epochs 100 --batch-size 16
```
Models and curves saved to `models/` and `outputs/training/`.

### Evaluation
```bash
python evaluate_model.py --model-path models/standard_unet/best_model.h5 --sample-plot
```
Results printed to console; optional visualization saved to `outputs/evaluation/sample_result.png`.

---

## Future Directions

1. **Class Balancing**: Current class weights are uniform. Imbalanced datasets (e.g., lots of grass, few buildings) could benefit from adaptive weighting.
2. **Data Augmentation**: Rotation, flip, jitter to increase effective training set size.
3. **Ensembling**: Combine predictions from both architectures for robustness.
4. **Instance Segmentation**: If you need to count distinct buildings/lakes, add a branch for instance boundaries.
5. **Multi-Temporal**: Incorporate temporal sequences (change detection over time).
6. **Quantization / Pruning**: Reduce model size and latency for edge deployment.

---

## Troubleshooting & Notes

- **Out of memory during training?** Reduce batch size (`--batch-size 8`) or patch size.
- **Poor validation accuracy?** Check that train and test images are not too different in distribution; consider augmentation.
- **Slow inference?** Use the ResNet-backed model with a smaller backbone (e.g., ResNet18) or batch predictions.
- **CUDA warnings on CPU-only Azure?** Benign; suppress with `export TF_CPP_MIN_LOG_LEVEL=2`.

---

## Citation & Acknowledgments

**U-Net Paper:**
```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```

**Segmentation Models Library:**
[GitHub: qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)

---

## License

This project is provided as-is for research and educational purposes.
