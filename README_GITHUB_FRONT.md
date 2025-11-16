# Aerial Semantic Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)

Pixel-level semantic segmentation of aerial/satellite imagery into six land-cover classes using U-Net and ResNet-34 backbones. Production-ready pipeline with modular data loading, training orchestration, and evaluation tools.

**Key Features:**
- ✅ Two architectures: attention-gated U-Net + ResNet-34 U-Net
- ✅ 256×256 patch-based processing for large-scale imagery
- ✅ Modular pipeline (reusable for inference, augmentation, etc.)
- ✅ Checkpoint callbacks + training curve visualization
- ✅ Azure ML compatible; CPU & GPU support
- ✅ ~200 lines production-ready core code

## Quick Start

**1. Clone & Install**
```bash
git clone https://github.com/yourusername/aerial-segmentation.git
cd aerial-segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Prepare Data**
Place images in `Dataset/images/*.jpg` and masks in `Dataset/masks/*.png` (RGB, color-coded by class).

**3. Train**
```bash
python aerial_seg.py --data-root Dataset --epochs 100 --batch-size 16
```

**4. Evaluate**
```bash
python evaluate_model.py --model-path models/standard_unet/best_model.h5 --sample-plot
```

For a deeper dive into architecture, design decisions, and deployment strategies, see [README.md](README.md).

## Class Legend

| Class      | RGB Code  | Purpose                       |
|------------|-----------|-------------------------------|
| Building   | #3C1098   | Structures, rooftops           |
| Land       | #8429F6   | Bare soil, dirt               |
| Road       | #6EC1E4   | Paved surfaces, streets       |
| Vegetation | #FEDD3A   | Trees, grass, crops           |
| Water      | #E2A929   | Rivers, lakes, oceans         |
| Unlabeled  | #9B9B9B   | Unknown or mixed pixels       |

## Project Structure

```
.
├── aerial_seg.py                    # Training CLI
├── evaluate_model.py                # Evaluation CLI
├── simple_multi_unet_model.py       # Custom U-Net architecture
├── segmentation_pipeline/           # Reusable pipeline components
│   ├── config.py                    # Configuration dataclasses
│   ├── labels.py                    # Class names & color palette
│   ├── data.py                      # Data loading & preprocessing
│   ├── training.py                  # Training orchestration
│   ├── evaluation.py                # Evaluation metrics & visualization
│   ├── callbacks.py                 # Model checkpoint callbacks
│   └── visualization.py             # Training curve plots
├── Dataset/                         # Data directory (create this)
│   ├── images/                      # Raw satellite images
│   └── masks/                       # Ground truth RGB masks
├── models/                          # Saved checkpoints (auto-created)
├── outputs/                         # Training curves & evaluation plots
├── requirements.txt                 # Python dependencies
└── README.md                        # Detailed documentation
```

## Technical Highlights

- **Architecture**: U-Net with spatial attention gates & multi-scale fusion
- **Loss**: Dice + Categorical Focal Loss (handles class imbalance)
- **Preprocessing**: MinMax normalization per-patch, patchify strategy
- **Metrics**: IoU (Jaccard), pixel accuracy, per-class breakdown
- **Deployment**: Inference helpers for FastAPI/TensorFlow Serving

## Inference Example

```python
import os
os.environ.setdefault('SM_FRAMEWORK', 'tf.keras')

from tensorflow.keras.models import load_model
from segmentation_pipeline import DatasetConfig
from segmentation_pipeline.data import DatasetBuilder
from simple_multi_unet_model import jacard_coef
import numpy as np

# Load model
model = load_model('models/standard_unet/best_model.h5', 
                   custom_objects={'jacard_coef': jacard_coef})

# Prepare image
cfg = DatasetConfig()
builder = DatasetBuilder(cfg)
patches = builder._split_into_patches(image_array, normalize=True)

# Predict
predictions = model.predict(np.array(patches))
class_ids = np.argmax(predictions, axis=-1)
```

## Performance (Example)

On test split (80/20):
- **Mean IoU**: 0.73
- **Pixel Accuracy**: 0.85
- **Per-class IoU**: Building (0.68), Land (0.75), Road (0.71), Vegetation (0.82), Water (0.69), Unlabeled (0.75)

(Actual metrics depend on your dataset and training configuration.)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [segmentation-models](https://github.com/qubvel/segmentation_models) by Pavel Yakubovskiy
- U-Net architecture by Ronneberger et al. (2015)
- Dataset inspiration from [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/)

---

**Questions?** Open an issue or check the [detailed README](README.md) for architecture & design decisions.
