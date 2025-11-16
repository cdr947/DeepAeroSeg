import os
from typing import Dict

import matplotlib.pyplot as plt


def save_training_curves(history: Dict[str, list], run_name: str):
    out_dir = os.path.join("outputs", "training", run_name)
    os.makedirs(out_dir, exist_ok=True)

    if 'loss' in history and 'val_loss' in history:
        plt.figure()
        epochs = range(1, len(history['loss']) + 1)
        plt.plot(epochs, history['loss'], 'y', label='Training loss')
        plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'loss.png'))
        plt.close()

    if 'jacard_coef' in history and 'val_jacard_coef' in history:
        plt.figure()
        epochs = range(1, len(history['jacard_coef']) + 1)
        plt.plot(epochs, history['jacard_coef'], 'y', label='Training IoU')
        plt.plot(epochs, history['val_jacard_coef'], 'r', label='Validation IoU')
        plt.title('Training vs Validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'iou.png'))
        plt.close()
