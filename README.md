# MMX

*Empowering early detection, inspiring hope.*

MammoMix is a deep learning-based object detection project for identifying breast cancer in medical images. It leverages state-of-the-art models and libraries for robust and reproducible results.

## Features

- Object detection for breast cancer in medical images
- Utilizes PyTorch, HuggingFace Transformers, and TIMM
- Experiment tracking with Weights & Biases (wandb)
- Evaluation metrics with torchmetrics

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/MammoMix.git
cd MammoMix
pip install -r requirements.txt
```

## Usage

### Train YOLOS

```bash
python train.py --config configs/train_config.yaml --dataset CSAW
python train.py --config configs/train_config.yaml --dataset DMID
python train.py --config configs/train_config.yaml --dataset DDSM
```

### Train Deformable DETR

```bash
python train_detrd.py --config configs/config_deformable_detr.yaml --dataset CSAW
python train_detrd.py --config configs/config_deformable_detr.yaml --dataset DMID
python train_detrd.py --config configs/config_deformable_detr.yaml --dataset DDSM
```

### Test YOLOS

```bash
python train.py --config configs/train_config.yaml --dataset CSAW --phase 2 --test
python train.py --config configs/train_config.yaml --dataset DMID --phase 2 --test
python train.py --config configs/train_config.yaml --dataset DDSM --phase 2 --test
```

### Test Deformable DETR

```bash
python train_detrd.py --config configs/config_deformable_detr.yaml --dataset CSAW
# (After training, test results will be printed automatically)
```

## Requirements

- Python 3.8+
- See `requirements.txt` for Python dependencies

## Citation

If you use this project, please cite:

```
@misc{mammomix2025,
  title={MammoMix: Breast Cancer Object Detection},
  author={UTS},
  year={2025},
  url={https://github.com/tommyngx/MammoMix}
}
```
