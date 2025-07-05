import argparse
import os

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import pickle
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET
import yaml
from PIL import Image
from functools import partial
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from pathlib import Path
import datetime
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)

from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_yolos_model(model_name, image_processor, model_type):
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
        ignore_mismatched_sizes=True,
    )
    return model

def MoE_architecture(models, image_processors, data_loader):
    """
    Extract features from multiple YOLOS models and prepare targets for MoE calibration.
    Returns:
        X: np.ndarray, stacked model outputs for all batches
        y: np.ndarray, binary targets for all batches
    """
    all_outputs = []
    all_targets = []
    for batch in tqdm(data_loader, desc="MoE feature extraction"):
        batch_outputs = []
        for model, processor in zip(models, image_processors):
            with torch.no_grad():
                inputs = batch['pixel_values'].to(model.device)
                outputs = model(inputs)
                batch_outputs.append(outputs.logits.cpu().numpy())
        # Stack outputs from all models
        batch_outputs = np.concatenate(batch_outputs, axis=-1)  # shape: (batch, ..., n_models*logit_dim)
        all_outputs.append(batch_outputs)
        # Assume binary label for MoE calibration (adjust as needed)
        batch_targets = [1 if any([l['class_labels'].sum().item() > 0 for l in batch['labels']]) else 0 for l in batch['labels']]
        all_targets.extend(batch_targets)
    X = np.concatenate(all_outputs, axis=0)
    y = np.array(all_targets)
    return X, y

class TopKMoE(BaseEstimator, ClassifierMixin):
    """
    Calibrated Mixture of Experts: For each sample, select top-k model predictions (by confidence)
    and use only those for the final logistic regression.
    """
    def __init__(self, k=2):
        self.k = k
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        # X: shape (n_samples, n_models)
        X_topk = self._select_topk(X)
        self.lr.fit(X_topk, y)
        return self

    def predict_proba(self, X):
        X_topk = self._select_topk(X)
        return self.lr.predict_proba(X_topk)

    def predict(self, X):
        X_topk = self._select_topk(X)
        return self.lr.predict(X_topk)

    def _select_topk(self, X):
        # X: (n_samples, n_models)
        # For each sample, select top-k values (by confidence)
        idx = np.argsort(-X, axis=1)[:, :self.k]  # indices of top-k
        X_topk = np.take_along_axis(X, idx, axis=1)
        return X_topk

def get_model_probs(models, image_processors, data_loader):
    """
    For each batch, get the predicted probability (sigmoid) from each model.
    Returns:
        probs: np.ndarray, shape (num_samples, n_models)
        targets: np.ndarray, shape (num_samples,)
    """
    all_probs = []
    all_targets = []
    for batch in tqdm(data_loader, desc="MoE feature extraction"):
        batch_probs = []
        for model, processor in zip(models, image_processors):
            with torch.no_grad():
                inputs = batch['pixel_values'].to(model.device)
                outputs = model(inputs)
                logits = outputs.logits
                # For binary classification, take sigmoid of first logit
                probs = torch.sigmoid(logits[..., 0]).cpu().numpy()
                batch_probs.append(probs)
        # shape: (n_models, batch_size)
        batch_probs = np.stack(batch_probs, axis=-1)  # (batch_size, n_models)
        all_probs.append(batch_probs)
        # Assume binary label for MoE calibration (adjust as needed)
        batch_targets = [1 if any([l['class_labels'].sum().item() > 0 for l in batch['labels']]) else 0 for l in batch['labels']]
        all_targets.extend(batch_targets)
    X = np.concatenate(all_probs, axis=0)
    y = np.array(all_targets)
    return X, y

def main(config_path, epoch=None, dataset=None, weight_dir=None):
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)

    # Get model subfolders from weight_dir
    if weight_dir is not None:
        yolos_models = []
        for sub in ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM']:
            yolos_models.append(os.path.join(weight_dir, sub))
    else:
        yolos_models = [
            config.get('moe', {}).get('model1', 'hustvl/yolos-base'),
            config.get('moe', {}).get('model2', 'hustvl/yolos-small'),
            config.get('moe', {}).get('model3', 'hustvl/yolos-tiny'),
        ]

    # Add wandb folder support
    wandb_dir = None
    if 'wandb' in config and 'wandb_dir' in config['wandb']:
        wandb_dir = config['wandb']['wandb_dir']
        print(f"Wandb directory: {wandb_dir}")

    # Use the same processor for all models (or load separately if needed)
    image_processors = [get_image_processor(m, MAX_SIZE) for m in yolos_models]
    model_types = [get_model_type(m) for m in yolos_models]
    models = [get_yolos_model(m, ip, mt) for m, ip, mt in zip(yolos_models, image_processors, model_types)]

    train_dataset = BreastCancerDataset(
        split='train',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processors[0],
        model_type=model_types[0],
        dataset_epoch=epoch
    )
    val_dataset = BreastCancerDataset(
        split='val',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processors[0],
        model_type=model_types[0],
        dataset_epoch=epoch
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f'Train loader: {len(train_dataset)} samples')
    print(f'Val loader: {len(val_dataset)} samples')
    print(f"Train loader batches: {len(train_loader)}")

    # Get model probabilities for train set
    print("Extracting YOLOS model probabilities for MoE training...")
    X_train, y_train = get_model_probs(models, image_processors, train_loader)
    print("Extracting YOLOS model probabilities for MoE validation...")
    X_val, y_val = get_model_probs(models, image_processors, val_loader)

    # Train calibrated MoE with top-2 expert selection
    print("Training calibrated MoE (Top-2 LogisticRegression)...")
    moe_model = TopKMoE(k=2)
    moe_model.fit(X_train, y_train)

    # Save the calibrated MoE model
    with open("moe_calibrator_top2.pkl", "wb") as f:
        pickle.dump(moe_model, f)
    print("MoE calibration model (top-2) saved as moe_calibrator_top2.pkl")

    # Evaluate on validation set using MoE
    print("Evaluating MoE on validation set...")
    val_probs = moe_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)
    acc = (val_preds == y_val).mean()
    print(f"MoE (top-2) validation accuracy: {acc:.4f}")

# To train the MoE model, run this script from the terminal:
# Example:
# python train_moe.py --config configs/train_config.yaml --weight_dir /path/to/weights

# Arguments:
# --config: Path to your config YAML file.
# --weight_dir: Path to directory containing yolos_CSAW, yolos_DMID, yolos_DDSM subfolders.
# --epoch: (optional) Dataset epoch value to pass to dataset.
# --dataset: (optional) Dataset name to use (overrides config).

# Example command:
# python train_moe.py --config configs/train_config.yaml --weight_dir ./weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to use (overrides config)')
    parser.add_argument('--weight_dir', type=str, default=None, help='Path to directory containing yolos_CSAW, yolos_DMID, yolos_DDSM subfolders')
    args = parser.parse_args()
    main(args.config, args.epoch, args.dataset, args.weight_dir)
