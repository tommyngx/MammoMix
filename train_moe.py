import argparse
import os
import pickle
import datetime
from pathlib import Path

# Suppress TensorFlow and CUDA warnings - Enhanced
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Redirect stderr to suppress cuDNN/cuBLAS warnings
import sys
import warnings

class DevNull:
    def write(self, msg):
        # Only suppress lines containing specific CUDA warnings
        if any(x in msg for x in ["Unable to register cuDNN", "Unable to register cuBLAS", "All log messages before absl"]):
            return
        sys.__stderr__.write(msg)
    def flush(self):
        sys.__stderr__.flush()

# Apply stderr filter
sys.stderr = DevNull()
warnings.filterwarnings("ignore")

import numpy as np
import yaml
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)

from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type

# ================================
# CONFIGURATION & UTILITIES
# ================================

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_yolos_model(model_name, image_processor, model_type):
    """Load a YOLOS model for object detection."""
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
        ignore_mismatched_sizes=True,
    )
    return model

# ================================
# MOE COMPONENTS
# ================================

class GatingNetwork(nn.Module):
    """
    Gating network that learns to select the best expert for each input.
    Takes model predictions as input and outputs expert weights.
    """
    def __init__(self, n_models=3, hidden_dim=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_models),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_probs):
        """Forward pass through gating network."""
        return self.gate(expert_probs)

class IntegratedMoE(nn.Module):
    """
    Integrated MoE model that contains all expert models and gating network.
    During training, only the gating network is trained while experts are frozen.
    """
    def __init__(self, expert_models, n_models=3, hidden_dim=16):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)
        self.gate = GatingNetwork(n_models, hidden_dim)
        
        # Freeze expert models
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values):
        """Forward pass through integrated MoE."""
        # Get predictions from all experts
        expert_probs = []
        for expert in self.experts:
            with torch.no_grad():
                outputs = expert(pixel_values)
                # Fix: Get proper probability shape (batch_size,)
                logits = outputs.logits  # Shape: (batch_size, num_queries, num_classes)
                # Take mean across queries, then sigmoid for binary classification
                probs = torch.sigmoid(logits[..., 0].mean(dim=1))  # Shape: (batch_size,)
                expert_probs.append(probs)
        
        # Stack expert predictions: (batch_size, n_models)
        expert_probs = torch.stack(expert_probs, dim=1)
        
        # Get gating weights
        weights = self.gate(expert_probs)
        
        # Compute weighted prediction
        final_pred = torch.sum(weights * expert_probs, dim=1)
        
        return final_pred, weights, expert_probs

# ================================
# TRAINING UTILITIES
# ================================

class MoETrainer:
    """Trainer for the integrated MoE model."""
    
    def __init__(self, integrated_moe, device):
        self.model = integrated_moe
        self.device = device
        self.model.to(device)
    
    def _extract_targets(self, batch):
        """Extract binary targets from batch labels."""
        return torch.tensor([
            1.0 if any([l['class_labels'].sum().item() > 0 for l in batch['labels']]) else 0.0 
            for l in batch['labels']
        ], dtype=torch.float32).to(self.device)
    
    def train_gate_only(self, train_loader, val_loader, epochs=50, lr=1e-3):
        """Train only the gating network while keeping experts frozen."""
        optimizer = torch.optim.Adam(self.model.gate.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                pixel_values = batch['pixel_values'].to(self.device)
                targets = self._extract_targets(batch)
                
                final_pred, weights, expert_probs = self.model(pixel_values)
                loss = criterion(final_pred, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            
            # Validation phase
            if (epoch + 1) % 10 == 0 or epoch == 0:
                val_acc = self.evaluate(val_loader)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), "integrated_moe_best.pth")
                
                print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Val acc: {val_acc:.4f}")
        
        return best_val_acc
    
    def evaluate(self, data_loader):
        """Evaluate the model on given data loader."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                targets = self._extract_targets(batch)
                
                final_pred, _, _ = self.model(pixel_values)
                predicted = (final_pred > 0.5).float()
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total

# ================================
# LEGACY APPROACHES (for reference)
# ================================

class TopKMoE(BaseEstimator, ClassifierMixin):
    """Calibrated Mixture of Experts: select top-k model predictions."""
    
    def __init__(self, k=2):
        self.k = k
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
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
        idx = np.argsort(-X, axis=1)[:, :self.k]
        X_topk = np.take_along_axis(X, idx, axis=1)
        return X_topk

# ================================
# DATA PROCESSING
# ================================

def get_model_probs(models, image_processors, data_loader):
    """Extract predicted probabilities from multiple models."""
    all_probs = []
    all_targets = []
    
    for batch in tqdm(data_loader, desc="MoE feature extraction"):
        batch_probs = []
        for model, processor in zip(models, image_processors):
            with torch.no_grad():
                inputs = batch['pixel_values'].to(model.device)
                outputs = model(inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits[..., 0]).cpu().numpy()
                batch_probs.append(probs)
        
        batch_probs = np.stack(batch_probs, axis=-1)
        all_probs.append(batch_probs)
        
        batch_targets = [
            1 if any([l['class_labels'].sum().item() > 0 for l in batch['labels']]) else 0 
            for l in batch['labels']
        ]
        all_targets.extend(batch_targets)
    
    X = np.concatenate(all_probs, axis=0)
    y = np.array(all_targets)
    return X, y

# ================================
# MAIN TRAINING PIPELINE
# ================================

def main(config_path, epoch=None, dataset=None, weight_dir=None):
    """Main training pipeline for MoE model."""
    # Load configuration
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)
    
    # Fix: Get epochs from argument -> config -> default (in that order)
    training_cfg = config.get('training', {})
    EPOCHS = epoch if epoch is not None else training_cfg.get('epochs', 50)

    # Setup model paths
    if weight_dir is not None:
        yolos_models = [
            os.path.join(weight_dir, sub) 
            for sub in ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM']
        ]
        image_processors = [AutoImageProcessor.from_pretrained(m) for m in yolos_models]
    else:
        yolos_models = [
            config.get('moe', {}).get('model1', 'hustvl/yolos-base'),
            config.get('moe', {}).get('model2', 'hustvl/yolos-small'),
            config.get('moe', {}).get('model3', 'hustvl/yolos-tiny'),
        ]
        image_processors = [get_image_processor(m, MAX_SIZE) for m in yolos_models]
    
    # Setup model types
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base') 
    model_types = [get_model_type(MODEL_NAME) for _ in yolos_models]
    
    # Load expert models
    print("Loading expert models...")
    models = [
        get_yolos_model(m, ip, mt) 
        for m, ip, mt in zip(yolos_models, image_processors, model_types)
    ]

    # Setup datasets
    print("Setting up datasets...")
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

    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, num_workers=2, 
        pin_memory=True, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, num_workers=2, 
        pin_memory=True, shuffle=False, collate_fn=collate_fn
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Train batches: {len(train_loader)}')
    
    # Setup device and move models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for i, model in enumerate(models):
        models[i] = model.to(device)
    
    # Create and train integrated MoE
    print("Creating integrated MoE model...")
    integrated_moe = IntegratedMoE(models, n_models=len(models))
    moe_trainer = MoETrainer(integrated_moe, device)
    
    print("Training integrated MoE (gating network only)...")
    best_acc = moe_trainer.train_gate_only(train_loader, val_loader, epochs=EPOCHS)
    
    # Save final model
    torch.save(integrated_moe.state_dict(), "integrated_moe_final.pth")
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    print("Models saved: integrated_moe_best.pth, integrated_moe_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MoE model with gating network")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', 
                       help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, 
                       help='Dataset epoch value to pass to dataset')
    parser.add_argument('--dataset', type=str, default=None, 
                       help='Dataset name to use (overrides config)')
    parser.add_argument('--weight_dir', type=str, default=None, 
                       help='Path to directory containing yolos_CSAW, yolos_DMID, yolos_DDSM subfolders')
    
    args = parser.parse_args()
    main(args.config, args.epoch, args.dataset, args.weight_dir)