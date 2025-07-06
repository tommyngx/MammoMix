import argparse
import os
import pickle
import datetime
from pathlib import Path

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

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
    def __init__(self, n_models=4, hidden_dim=16, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_models),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_probs):
        """Forward pass through gating network with top-k selection."""
        weights = self.gate(expert_probs)
        
        # Apply top-k selection: zero out all but top-k weights
        _, top_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # Create mask for top-k selection
        mask = torch.zeros_like(weights)
        mask.scatter_(-1, top_indices, 1.0)
        
        # Apply mask and renormalize
        masked_weights = weights * mask
        normalized_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return normalized_weights, top_indices

class IntegratedMoE(nn.Module):
    """
    Integrated MoE model that contains all expert models and gating network.
    During training, only the gating network is trained while experts are frozen.
    """
    def __init__(self, expert_models, n_models=4, hidden_dim=16, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)
        self.gate = GatingNetwork(n_models, hidden_dim, top_k)
        self.top_k = top_k
        
        # Freeze expert models
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values):
        """Forward pass through integrated MoE with proper output structure preservation."""
        # Get full outputs from all experts (not just probabilities)
        expert_outputs = []
        expert_probs = []
        
        for expert in self.experts:
            with torch.no_grad():
                outputs = expert(pixel_values)
                expert_outputs.append(outputs)
                # Extract probabilities for gating
                logits = outputs.logits  # Shape: (batch_size, num_queries, num_classes)
                probs = torch.sigmoid(logits[..., 0].mean(dim=1))  # Shape: (batch_size,)
                expert_probs.append(probs)
        
        # Stack expert predictions: (batch_size, n_models)
        expert_probs = torch.stack(expert_probs, dim=1)
        
        # Get gating weights and top-k indices
        weights, top_indices = self.gate(expert_probs)
        
        # Combine outputs from selected experts
        batch_size = pixel_values.shape[0]
        combined_outputs = self._combine_expert_outputs(expert_outputs, weights, top_indices, batch_size)
        
        # Compute final prediction probability for training
        final_pred = torch.sum(weights * expert_probs, dim=1)
        
        return combined_outputs, final_pred, weights, expert_probs, top_indices
    
    def _combine_expert_outputs(self, expert_outputs, weights, top_indices, batch_size):
        """Combine outputs from top-k selected experts maintaining object detection structure."""
        # Initialize combined output structure based on first expert
        reference_output = expert_outputs[0]
        
        # Get dimensions
        num_queries = reference_output.logits.shape[1]
        num_classes = reference_output.logits.shape[2]
        
        # Initialize combined logits and boxes
        combined_logits = torch.zeros(batch_size, num_queries, num_classes, 
                                    device=reference_output.logits.device)
        
        # For each sample in batch, combine outputs from its top-k experts
        for batch_idx in range(batch_size):
            sample_combined_logits = torch.zeros(num_queries, num_classes, 
                                               device=reference_output.logits.device)
            
            # Get top-k experts for this sample
            for k_idx in range(self.top_k):
                expert_idx = top_indices[batch_idx, k_idx].item()
                expert_weight = weights[batch_idx, expert_idx].item()
                
                # Weight and add expert's contribution
                expert_logits = expert_outputs[expert_idx].logits[batch_idx]
                sample_combined_logits += expert_weight * expert_logits
            
            combined_logits[batch_idx] = sample_combined_logits
        
        # Create output object compatible with transformers
        class ObjectDetectionOutput:
            def __init__(self, logits, pred_boxes=None):
                self.logits = logits
                self.pred_boxes = pred_boxes if pred_boxes is not None else reference_output.pred_boxes
                self.loss = torch.tensor(0.0, device=logits.device, requires_grad=False)
                
            def __getitem__(self, key):
                """Make object subscriptable for Trainer compatibility"""
                if key == 0:
                    return self.loss
                elif key == "loss":
                    return self.loss
                elif isinstance(key, slice):
                    if key == slice(1, None, None):  # outputs[1:]
                        return (self.logits,)
                    else:
                        return ()
                else:
                    raise KeyError(f"Key {key} not found")
                    
            def keys(self):
                """Return available keys"""
                return ["loss", "logits", "pred_boxes"]
                
        return ObjectDetectionOutput(combined_logits)

# ================================
# TRAINING UTILITIES
# ================================

class MoETrainer:
    """Trainer for the integrated MoE model."""
    
    def __init__(self, integrated_moe, device, output_dir):
        self.model = integrated_moe
        self.device = device
        self.output_dir = output_dir
        self.model.to(device)
    
    def _extract_targets(self, batch):
        """Extract binary targets from batch labels."""
        targets = []
        for labels_list in batch['labels']:
            # Handle case where labels_list might be a single dict or list of dicts
            if isinstance(labels_list, dict):
                # Single label dict
                has_cancer = labels_list.get('class_labels', torch.tensor([])).sum().item() > 0
            elif isinstance(labels_list, list) and len(labels_list) > 0:
                # List of label dicts
                has_cancer = any([
                    l.get('class_labels', torch.tensor([])).sum().item() > 0 
                    for l in labels_list if isinstance(l, dict)
                ])
            else:
                # Empty or invalid labels
                has_cancer = False
            
            targets.append(1.0 if has_cancer else 0.0)
        
        return torch.tensor(targets, dtype=torch.float32).to(self.device)
    
    def train_gate_only(self, train_loader, val_loader, epochs=50, lr=1e-3):
        """Train only the gating network while keeping experts frozen."""
        optimizer = torch.optim.Adam(self.model.gate.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        best_val_acc = 0.0
        best_model_path = os.path.join(self.output_dir, "integrated_moe_best.pth")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                pixel_values = batch['pixel_values'].to(self.device)
                targets = self._extract_targets(batch)
                
                combined_outputs, final_pred, weights, expert_probs, top_indices = self.model(pixel_values)
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
                    # Save only the best model
                    os.makedirs(self.output_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"New best model saved: {best_model_path}")
                
                print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Val acc: {val_acc:.4f}")
        
        return best_val_acc, best_model_path
    
    def evaluate(self, data_loader):
        """Evaluate the model on given data loader."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                targets = self._extract_targets(batch)
                
                combined_outputs, final_pred, weights, expert_probs, top_indices = self.model(pixel_values)
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

class MoEObjectDetectionModel(nn.Module):
    """
    Wrapper around IntegratedMoE to provide object detection interface
    similar to standard YOLOS models for compatibility with test.py
    """
    def __init__(self, integrated_moe):
        super().__init__()
        self.moe = integrated_moe
        
    def forward(self, pixel_values, labels=None, **kwargs):
        """Return object detection outputs compatible with test.py evaluation"""
        # Get MoE combined output - this now returns properly structured output
        combined_outputs, final_pred, weights, expert_probs, top_indices = self.moe(pixel_values)
        
        # The issue might be here - we need to preserve the original labels structure
        # Don't modify the combined_outputs, instead create a new one that maintains labels
        class ObjectDetectionOutputWithLabels:
            def __init__(self, logits, pred_boxes=None, labels=None):
                self.logits = logits
                self.pred_boxes = pred_boxes
                self.labels = labels
                self.loss = torch.tensor(0.0, device=logits.device, requires_grad=False)
                
            def __getitem__(self, key):
                """Make object subscriptable for Trainer compatibility"""
                if key == 0:
                    return self.loss
                elif key == "loss":
                    return self.loss
                elif isinstance(key, slice):
                    if key == slice(1, None, None):  # outputs[1:]
                        return (self.logits,)
                    else:
                        return ()
                else:
                    raise KeyError(f"Key {key} not found")
                    
            def keys(self):
                """Return available keys"""
                return ["loss", "logits", "pred_boxes", "labels"]
        
        # Create new output that preserves labels
        return ObjectDetectionOutputWithLabels(
            logits=combined_outputs.logits,
            pred_boxes=combined_outputs.pred_boxes,
            labels=labels
        )

def test_moe_model(config_path, model_path, dataset_name, weight_dir, epoch=None):
    """Test the trained MoE model using evaluation metrics like test.py"""
    from transformers import Trainer, TrainingArguments
    from evaluation import get_eval_compute_metrics_fn
    
    config = load_config(config_path)
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)
    
    # Load expert models (same as training)
    if weight_dir is not None:
        yolos_models = [
            os.path.join(weight_dir, sub) 
            for sub in ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
        ]
        image_processors = [AutoImageProcessor.from_pretrained(m) for m in yolos_models]
    else:
        yolos_models = [
            config.get('moe', {}).get('model1', 'hustvl/yolos-base'),
            config.get('moe', {}).get('model2', 'hustvl/yolos-small'),
            config.get('moe', {}).get('model3', 'hustvl/yolos-tiny'),
            config.get('moe', {}).get('model4', 'hustvl/yolos-base'),
        ]
        image_processors = [get_image_processor(m, MAX_SIZE) for m in yolos_models]
    
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base') 
    model_types = [get_model_type(MODEL_NAME) for _ in yolos_models]
    
    # Load expert models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        get_yolos_model(m, ip, mt).to(device)
        for m, ip, mt in zip(yolos_models, image_processors, model_types)
    ]
    
    # Create MoE model and load trained weights
    integrated_moe = IntegratedMoE(models, n_models=len(models), top_k=2)
    integrated_moe.load_state_dict(torch.load(model_path, map_location=device))
    integrated_moe.eval()
    
    # Wrap MoE for object detection compatibility
    moe_detector = MoEObjectDetectionModel(integrated_moe)
    
    # Create test dataset
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processors[0],
        model_type=model_types[0],
        dataset_epoch=epoch
    )
    
    # Setup evaluation using same approach as test.py
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processors[0])
    
    training_cfg = config.get('training', {})
    output_dir = training_cfg.get('output_dir', './moe_output')
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=[],
    )
    
    # Custom data collator that preserves label structure
    def moe_collate_fn(examples):
        batch = collate_fn(examples)
        
        # Ensure labels are in the correct format for evaluation
        if 'labels' in batch:
            # Convert labels to the format expected by evaluation
            formatted_labels = []
            for i, labels_item in enumerate(batch['labels']):
                # Handle BatchFeature objects from transformers
                if hasattr(labels_item, 'data') and isinstance(labels_item.data, dict):
                    # Convert BatchFeature to regular dict
                    label_dict = dict(labels_item.data)
                    formatted_labels.append(label_dict)
                elif isinstance(labels_item, dict):
                    # Already a regular dict
                    formatted_labels.append(labels_item)
                elif hasattr(labels_item, '__dict__'):
                    # Try to convert object with attributes to dict
                    try:
                        label_dict = {
                            'boxes': getattr(labels_item, 'boxes', torch.tensor([]).reshape(0, 4)),
                            'class_labels': getattr(labels_item, 'class_labels', torch.tensor([])),
                            'image_id': getattr(labels_item, 'image_id', torch.tensor([i])),
                            'size': getattr(labels_item, 'size', torch.tensor([640, 640])),
                            'area': getattr(labels_item, 'area', torch.tensor([])),
                            'iscrowd': getattr(labels_item, 'iscrowd', torch.tensor([])),
                            'orig_size': getattr(labels_item, 'orig_size', torch.tensor([640, 640]))
                        }
                        formatted_labels.append(label_dict)
                    except Exception as e:
                        # Create empty label dict for consistency
                        empty_label = {
                            'boxes': torch.tensor([]).reshape(0, 4),
                            'class_labels': torch.tensor([]),
                            'image_id': torch.tensor([i]),
                            'size': torch.tensor([640, 640]),
                            'area': torch.tensor([]),
                            'iscrowd': torch.tensor([]),
                            'orig_size': torch.tensor([640, 640])
                        }
                        formatted_labels.append(empty_label)
                else:
                    # Create empty label dict for consistency
                    empty_label = {
                        'boxes': torch.tensor([]).reshape(0, 4),
                        'class_labels': torch.tensor([]),
                        'image_id': torch.tensor([i]),
                        'size': torch.tensor([640, 640]),
                        'area': torch.tensor([]),
                        'iscrowd': torch.tensor([]),
                        'orig_size': torch.tensor([640, 640])
                    }
                    formatted_labels.append(empty_label)
            
            batch['labels'] = formatted_labels
        
        return batch
    
    # Let's also create a custom compute_metrics function that adds debugging
    def debug_compute_metrics(eval_pred):
        """Debug wrapper around the original compute_metrics function"""
        print(f"DEBUG compute_metrics: eval_pred type: {type(eval_pred)}")
        print(f"DEBUG compute_metrics: eval_pred keys: {eval_pred.keys() if hasattr(eval_pred, 'keys') else 'No keys'}")
        
        if hasattr(eval_pred, 'label_ids'):
            print(f"DEBUG compute_metrics: label_ids type: {type(eval_pred.label_ids)}")
            if eval_pred.label_ids is not None and len(eval_pred.label_ids) > 0:
                print(f"DEBUG compute_metrics: first label type: {type(eval_pred.label_ids[0])}")
                print(f"DEBUG compute_metrics: first label content keys: {eval_pred.label_ids[0].keys() if isinstance(eval_pred.label_ids[0], dict) else 'Not a dict'}")
                
                # The issue is that the evaluation expects individual image targets
                # but we're getting batch-level concatenated data. Let's restructure it.
                
                # Check if we have the flattened format and need to restructure
                if isinstance(eval_pred.label_ids[0], dict) and 'image_id' in eval_pred.label_ids[0]:
                    first_label = eval_pred.label_ids[0]
                    if isinstance(first_label['image_id'], (list, tuple, np.ndarray)) and len(first_label['image_id']) > 1:
                        # We have flattened data - need to restructure to individual images
                        print("DEBUG: Restructuring flattened labels to individual image targets...")
                        
                        # Extract the individual components
                        image_ids = first_label['image_id']
                        boxes = first_label['boxes']
                        class_labels = first_label['class_labels']
                        areas = first_label['area']
                        iscrowd = first_label['iscrowd']
                        sizes = first_label['size'].reshape(-1, 2)  # Reshape to (n_images, 2)
                        orig_sizes = first_label['orig_size'].reshape(-1, 2)  # Reshape to (n_images, 2)
                        
                        # Reconstruct individual image targets
                        restructured_labels = []
                        for i, img_id in enumerate(image_ids):
                            # Each image gets one box (since we have equal number of images and boxes)
                            individual_target = {
                                'image_id': torch.tensor([int(img_id)]),
                                'boxes': torch.tensor(np.array([boxes[i]], dtype=np.float32)).reshape(1, 4),
                                'class_labels': torch.tensor([int(class_labels[i])]),
                                'area': torch.tensor([float(areas[i])]),
                                'iscrowd': torch.tensor([int(iscrowd[i])]),
                                'size': torch.tensor(sizes[i].astype(np.int32)),
                                'orig_size': torch.tensor(orig_sizes[i].astype(np.int32))
                            }
                            restructured_labels.append(individual_target)
                        
                        # Create a new eval_pred with restructured labels
                        class RestructuredEvalPred:
                            def __init__(self, predictions, label_ids):
                                self.predictions = predictions
                                self.label_ids = label_ids
                        
                        eval_pred = RestructuredEvalPred(eval_pred.predictions, restructured_labels)
                        print(f"DEBUG: Restructured to {len(restructured_labels)} individual targets")
                        print(f"DEBUG: First restructured target type: {type(restructured_labels[0])}")
                        print(f"DEBUG: First restructured target keys: {restructured_labels[0].keys()}")
                        print(f"DEBUG: First restructured boxes type: {type(restructured_labels[0]['boxes'])}")
                        print(f"DEBUG: First restructured boxes shape: {restructured_labels[0]['boxes'].shape}")
        
        # Create a custom evaluation function that has extra debugging
        def debug_eval_compute_metrics_fn(eval_pred):
            """Debug version of the evaluation function"""
            print(f"DEBUG inside eval_compute_metrics_fn: eval_pred type: {type(eval_pred)}")
            print(f"DEBUG inside eval_compute_metrics_fn: label_ids type: {type(eval_pred.label_ids)}")
            print(f"DEBUG inside eval_compute_metrics_fn: label_ids length: {len(eval_pred.label_ids)}")
            
            if len(eval_pred.label_ids) > 0:
                first_target = eval_pred.label_ids[0]
                print(f"DEBUG inside eval_compute_metrics_fn: first_target type: {type(first_target)}")
                print(f"DEBUG inside eval_compute_metrics_fn: first_target: {first_target}")
                
                if isinstance(first_target, dict):
                    print(f"DEBUG inside eval_compute_metrics_fn: first_target keys: {first_target.keys()}")
                    if 'boxes' in first_target:
                        print(f"DEBUG inside eval_compute_metrics_fn: boxes type: {type(first_target['boxes'])}")
                        print(f"DEBUG inside eval_compute_metrics_fn: boxes value: {first_target['boxes']}")
                    else:
                        print("DEBUG inside eval_compute_metrics_fn: 'boxes' key not found!")
                else:
                    print(f"DEBUG inside eval_compute_metrics_fn: first_target is not a dict, it's a {type(first_target)}")
            
            # Call the original evaluation function
            return eval_compute_metrics_fn(eval_pred)
        
        # Call the debug version instead
        return debug_eval_compute_metrics_fn(eval_pred)
    
    trainer = Trainer(
        model=moe_detector,
        args=training_args,
        eval_dataset=test_dataset,
        processing_class=image_processors[0],
        data_collator=moe_collate_fn,
        compute_metrics=debug_compute_metrics,
    )
    
    print(f'Test loader: {len(test_dataset)} samples')
    
    # Test a single batch to verify label structure
    test_batch = moe_collate_fn([test_dataset[0], test_dataset[1]])
    print(f"Test batch label structure: {type(test_batch['labels'][0])}")
    print(f"Test batch label keys: {test_batch['labels'][0].keys()}")
    print(f"Test batch boxes type: {type(test_batch['labels'][0]['boxes'])}")
    
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    print("\n=== MoE Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return test_results

def main(config_path, epoch=None, dataset=None, weight_dir=None, weight_test=None):
    """Main training pipeline for MoE model."""
    # If weight_test is provided, only run testing
    if weight_test is not None:
        print(f"Testing mode: Loading trained MoE model from {weight_test}")
        config = load_config(config_path)
        DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
        
        print("=== Testing trained MoE model with object detection metrics ===")
        test_results = test_moe_model(config_path, weight_test, DATASET_NAME, weight_dir, epoch)
        return None, test_results
    
    # Load configuration
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)
    
    # Fix: Get epochs from argument -> config -> default (in that order)
    training_cfg = config.get('training', {})
    output_dir = training_cfg.get('output_dir', './moe_output')
    EPOCHS = epoch if epoch is not None else training_cfg.get('epochs', 50)

    # Setup model paths - Add yolos_MOMO as 4th expert
    if weight_dir is not None:
        yolos_models = [
            os.path.join(weight_dir, sub) 
            for sub in ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
        ]
        image_processors = [AutoImageProcessor.from_pretrained(m) for m in yolos_models]
    else:
        yolos_models = [
            config.get('moe', {}).get('model1', 'hustvl/yolos-base'),
            config.get('moe', {}).get('model2', 'hustvl/yolos-small'),
            config.get('moe', {}).get('model3', 'hustvl/yolos-tiny'),
            config.get('moe', {}).get('model4', 'hustvl/yolos-base'),
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
    
    # Create and train integrated MoE with top-k=2
    print("Creating integrated MoE model...")
    integrated_moe = IntegratedMoE(models, n_models=len(models), top_k=2)
    moe_trainer = MoETrainer(integrated_moe, device, output_dir)
    
    print("Training integrated MoE (gating network only)...")
    best_acc, best_model_path = moe_trainer.train_gate_only(train_loader, val_loader, epochs=EPOCHS)
    
    print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")
    
    # Test the trained model using object detection evaluation
    print("\n=== Testing trained MoE model with object detection metrics ===")
    test_results = test_moe_model(config_path, best_model_path, DATASET_NAME, weight_dir, epoch)
    
    return best_acc, test_results

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
    parser.add_argument('--weight_test', type=str, default=None, 
                       help='Path to trained MoE model for testing only (skips training)')
    
    args = parser.parse_args()
    main(args.config, args.epoch, args.dataset, args.weight_dir, args.weight_test)