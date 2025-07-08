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
import yaml
from PIL import Image
from functools import partial
from pathlib import Path
import datetime
from tqdm.auto import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn, run_model_inference_with_map
import pandas as pd

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class SimpleDatasetClassifier(nn.Module):
    """Simple CNN to classify which dataset an image belongs to."""
    def __init__(self, num_classes=3, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.class_names = ['CSAW', 'DMID', 'DDSM']
        
        # Simple CNN for dataset classification
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass."""
        features = self.features(x)
        logits = self.classifier(features)
        return logits

class BoundingBoxCalibrationLayer(nn.Module):
    """Calibration layer to refine bounding box predictions."""
    def __init__(self, input_dim=4, hidden_dim=128, device='cuda'):
        super().__init__()
        self.device = device
        
        # Multi-layer refinement network for bounding boxes
        self.box_refiner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Output refinement deltas in [-1, 1]
        )
        
        # Confidence calibration for logits
        self.logits_calibrator = nn.Sequential(
            nn.Linear(2, 64),  # Assuming 2 classes (background, cancer)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            
            nn.Linear(32, 2),
        )
        
        # Learnable refinement weights
        self.box_refine_weight = nn.Parameter(torch.tensor(0.1))
        self.logits_refine_weight = nn.Parameter(torch.tensor(0.1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
    
    def forward(self, pred_boxes, logits):
        """
        Refine predictions.
        
        Args:
            pred_boxes: [batch_size, num_queries, 4] - original box predictions
            logits: [batch_size, num_queries, num_classes] - original logits
            
        Returns:
            refined_boxes: [batch_size, num_queries, 4] - refined box predictions
            refined_logits: [batch_size, num_queries, num_classes] - refined logits
        """
        batch_size, num_queries, _ = pred_boxes.shape
        
        # Flatten for processing
        flat_boxes = pred_boxes.view(-1, 4)  # [batch_size * num_queries, 4]
        flat_logits = logits.view(-1, logits.size(-1))  # [batch_size * num_queries, num_classes]
        
        # Refine bounding boxes
        box_deltas = self.box_refiner(flat_boxes)
        refined_flat_boxes = flat_boxes + self.box_refine_weight * box_deltas
        
        # Clamp refined boxes to valid range [0, 1]
        refined_flat_boxes = torch.clamp(refined_flat_boxes, 0.0, 1.0)
        
        # Refine logits
        logits_deltas = self.logits_calibrator(flat_logits)
        refined_flat_logits = flat_logits + self.logits_refine_weight * logits_deltas
        
        # Reshape back
        refined_boxes = refined_flat_boxes.view(batch_size, num_queries, 4)
        refined_logits = refined_flat_logits.view(batch_size, num_queries, -1)
        
        return refined_boxes, refined_logits

class SimpleMoE(nn.Module):
    """Simple MoE with frozen components."""
    def __init__(self, expert_models, dataset_classifier, device):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)
        self.classifier = dataset_classifier
        self.device = device
        self.expert_names = ['CSAW', 'DMID', 'DDSM']
        
        # Freeze ALL components
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
            expert.eval()
        
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()
        
        # Statistics
        self.routing_counts = torch.zeros(3, device=device)
        self.total_routed = 0

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass
    
    def gradient_checkpointing_disable(self):
        pass
    
    def forward(self, pixel_values, labels=None, return_routing=False):
        """Forward pass with dataset classification routing."""
        batch_size = pixel_values.shape[0]
        
        # 1. Classifier chooses expert
        with torch.no_grad():
            self.classifier.eval()
            dataset_logits = self.classifier(pixel_values)
            expert_choices = torch.argmax(dataset_logits, dim=1)
        
        # 2. Update statistics
        if not self.training:
            for choice in expert_choices:
                self.routing_counts[choice] += 1
            self.total_routed += batch_size
        
        # 3. Group by expert
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # 4. Get expert outputs
        sample_to_output = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for expert_idx, sample_indices in expert_groups.items():
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            with torch.no_grad():
                self.experts[expert_idx].eval()
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
            
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                weight = len(sample_indices) / batch_size
                total_loss = total_loss + expert_output.loss * weight
            
            for i, sample_idx in enumerate(sample_indices):
                sample_to_output[sample_idx] = {
                    'logits': expert_output.logits[i].clone(),
                    'pred_boxes': expert_output.pred_boxes[i].clone(),
                }
        
        # 5. Reconstruct batch
        batch_logits = []
        batch_pred_boxes = []
        
        first_sample_idx = list(sample_to_output.keys())[0]
        ref_logits_shape = sample_to_output[first_sample_idx]['logits'].shape
        ref_boxes_shape = sample_to_output[first_sample_idx]['pred_boxes'].shape
        
        for i in range(batch_size):
            if i in sample_to_output:
                batch_logits.append(sample_to_output[i]['logits'])
                batch_pred_boxes.append(sample_to_output[i]['pred_boxes'])
            else:
                dummy_logits = torch.zeros(ref_logits_shape, device=pixel_values.device)
                dummy_boxes = torch.zeros(ref_boxes_shape, device=pixel_values.device)
                batch_logits.append(dummy_logits)
                batch_pred_boxes.append(dummy_boxes)
        
        batch_logits = torch.stack(batch_logits, dim=0)
        batch_pred_boxes = torch.stack(batch_pred_boxes, dim=0)
        batch_pred_boxes = batch_pred_boxes[..., :4].contiguous()
        
        # 6. Create output
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        final_loss = None
        if labels is not None:
            final_loss = total_loss
        
        combined_output = YolosObjectDetectionOutput(
            loss=final_loss,
            logits=batch_logits,
            pred_boxes=batch_pred_boxes,
            last_hidden_state=None
        )
        
        if return_routing:
            routing_info = {
                'choices': expert_choices,
                'groups': expert_groups,
                'stats': self.get_routing_stats()
            }
            return combined_output, routing_info
        else:
            return combined_output
    
    def get_routing_stats(self):
        """Get routing statistics."""
        if self.total_routed == 0:
            return {name: 0.0 for name in self.expert_names}
        
        stats = {}
        usage = (self.routing_counts / self.total_routed).cpu().numpy()
        for i, name in enumerate(self.expert_names):
            stats[name] = float(usage[i])
        return stats
    
    def reset_routing_stats(self):
        """Reset routing statistics."""
        self.routing_counts = torch.zeros(3, device=self.device)
        self.total_routed = 0

class CalibratedMoE(nn.Module):
    """MoE with calibration layer for refined outputs."""
    def __init__(self, moe_model, calibration_layer, device):
        super().__init__()
        self.moe = moe_model
        self.calibration = calibration_layer
        self.device = device
        
        # Freeze MoE completely
        for param in self.moe.parameters():
            param.requires_grad = False
        self.moe.eval()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass
    
    def gradient_checkpointing_disable(self):
        pass
    
    def forward(self, pixel_values, labels=None):
        """Forward pass with calibration."""
        # Get MoE predictions (frozen)
        with torch.no_grad():
            self.moe.eval()
            moe_output = self.moe(pixel_values, labels=labels)
        
        # Apply calibration (trainable)
        if self.training:
            refined_boxes, refined_logits = self.calibration(
                moe_output.pred_boxes.detach(), 
                moe_output.logits.detach()
            )
        else:
            refined_boxes, refined_logits = self.calibration(
                moe_output.pred_boxes, 
                moe_output.logits
            )
        
        # Create output with refined predictions
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        calibrated_output = YolosObjectDetectionOutput(
            loss=moe_output.loss,
            logits=refined_logits,
            pred_boxes=refined_boxes,
            last_hidden_state=moe_output.last_hidden_state
        )
        
        return calibrated_output
    
    def get_routing_stats(self):
        """Delegate to MoE."""
        return self.moe.get_routing_stats()
    
    def reset_routing_stats(self):
        """Delegate to MoE."""
        self.moe.reset_routing_stats()

def load_expert_models(weight_dir, device):
    """Load expert models."""
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM']
    expert_paths = [os.path.join(weight_dir, name) for name in expert_names]
    
    models = []
    processors = []
    
    for path in expert_paths:
        processor = AutoImageProcessor.from_pretrained(path)
        model = AutoModelForObjectDetection.from_pretrained(
            path,
            id2label={0: 'cancer'},
            label2id={'cancer': 0},
            auxiliary_loss=False,
        )
        model = model.to(device)
        model.eval()
        models.append(model)
        processors.append(processor)
    
    return models, processors

def load_pretrained_moe(expert_weights_dir, device):
    """Load pre-trained MoE with classifier."""
    # Load expert models
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    image_processor = expert_processors[0]
    
    # Load trained classifier
    moe_save_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    classifier_path = os.path.join(moe_save_dir, 'classifier_best.pth')
    
    if not os.path.exists(classifier_path):
        classifier_final_path = os.path.join(moe_save_dir, 'classifier_final.pth')
        if os.path.exists(classifier_final_path):
            classifier_path = classifier_final_path
        else:
            raise FileNotFoundError(f"No classifier found in {moe_save_dir}. Run train_moe3.py first.")
    
    # Initialize and load classifier
    classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    
    # Create MoE model
    moe_model = SimpleMoE(expert_models, classifier, device).to(device)
    moe_model.eval()
    
    return moe_model, image_processor

def create_calibration_dataset(config, image_processor, device):
    """Create dataset for calibration training using all datasets."""
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    datasets = []
    dataset_names = ['CSAW', 'DMID', 'DDSM']
    
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name} training data for calibration...")
        
        train_dataset = BreastCancerDataset(
            split='train',
            splits_dir=SPLITS_DIR,
            dataset_name=dataset_name,
            image_processor=image_processor,
            model_type=get_model_type(MODEL_NAME),
        )
        
        datasets.append(train_dataset)
        print(f"  {dataset_name}: {len(train_dataset)} samples")
    
    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"Total calibration training samples: {len(combined_dataset)}")
    
    return combined_dataset

def train_calibration_layer(config, device, expert_weights_dir, epochs=15):
    """Train only the calibration layer."""
    print("="*60)
    print("PHASE: TRAINING CALIBRATION LAYER")
    print("="*60)
    
    # Load pre-trained MoE
    moe_model, image_processor = load_pretrained_moe(expert_weights_dir, device)
    print("✅ Pre-trained MoE loaded successfully")
    
    # Create calibration layer
    calibration_layer = BoundingBoxCalibrationLayer(device=device).to(device)
    print("✅ Calibration layer initialized")
    
    # Create calibrated model
    calibrated_model = CalibratedMoE(moe_model, calibration_layer, device).to(device)
    
    # Create calibration dataset
    calibration_dataset = create_calibration_dataset(config, image_processor, device)
    
    # Split dataset
    dataset_size = len(calibration_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        calibration_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Calibration training: {len(train_dataset)} samples")
    print(f"Calibration validation: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,  # Smaller batch size for object detection
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(
        calibrated_model.calibration.parameters(), 
        lr=1e-4, 
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    # Training tracking
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    
    # Save directory
    calibrated_save_dir = os.path.join(expert_weights_dir, 'moec_MOMO')
    os.makedirs(calibrated_save_dir, exist_ok=True)
    
    print(f"\nStarting calibration training for {epochs} epochs...")
    print(f"Save directory: {calibrated_save_dir}")
    print(f"Only training calibration layer parameters: {sum(p.numel() for p in calibrated_model.calibration.parameters()):,}")
    
    for epoch in range(epochs):
        # Training phase
        calibrated_model.train()
        calibrated_model.calibration.train()
        train_loss = 0.0
        train_steps = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            optimizer.zero_grad()
            
            # Forward pass (only calibration layer is trainable)
            outputs = calibrated_model(pixel_values, labels=labels)
            
            # Use the loss from the output
            loss = outputs.loss
            if loss is not None and torch.isfinite(loss):
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(calibrated_model.calibration.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                train_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg': f"{train_loss/train_steps:.4f}"
                })
        
        avg_train_loss = train_loss / max(train_steps, 1)
        
        # Validation phase
        calibrated_model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels']
                
                outputs = calibrated_model(pixel_values, labels=labels)
                loss = outputs.loss
                
                if loss is not None and torch.isfinite(loss):
                    val_loss += loss.item()
                    val_steps += 1
                
                val_pbar.set_postfix({
                    'Loss': f"{loss.item() if loss is not None else 0:.4f}",
                    'Avg': f"{val_loss/max(val_steps,1):.4f}"
                })
        
        avg_val_loss = val_loss / max(val_steps, 1)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save calibration layer
            calibration_path = os.path.join(calibrated_save_dir, 'calibration_best.pth')
            torch.save(calibrated_model.calibration.state_dict(), calibration_path)
            
            # Save complete model
            complete_model_path = os.path.join(calibrated_save_dir, 'calibrated_moe_best.pth')
            torch.save({
                'calibration_state_dict': calibrated_model.calibration.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'config': config
            }, complete_model_path)
            
            print(f"  ✅ New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 50)
    
    # Save final model
    calibration_final_path = os.path.join(calibrated_save_dir, 'calibration_final.pth')
    torch.save(calibrated_model.calibration.state_dict(), calibration_final_path)
    
    complete_final_path = os.path.join(calibrated_save_dir, 'calibrated_moe_final.pth')
    torch.save({
        'calibration_state_dict': calibrated_model.calibration.state_dict(),
        'epoch': epochs,
        'final_val_loss': avg_val_loss,
        'config': config
    }, complete_final_path)
    
    print(f"\n" + "="*60)
    print("CALIBRATION TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Calibration layer saved to: {calibration_path}")
    print(f"Complete model saved to: {complete_model_path}")
    print("="*60)
    
    return calibrated_model, best_val_loss

def evaluate_calibrated_moe(config, device, dataset_name, expert_weights_dir):
    """Evaluate calibrated MoE and compare with original MoE."""
    print(f"="*60)
    print(f"EVALUATING CALIBRATED MOE ON {dataset_name}")
    print("="*60)
    
    # Load original MoE
    original_moe, image_processor = load_pretrained_moe(expert_weights_dir, device)
    
    # Load calibration layer
    calibrated_save_dir = os.path.join(expert_weights_dir, 'moec_MOMO')
    calibration_path = os.path.join(calibrated_save_dir, 'calibration_best.pth')
    
    if not os.path.exists(calibration_path):
        calibration_final_path = os.path.join(calibrated_save_dir, 'calibration_final.pth')
        if os.path.exists(calibration_final_path):
            calibration_path = calibration_final_path
        else:
            raise FileNotFoundError(f"No calibration layer found in {calibrated_save_dir}. Run training first.")
    
    # Create calibrated model
    calibration_layer = BoundingBoxCalibrationLayer(device=device).to(device)
    calibration_layer.load_state_dict(torch.load(calibration_path, map_location=device))
    calibrated_moe = CalibratedMoE(original_moe, calibration_layer, device).to(device)
    calibrated_moe.eval()
    
    # Create test dataset
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Evaluate both models
    from evaluation import run_model_inference_with_map
    
    print(f"\n=== Evaluating Original MoE ===")
    original_metrics = run_model_inference_with_map(original_moe, test_dataset, image_processor, device)
    
    print(f"\n=== Evaluating Calibrated MoE ===")
    calibrated_metrics = run_model_inference_with_map(calibrated_moe, test_dataset, image_processor, device)
    
    # Compare results
    print(f"\n" + "="*80)
    print(f"CALIBRATION RESULTS COMPARISON - {dataset_name}")
    print("="*80)
    
    comparison_data = []
    for metric in ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large']:
        original_val = original_metrics.get(metric, 0.0)
        calibrated_val = calibrated_metrics.get(metric, 0.0)
        improvement = calibrated_val - original_val
        improvement_pct = (improvement / original_val * 100) if original_val > 0 else 0.0
        
        comparison_data.append({
            'Metric': metric,
            'Original_MoE': f"{original_val:.4f}",
            'Calibrated_MoE': f"{calibrated_val:.4f}",
            'Improvement': f"{improvement:.4f}",
            'Improvement_%': f"{improvement_pct:.2f}%"
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Summary
    map50_improvement = calibrated_metrics.get('map_50', 0) - original_metrics.get('map_50', 0)
    print(f"\n📊 Summary:")
    print(f"   mAP@50 improvement: {map50_improvement:.4f}")
    
    if map50_improvement > 0.01:
        print(f"   Status: ✅ Significant improvement with calibration")
    elif map50_improvement > 0:
        print(f"   Status: ✅ Minor improvement with calibration")
    else:
        print(f"   Status: ⚠️ No improvement or degradation")
    
    return calibrated_metrics, original_metrics

def main(config_path, epoch=None, dataset=None, weight_dir=None, test=False):
    """Main function."""
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if weight_dir is not None:
        expert_weights_dir = weight_dir
    else:
        expert_weights_dir = config.get('moe', {}).get('expert_weights_dir', '/content/Weights')
    
    print(f"Using device: {device}")
    print(f"Expert weights: {expert_weights_dir}")
    
    if test:
        # Test-only mode: evaluate calibrated MoE
        target_dataset = dataset if dataset else 'CSAW'
        print(f"Running test-only mode on dataset: {target_dataset}")
        
        try:
            calibrated_results, original_results = evaluate_calibrated_moe(
                config, device, target_dataset, expert_weights_dir
            )
            return calibrated_results
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {'error': str(e)}
    else:
        # Training mode
        epochs = epoch if epoch else 15
        print(f"Training calibration layer for {epochs} epochs...")
        
        try:
            # Train calibration layer
            calibrated_model, best_loss = train_calibration_layer(
                config, device, expert_weights_dir, epochs
            )
            
            print(f"\nCalibration training completed with best loss: {best_loss:.4f}")
            
            # Test on specified dataset or all datasets
            if dataset:
                target_datasets = [dataset]
            else:
                target_datasets = ['CSAW', 'DMID', 'DDSM']
            
            print(f"\nTesting calibrated MoE on: {target_datasets}")
            all_results = {}
            
            for test_dataset in target_datasets:
                print(f"\n{'='*50}")
                print(f"Testing Calibrated MoE on {test_dataset}")
                print('='*50)
                
                try:
                    calibrated_results, original_results = evaluate_calibrated_moe(
                        config, device, test_dataset, expert_weights_dir
                    )
                    all_results[test_dataset] = {
                        'calibrated': calibrated_results,
                        'original': original_results
                    }
                except Exception as e:
                    print(f"Evaluation on {test_dataset} failed: {e}")
                    all_results[test_dataset] = {'error': str(e)}
            
            print(f"\n" + "="*60)
            print("CALIBRATED MOE TRAINING & TESTING COMPLETED!")
            print("✅ Calibration layer trained and saved")
            print("✅ Performance comparison completed")
            print("="*60)
            
            return all_results
            
        except Exception as e:
            print(f"Training failed: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Number of epochs for calibration training')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name for testing (CSAW/DMID/DDSM)')
    parser.add_argument('--weight_dir', type=str, default=None, help='Expert weights directory')
    parser.add_argument('--test', action='store_true', help='Test-only mode: evaluate calibrated MoE without training')
    args = parser.parse_args()

    main(args.config, args.epoch, args.dataset, args.weight_dir, args.test)
