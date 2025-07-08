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
        """Delegate to MoE if available."""
        if hasattr(self.moe, 'get_routing_stats'):
            return self.moe.get_routing_stats()
        return {}
    
    def reset_routing_stats(self):
        """Delegate to MoE if available."""
        if hasattr(self.moe, 'reset_routing_stats'):
            self.moe.reset_routing_stats()

def load_pretrained_moe(expert_weights_dir, device):
    """Load pre-trained MoE model directly from saved checkpoint."""
    moe_save_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    
    # Check for saved MoE model files
    saved_moe_path = None
    for filename in ['model.safetensors', 'pytorch_model.bin']:
        potential_path = os.path.join(moe_save_dir, filename)
        if os.path.exists(potential_path):
            saved_moe_path = moe_save_dir
            break
    
    if saved_moe_path:
        print(f"Loading pre-trained MoE from: {saved_moe_path}")
        try:
            moe_model = AutoModelForObjectDetection.from_pretrained(
                saved_moe_path,
                id2label={0: 'cancer'},
                label2id={'cancer': 0},
                auxiliary_loss=False,
            ).to(device)
            
            # Load image processor
            image_processor = AutoImageProcessor.from_pretrained(saved_moe_path)
            
            print("✅ Pre-trained MoE loaded from saved checkpoint")
            return moe_model, image_processor
            
        except Exception as e:
            print(f"Failed to load saved MoE: {e}")
            raise FileNotFoundError(f"Cannot load MoE model from {saved_moe_path}. Please ensure the model exists.")
    else:
        raise FileNotFoundError(f"No MoE model found in {moe_save_dir}. Please run train_moe3.py first.")

def create_calibration_dataset(config, image_processor, device, dataset_name):
    """Create dataset for calibration training using only the specified dataset."""
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    print(f"Loading {dataset_name} training data for calibration...")
    
    train_dataset = BreastCancerDataset(
        split='train',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
    )
    
    print(f"  {dataset_name}: {len(train_dataset)} samples")
    print(f"Total calibration training samples: {len(train_dataset)}")
    
    return train_dataset

def compute_calibration_loss(pred_boxes, pred_logits, target_labels, device):
    """Compute calibration loss comparing predictions with ground truth."""
    # Move all tensors to the same device
    pred_boxes = pred_boxes.to(device)
    pred_logits = pred_logits.to(device)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Box regression loss (only for positive targets)
    for i, labels in enumerate(target_labels):
        if labels and len(labels) > 0:
            target_boxes = torch.stack([torch.tensor(label['boxes'], device=device) for label in labels])
            target_classes = torch.tensor([label['class_labels'] for label in labels], device=device)
            
            # Simple L1 loss for box refinement
            if len(target_boxes) > 0:
                # Take first few predictions that match number of targets
                num_targets = min(len(target_boxes), pred_boxes.shape[1])
                box_loss = F.l1_loss(pred_boxes[i, :num_targets], target_boxes[:num_targets])
                total_loss = total_loss + box_loss
    
    return total_loss

def save_calibrated_moe_model(calibrated_model, image_processor, save_dir, dataset_name):
    """Save calibrated MoE model in standard HuggingFace format."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(save_dir, 'model.safetensors')
    from safetensors.torch import save_file
    
    # Only save calibration layer weights
    calibration_state = {f"calibration.{k}": v for k, v in calibrated_model.calibration.state_dict().items()}
    save_file(calibration_state, model_path)
    
    # Save preprocessor config
    image_processor.save_pretrained(save_dir)
    
    # Create and save model config
    config = {
        "model_type": "calibrated_moe",
        "base_model": "simple_moe",
        "dataset_trained": dataset_name,
        "calibration_layers": ["box_refiner", "logits_calibrator"],
        "num_labels": 2,
        "id2label": {"0": "cancer"},
        "label2id": {"cancer": 0},
        "problem_type": "single_label_classification",
        "torch_dtype": "float32",
        "transformers_version": "4.36.0"
    }
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Calibrated MoE model saved to: {save_dir}")
    print(f"  - model.safetensors (calibration weights)")
    print(f"  - config.json") 
    print(f"  - preprocessor_config.json")

def train_calibration_layer(config, device, expert_weights_dir, dataset_name, epochs=15):
    """Train only the calibration layer on specified dataset."""
    print("="*60)
    print(f"PHASE: TRAINING CALIBRATION LAYER ON {dataset_name}")
    print("="*60)
    
    # Load pre-trained MoE
    moe_model, image_processor = load_pretrained_moe(expert_weights_dir, device)
    print("✅ Pre-trained MoE loaded successfully")
    
    # Create calibration layer
    calibration_layer = BoundingBoxCalibrationLayer(device=device).to(device)
    print("✅ Calibration layer initialized")
    
    # Create calibrated model
    calibrated_model = CalibratedMoE(moe_model, calibration_layer, device).to(device)
    
    # Create calibration dataset for specific dataset only
    calibration_dataset = create_calibration_dataset(config, image_processor, device, dataset_name)
    
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
    
    # Create data loaders with proper device handling
    def custom_collate_fn(batch):
        """Custom collate function ensuring all data is on correct device."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch]).to(device)
        labels = [item['labels'] for item in batch]
        return {'pixel_values': pixel_values, 'labels': labels}
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,  # Very small batch size for stability
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=custom_collate_fn,
        pin_memory=False  # Disable pin_memory to avoid device issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=False
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(
        calibrated_model.calibration.parameters(), 
        lr=5e-5,  # Lower learning rate for fine-tuning
        weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.7, verbose=True
    )
    
    # Training tracking
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Save directory
    calibrated_save_dir = os.path.join(expert_weights_dir, 'moec_MOMO')
    os.makedirs(calibrated_save_dir, exist_ok=True)
    
    print(f"\nStarting calibration training for {epochs} epochs...")
    print(f"Save directory: {calibrated_save_dir}")
    print(f"Training dataset: {dataset_name}")
    print(f"Only training calibration layer parameters: {sum(p.numel() for p in calibrated_model.calibration.parameters()):,}")
    
    for epoch in range(epochs):
        # Training phase
        calibrated_model.train()
        calibrated_model.calibration.train()
        train_loss = 0.0
        train_steps = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            try:
                pixel_values = batch['pixel_values']  # Already on device
                labels = batch['labels']
                
                optimizer.zero_grad()
                
                # Get MoE output first (frozen)
                with torch.no_grad():
                    calibrated_model.moe.eval()
                    moe_output = calibrated_model.moe(pixel_values, labels=labels)
                
                # Apply calibration (trainable)
                refined_boxes, refined_logits = calibrated_model.calibration(
                    moe_output.pred_boxes.detach(), 
                    moe_output.logits.detach()
                )
                
                # Compute custom calibration loss with ground truth
                calibration_loss = compute_calibration_loss(refined_boxes, refined_logits, labels, device)
                
                # Add original MoE loss for stability
                total_loss = calibration_loss
                if moe_output.loss is not None:
                    total_loss = total_loss + 0.1 * moe_output.loss.detach()
                
                if torch.isfinite(total_loss) and total_loss > 0:
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(calibrated_model.calibration.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    
                    train_loss += total_loss.item()
                    train_steps += 1
                    
                    train_pbar.set_postfix({
                        'Loss': f"{total_loss.item():.4f}",
                        'Avg': f"{train_loss/train_steps:.4f}"
                    })
                    
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        avg_train_loss = train_loss / max(train_steps, 1)
        
        # Validation phase
        calibrated_model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                try:
                    pixel_values = batch['pixel_values']
                    labels = batch['labels']
                    
                    # Get MoE output
                    moe_output = calibrated_model.moe(pixel_values, labels=labels)
                    
                    # Apply calibration
                    refined_boxes, refined_logits = calibrated_model.calibration(
                        moe_output.pred_boxes, 
                        moe_output.logits
                    )
                    
                    # Compute validation loss
                    calibration_loss = compute_calibration_loss(refined_boxes, refined_logits, labels, device)
                    
                    if torch.isfinite(calibration_loss):
                        val_loss += calibration_loss.item()
                        val_steps += 1
                    
                    val_pbar.set_postfix({
                        'Loss': f"{calibration_loss.item():.4f}",
                        'Avg': f"{val_loss/max(val_steps,1):.4f}"
                    })
                    
                except Exception as e:
                    print(f"Error in validation step: {e}")
                    continue
        
        avg_val_loss = val_loss / max(val_steps, 1)
        
        # Learning rate scheduling
        if val_steps > 0:
            scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} ({train_steps} steps)")
        print(f"  Val Loss: {avg_val_loss:.4f} ({val_steps} steps)")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_steps > 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save calibration layer
            calibration_path = os.path.join(calibrated_save_dir, f'calibration_best_{dataset_name}.pth')
            torch.save(calibrated_model.calibration.state_dict(), calibration_path)
            
            # Save complete model
            complete_model_path = os.path.join(calibrated_save_dir, f'calibrated_moe_best_{dataset_name}.pth')
            torch.save({
                'calibration_state_dict': calibrated_model.calibration.state_dict(),
                'dataset_name': dataset_name,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'config': config
            }, complete_model_path)
            
            # Save in standard format
            standard_save_dir = os.path.join(calibrated_save_dir, f'standard_{dataset_name}')
            save_calibrated_moe_model(calibrated_model, image_processor, standard_save_dir, dataset_name)
            
            print(f"  ✅ New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 50)
    
    # Save final model in standard format
    final_standard_dir = os.path.join(calibrated_save_dir, f'final_{dataset_name}')
    save_calibrated_moe_model(calibrated_model, image_processor, final_standard_dir, dataset_name)
    
    print(f"\n" + "="*60)
    print("CALIBRATION TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Dataset: {dataset_name}")
    print(f"Calibration layer saved to: {calibration_path}")
    print(f"Complete model saved to: {complete_model_path}")
    print(f"Standard format saved to: {final_standard_dir}")
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
    calibration_path = os.path.join(calibrated_save_dir, f'calibration_best_{dataset_name}.pth')
    
    if not os.path.exists(calibration_path):
        calibration_final_path = os.path.join(calibrated_save_dir, f'calibration_final_{dataset_name}.pth')
        if os.path.exists(calibration_final_path):
            calibration_path = calibration_final_path
        else:
            raise FileNotFoundError(f"No calibration layer found for {dataset_name} in {calibrated_save_dir}. Run training first.")
    
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
        # Training mode - require dataset specification
        if not dataset:
            print("Error: Please specify --dataset for calibration training")
            print("Available datasets: CSAW, DMID, DDSM")
            return {'error': 'Dataset not specified'}
        
        epochs = epoch if epoch else 15
        print(f"Training calibration layer for {epochs} epochs on {dataset} dataset...")
        
        try:
            # Train calibration layer on specific dataset
            calibrated_model, best_loss = train_calibration_layer(
                config, device, expert_weights_dir, dataset, epochs
            )
            
            print(f"\nCalibration training completed with best loss: {best_loss:.4f}")
            
            # Test on the same dataset
            print(f"\nTesting calibrated MoE on: {dataset}")
            
            try:
                calibrated_results, original_results = evaluate_calibrated_moe(
                    config, device, dataset, expert_weights_dir
                )
                all_results = {
                    dataset: {
                        'calibrated': calibrated_results,
                        'original': original_results
                    }
                }
            except Exception as e:
                print(f"Evaluation on {dataset} failed: {e}")
                all_results = {dataset: {'error': str(e)}}
            
            print(f"\n" + "="*60)
            print("CALIBRATED MOE TRAINING & TESTING COMPLETED!")
            print("✅ Calibration layer trained and saved")
            print("✅ Performance comparison completed")
            print(f"✅ Trained on dataset: {dataset}")
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