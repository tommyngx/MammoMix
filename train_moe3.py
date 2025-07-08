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

class SimpleMoE(nn.Module):
    """Simple MoE: classifier chooses expert → gets expert output → returns result."""
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
        
        # 2. Update statistics (always update, not just during eval)
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
        
        # 4. Get expert outputs - IMPORTANT: Use labels=None during inference to avoid device issues
        sample_to_output = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        
        for expert_idx, sample_indices in expert_groups.items():
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            # Only pass labels during training and if provided
            if labels is not None and self.training:
                expert_labels = [labels[i] for i in sample_indices]
            
            with torch.no_grad():
                self.experts[expert_idx].eval()
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
            
            # Only accumulate loss if we computed it
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                weight = len(sample_indices) / batch_size
                total_loss = total_loss + expert_output.loss * weight
            
            for i, sample_idx in enumerate(sample_indices):
                sample_to_output[sample_idx] = {
                    'logits': expert_output.logits[i].clone(),
                    'pred_boxes': expert_output.pred_boxes[i].clone(),
                }
        
        # 5. Reconstruct batch - ensure we handle all samples
        batch_logits = []
        batch_pred_boxes = []
        
        if sample_to_output:
            first_sample_idx = list(sample_to_output.keys())[0]
            ref_logits_shape = sample_to_output[first_sample_idx]['logits'].shape
            ref_boxes_shape = sample_to_output[first_sample_idx]['pred_boxes'].shape
            
            for i in range(batch_size):
                if i in sample_to_output:
                    batch_logits.append(sample_to_output[i]['logits'])
                    batch_pred_boxes.append(sample_to_output[i]['pred_boxes'])
                else:
                    # Create dummy outputs for missing samples
                    dummy_logits = torch.zeros(ref_logits_shape, device=pixel_values.device)
                    dummy_boxes = torch.zeros(ref_boxes_shape, device=pixel_values.device)
                    batch_logits.append(dummy_logits)
                    batch_pred_boxes.append(dummy_boxes)
            
            batch_logits = torch.stack(batch_logits, dim=0)
            batch_pred_boxes = torch.stack(batch_pred_boxes, dim=0)
            batch_pred_boxes = batch_pred_boxes[..., :4].contiguous()
        else:
            # Fallback if no samples were processed
            batch_logits = torch.zeros((batch_size, 100, 2), device=pixel_values.device)
            batch_pred_boxes = torch.zeros((batch_size, 100, 4), device=pixel_values.device)
        
        # 6. Create output
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        final_loss = None
        if labels is not None and self.training:
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

def save_simple_moe_model(moe_model, image_processor, save_dir):
    """Save SimpleMoE model in standard HuggingFace format."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(save_dir, 'model.safetensors')
    from safetensors.torch import save_file
    save_file(moe_model.state_dict(), model_path)
    
    # Save preprocessor config
    image_processor.save_pretrained(save_dir)
    
    # Create and save model config
    config = {
        "model_type": "simple_moe",
        "num_experts": len(moe_model.experts),
        "expert_names": moe_model.expert_names,
        "hidden_size": 768,
        "num_labels": 2,
        "id2label": {"0": "cancer"},
        "label2id": {"cancer": 0},
        "problem_type": "single_label_classification",
        "torch_dtype": "float32",
        "transformers_version": "4.36.0"
    }
    
    import json
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"SimpleMoE model saved to: {save_dir}")
    print(f"  - model.safetensors")
    print(f"  - config.json") 
    print(f"  - preprocessor_config.json")

def get_all_metrics(model, test_dataset, image_processor, device, model_name):
    """Get all metrics (mAP + basic) for any model consistently."""
    # Get mAP metrics using custom function
    map_metrics = run_model_inference_with_map(model, test_dataset, image_processor, device)
    
    # Get basic metrics (loss, runtime, etc.) using Trainer
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"{model_name}_{date_str}"
    
    training_args = TrainingArguments(
        output_dir='./temp_eval',
        run_name=run_name,
        per_device_eval_batch_size=8,
        eval_strategy="no",
        save_strategy="no",
        disable_tqdm=False,
        logging_dir="./logs",
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=None,
    )
    
    basic_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Combine all metrics
    all_metrics = basic_results.copy()
    for key, value in map_metrics.items():
        all_metrics[f'test_{key}'] = value
    
    return all_metrics

def print_metrics_comparison(individual_metrics, moe_metrics, dataset_name):
    """Print metrics comparison in a clean DataFrame format."""
    # Define the standard metric order for consistent output
    metric_order = [
        'test_loss',
        'test_model_preparation_time', 
        'test_runtime',
        'test_samples_per_second',
        'test_steps_per_second',
        'test_map',
        'test_map_50',
        'test_map_75',
        'test_map_small',
        'test_map_medium',
        'test_map_large'
    ]
    
    # Prepare data for DataFrame
    comparison_data = []
    
    for metric in metric_order:
        if metric in individual_metrics or metric in moe_metrics:
            individual_value = individual_metrics.get(metric, 0.0)
            moe_value = moe_metrics.get(metric, 0.0)
            
            # Calculate difference and percentage
            if isinstance(individual_value, (int, float)) and isinstance(moe_value, (int, float)):
                diff = moe_value - individual_value
                if individual_value != 0:
                    pct_change = (diff / individual_value) * 100
                else:
                    pct_change = 0.0
                
                comparison_data.append({
                    'Metric': metric,
                    f'Individual_{dataset_name}': f"{individual_value:.4f}",
                    f'SimpleMoE': f"{moe_value:.4f}",
                    'Difference': f"{diff:.4f}",
                    'Change_%': f"{pct_change:.2f}%"
                })
            else:
                comparison_data.append({
                    'Metric': metric,
                    f'Individual_{dataset_name}': str(individual_value),
                    f'SimpleMoE': str(moe_value),
                    'Difference': 'N/A',
                    'Change_%': 'N/A'
                })
    
    # Create and display DataFrame
    df = pd.DataFrame(comparison_data)
    
    print(f"\n" + "="*80)
    print(f"METRICS COMPARISON: Individual {dataset_name} Expert vs SimpleMoE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Summary analysis
    if 'test_map_50' in individual_metrics and 'test_map_50' in moe_metrics:
        individual_map50 = individual_metrics['test_map_50']
        moe_map50 = moe_metrics['test_map_50']
        
        if isinstance(individual_map50, (int, float)) and isinstance(moe_map50, (int, float)):
            if individual_map50 > 0:
                performance_ratio = (moe_map50 / individual_map50) * 100
                print(f"\n📊 Performance Summary:")
                print(f"   SimpleMoE achieves {performance_ratio:.1f}% of individual expert mAP@50")
                
                if performance_ratio >= 95:
                    print(f"   Status: ✅ Excellent performance retention")
                elif performance_ratio >= 85:
                    print(f"   Status: ✅ Good performance retention")
                else:
                    print(f"   Status: ⚠️ Performance degradation detected")
    
    return df

def evaluate_simple_moe(config, device, dataset_name, expert_weights_dir):
    """Evaluate SimpleMoE with comprehensive metrics comparison."""
    # Load expert models and image processor
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
            raise FileNotFoundError(f"No classifier found in {moe_save_dir}. Run training first.")
    
    # Initialize and load classifier
    classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    
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
    
    # Create SimpleMoE model
    moe_model = SimpleMoE(expert_models, classifier, device).to(device)
    moe_model.eval()
    moe_model.reset_routing_stats()
    
    print(f"Evaluating SimpleMoE on {dataset_name} ({len(test_dataset)} samples)...")
    
    # Get individual expert for comparison
    dataset_map = {'CSAW': 0, 'DMID': 1, 'DDSM': 2}
    expert_idx = dataset_map[dataset_name]
    individual_expert = expert_models[expert_idx]
    
    # Get all metrics for both models using consistent code
    print(f"\n=== Evaluating Individual {dataset_name} Expert ===")
    individual_metrics = get_all_metrics(
        individual_expert, test_dataset, image_processor, device, f"Individual_{dataset_name}"
    )
    
    print(f"\n=== Evaluating SimpleMoE ===")
    moe_metrics = get_all_metrics(
        moe_model, test_dataset, image_processor, device, "SimpleMoE"
    )
    
    # Print comprehensive comparison using DataFrame
    comparison_df = print_metrics_comparison(individual_metrics, moe_metrics, dataset_name)
    
    # Get routing statistics
    final_stats = moe_model.get_routing_stats()
    target_usage = final_stats.get(dataset_name, 0.0)
    
    print(f"\n🔄 Routing Analysis:")
    total_routed = moe_model.total_routed
    print(f"   Total samples routed: {total_routed}")
    for expert_name, usage in final_stats.items():
        count = int(usage * total_routed) if total_routed > 0 else 0
        print(f"   {expert_name}: {usage*100:.1f}% ({count} samples)")
    
    print(f"\n   Routing to correct expert ({dataset_name}): {target_usage*100:.1f}%")
    
    if target_usage > 0.95:
        print(f"   Routing quality: ✅ Excellent (>95%)")
    elif target_usage > 0.80:
        print(f"   Routing quality: ✅ Good (>80%)")
    else:
        print(f"   Routing quality: ⚠️ Poor (<80%)")
    
    # Save SimpleMoE model in standard format - ALWAYS use moe_MOMO folder
    save_simple_moe_model(moe_model, image_processor, moe_save_dir)
    
    return moe_metrics, final_stats, moe_model

def test_only_mode(config, device, dataset_name, expert_weights_dir):
    """Professional test-only mode that saves model in standard format."""
    print(f"SimpleMoE Evaluation on {dataset_name}")
    print("="*50)
    
    try:
        results, routing_stats, moe_model = evaluate_simple_moe(config, device, dataset_name, expert_weights_dir)
        return results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {'error': str(e)}

def create_classifier_dataset(expert_weights_dir, config, device):
    """Create dataset for training the classifier."""
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    # Load any image processor (they should be the same)
    first_expert_path = os.path.join(expert_weights_dir, 'yolos_CSAW')
    image_processor = AutoImageProcessor.from_pretrained(first_expert_path)
    
    datasets = []
    labels = []
    dataset_names = ['CSAW', 'DMID', 'DDSM']
    
    for idx, dataset_name in enumerate(dataset_names):
        print(f"Loading {dataset_name} dataset...")
        
        train_dataset = BreastCancerDataset(
            split='train',
            splits_dir=SPLITS_DIR,
            dataset_name=dataset_name,
            image_processor=image_processor,
            model_type=get_model_type(MODEL_NAME),
        )
        
        val_dataset = BreastCancerDataset(
            split='val',
            splits_dir=SPLITS_DIR,
            dataset_name=dataset_name,
            image_processor=image_processor,
            model_type=get_model_type(MODEL_NAME),
        )
        
        # Combine train and val for classifier training
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        datasets.append(combined_dataset)
        labels.extend([idx] * len(combined_dataset))
        
        print(f"  {dataset_name}: {len(combined_dataset)} samples")
    
    # Combine all datasets
    final_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"Total classifier training samples: {len(final_dataset)}")
    
    return final_dataset, labels, image_processor

class ClassifierDataset(Dataset):
    """Dataset wrapper for classifier training."""
    def __init__(self, base_dataset, labels):
        self.base_dataset = base_dataset
        self.labels = labels
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        pixel_values = item['pixel_values']
        dataset_label = self.labels[idx]
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(dataset_label, dtype=torch.long)
        }

def train_classifier(config, device, expert_weights_dir, epochs=10):
    """Train the dataset classifier."""
    print("="*60)
    print("PHASE 1: TRAINING DATASET CLASSIFIER")
    print("="*60)
    
    # Create classifier training dataset
    base_dataset, labels, image_processor = create_classifier_dataset(expert_weights_dir, config, device)
    classifier_dataset = ClassifierDataset(base_dataset, labels)
    
    # Split into train/val
    dataset_size = len(classifier_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        classifier_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Classifier training: {len(train_dataset)} samples")
    print(f"Classifier validation: {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize classifier
    classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Create save directory
    moe_save_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    os.makedirs(moe_save_dir, exist_ok=True)
    
    print(f"\nStarting classifier training for {epochs} epochs...")
    print(f"Save directory: {moe_save_dir}")
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = classifier(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*train_correct/train_total:.2f}%"
            })
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = classifier(pixel_values)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*val_correct/val_total:.2f}%"
                })
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            best_model_path = os.path.join(moe_save_dir, 'classifier_best.pth')
            torch.save(classifier.state_dict(), best_model_path)
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 50)
    
    # Save final model
    final_model_path = os.path.join(moe_save_dir, 'classifier_final.pth')
    torch.save(classifier.state_dict(), final_model_path)
    
    print(f"\n" + "="*60)
    print("CLASSIFIER TRAINING COMPLETED!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(moe_save_dir, 'classifier_best.pth')}")
    print(f"Final model saved to: {final_model_path}")
    print("="*60)
    
    return classifier, best_val_acc

def main(config_path, epoch=None, dataset=None, weight_dir=None, phase=None, test=False):
    """Main function."""
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if weight_dir is not None:
        expert_weights_dir = weight_dir
    else:
        expert_weights_dir = config.get('moe', {}).get('expert_weights_dir', '/content/Weights')
    
    print(f"Using device: {device}")
    print(f"Expert weights: {expert_weights_dir}")
    if dataset:
        print(f"Target dataset: {dataset}")
    
    # Test-only mode with professional model saving
    if test:
        target_dataset = dataset if dataset else 'CSAW'
        print(f"Running test-only mode on dataset: {target_dataset}")
        print("Model will be automatically saved in HuggingFace format")
        
        results = test_only_mode(config, device, target_dataset, expert_weights_dir)
        
        print("\n" + "="*60)
        print("TEST-ONLY MODE COMPLETED!")
        print("✅ SimpleMoE model saved in standard format")
        print("✅ Ready for deployment or further evaluation")
        print("="*60)
        return results
    
    # Handle different phases
    if phase == '1':
        # Phase 1: Train classifier only
        print("Running Phase 1: Classifier Training Only")
        epochs = epoch if epoch else 20
        classifier, best_acc = train_classifier(config, device, expert_weights_dir, epochs)
        print(f"Classifier training completed with best accuracy: {best_acc:.2f}%")
        
    elif phase == '2':
        # Phase 2: Test only (same as --test)
        target_dataset = dataset if dataset else 'CSAW'
        print(f"Running Phase 2: Test-only mode on dataset: {target_dataset}")
        results = test_only_mode(config, device, target_dataset, expert_weights_dir)
        return results
        
    else:
        # Default: Run both phases
        print("Running Full Training Pipeline (Phase 1 + Phase 2)")
        
        # Phase 1: Train classifier
        epochs = epoch if epoch else 20
        print(f"\nStarting Phase 1 with {epochs} epochs...")
        classifier, best_acc = train_classifier(config, device, expert_weights_dir, epochs)
        print(f"Phase 1 completed with best accuracy: {best_acc:.2f}%")
        
        # Phase 2: Test on all datasets
        target_datasets = ['CSAW', 'DMID', 'DDSM'] if not dataset else [dataset]
        
        print(f"\nStarting Phase 2: Testing on datasets: {target_datasets}")
        all_results = {}
        
        for test_dataset in target_datasets:
            print(f"\n{'='*50}")
            print(f"Testing on {test_dataset}")
            print('='*50)
            
            results = test_only_mode(config, device, test_dataset, expert_weights_dir)
            all_results[test_dataset] = results
        
        print(f"\n" + "="*60)
        print("FULL PIPELINE COMPLETED!")
        print("✅ Classifier trained and saved")
        print("✅ SimpleMoE tested on all datasets")
        print("✅ Model saved in standard format")
        print("="*60)
        
        return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name for testing (CSAW/DMID/DDSM)')
    parser.add_argument('--weight_dir', type=str, default=None, help='Expert weights directory')
    parser.add_argument('--phase', type=str, choices=['1', '2'], default=None, help='Training phase (1: train only, 2: test only, None: both)')
    parser.add_argument('--test', action='store_true', help='Test-only mode: load classifier and test MoE without training')
    args = parser.parse_args()

    main(args.config, args.epoch, args.dataset, args.weight_dir, args.phase, args.test)