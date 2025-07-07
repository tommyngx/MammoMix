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
from evaluation import get_eval_compute_metrics_fn

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class SimpleDatasetClassifier(nn.Module):
    """
    Simple CNN to classify which dataset an image belongs to.
    Much simpler than the complex router approach.
    """
    def __init__(self, num_classes=3, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.class_names = ['CSAW', 'DMID', 'DDSM']
        
        # Simple CNN for dataset classification
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 640 -> 320
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 320 -> 160
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 160 -> 80
            
            # Fourth block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 80 -> 4x4
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
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"SimpleDatasetClassifier initialized for {num_classes} datasets")
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total parameters: {total_params / 1e6:.2f}M')
    
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
    """
    Simple MoE that uses the trained dataset classifier for routing.
    Based on train_moe2.py's ImageRouterMoE structure.
    """
    def __init__(self, expert_models, dataset_classifier, device):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)
        self.classifier = dataset_classifier
        self.device = device
        self.expert_names = ['CSAW', 'DMID', 'DDSM']
        
        # Freeze everything except classifier
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        # Statistics
        self.routing_counts = torch.zeros(3, device=device)
        self.total_routed = 0
        
        print(f"SimpleMoE initialized with dataset classifier routing")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for compatibility with Transformers Trainer."""
        pass
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for compatibility with Transformers Trainer."""
        pass
    
    def forward(self, pixel_values, labels=None, return_routing=False):
        """Forward pass with dataset classification routing - based on train_moe2.py structure."""
        batch_size = pixel_values.shape[0]
        
        # Get dataset predictions using our simple classifier
        with torch.no_grad():
            dataset_logits = self.classifier(pixel_values)
            dataset_probs = F.softmax(dataset_logits, dim=1)
            expert_choices = torch.argmax(dataset_probs, dim=1)
        
        # Update statistics (only during eval to avoid overhead)
        if not self.training:
            for choice in expert_choices:
                self.routing_counts[choice] += 1
            self.total_routed += batch_size
        
        # Group samples by expert choice (same as train_moe2.py)
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # Print routing distribution (reduced frequency like train_moe2.py)
        if not self.training and self.total_routed % 500 == 0:
            usage_pct = (self.routing_counts / self.total_routed * 100).cpu().numpy()
            print(f"Routing: CSAW={usage_pct[0]:.1f}%, DMID={usage_pct[1]:.1f}%, DDSM={usage_pct[2]:.1f}%")
        
        # Get expert outputs for each group (same as train_moe2.py)
        batch_outputs = {}
        total_detection_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for expert_idx, sample_indices in expert_groups.items():
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            # Get expert output (frozen expert, no gradients needed)
            with torch.no_grad():
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
                expert_output = self._fix_expert_output_dimensions(expert_output)
            
            batch_outputs[expert_idx] = {
                'output': expert_output,
                'indices': sample_indices
            }
            
            # Accumulate detection loss
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                weight = len(sample_indices) / batch_size
                total_detection_loss = total_detection_loss + expert_output.loss * weight
        
        # Reconstruct batch outputs in original order (same as train_moe2.py)
        if not batch_outputs:
            raise RuntimeError("No expert outputs generated")
        
        first_expert_idx = list(batch_outputs.keys())[0]
        reference_output = batch_outputs[first_expert_idx]['output']
        
        # Initialize batch tensors (same as train_moe2.py)
        num_queries = reference_output.logits.shape[1]
        num_classes = reference_output.logits.shape[2]
        
        batch_logits = torch.zeros(batch_size, num_queries, num_classes, 
                                 device=pixel_values.device, dtype=reference_output.logits.dtype)
        batch_pred_boxes = torch.zeros(batch_size, num_queries, 4, 
                                     device=pixel_values.device, dtype=reference_output.pred_boxes.dtype)
        batch_last_hidden_state = None
        
        if hasattr(reference_output, 'last_hidden_state') and reference_output.last_hidden_state is not None:
            batch_last_hidden_state = torch.zeros(batch_size, reference_output.last_hidden_state.shape[1], 
                                                 reference_output.last_hidden_state.shape[2], 
                                                 device=pixel_values.device, dtype=reference_output.last_hidden_state.dtype)
        
        # Fill batch tensors with expert outputs (same as train_moe2.py)
        for expert_idx, group_data in batch_outputs.items():
            expert_output = group_data['output']
            sample_indices = group_data['indices']
            
            batch_logits[sample_indices] = expert_output.logits
            batch_pred_boxes[sample_indices] = expert_output.pred_boxes
            
            if batch_last_hidden_state is not None and hasattr(expert_output, 'last_hidden_state'):
                batch_last_hidden_state[sample_indices] = expert_output.last_hidden_state
        
        # Final safety check (same as train_moe2.py)
        assert batch_pred_boxes.shape[-1] == 4, f"pred_boxes has {batch_pred_boxes.shape[-1]} dims, expected 4"
        batch_pred_boxes = batch_pred_boxes[..., :4].contiguous()
        
        # Create output (same as train_moe2.py)
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        combined_output = YolosObjectDetectionOutput(
            loss=total_detection_loss,
            logits=batch_logits,
            pred_boxes=batch_pred_boxes,
            last_hidden_state=batch_last_hidden_state
        )
        
        if return_routing:
            routing_info = {
                'probs': dataset_probs,
                'choices': expert_choices,
                'groups': expert_groups,
                'stats': self.get_routing_stats()
            }
            return combined_output, routing_info
        else:
            return combined_output
    
    def _fix_expert_output_dimensions(self, expert_output):
        """Fix expert output dimensions to ensure pred_boxes has exactly 4 dimensions (from train_moe2.py)."""
        if expert_output.pred_boxes.shape[-1] != 4:
            fixed_pred_boxes = expert_output.pred_boxes[..., :4].contiguous()
            fixed_output = type(expert_output)(
                loss=expert_output.loss,
                logits=expert_output.logits,
                pred_boxes=fixed_pred_boxes,
                last_hidden_state=expert_output.last_hidden_state if hasattr(expert_output, 'last_hidden_state') else None
            )
            return fixed_output
        return expert_output
    
    def get_routing_stats(self):
        """Get current routing statistics (same as train_moe2.py)."""
        if self.total_routed == 0:
            return {name: 0.0 for name in self.expert_names}
        
        stats = {}
        usage = (self.routing_counts / self.total_routed).cpu().numpy()
        for i, name in enumerate(self.expert_names):
            stats[name] = float(usage[i])
        return stats
    
    def reset_routing_stats(self):
        """Reset routing statistics (same as train_moe2.py)."""
        self.routing_counts.zero_()
        self.total_routed = 0

def load_expert_models(weight_dir, device):
    """Load expert models."""
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM']
    expert_paths = [os.path.join(weight_dir, name) for name in expert_names]
    
    models = []
    processors = []
    
    for path in expert_paths:
        print(f"Loading expert from: {path}")
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

def create_dataset_with_labels(config, image_processor, dataset_name, split, epoch=None):
    """Create dataset with dataset labels (0=CSAW, 1=DMID, 2=DDSM)."""
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    dataset = BreastCancerDataset(
        split=split,
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    
    # Create simple dataset labels
    dataset_map = {'CSAW': 0, 'DMID': 1, 'DDSM': 2}
    dataset_label = dataset_map[dataset_name]
    
    # Create labels for all samples
    labels = [dataset_label] * len(dataset)
    
    return dataset, labels

def train_dataset_classifier(config, device, epoch=None, weight_dir=None):
    """
    Train simple CNN to classify datasets.
    Much simpler than the complex expert evaluation approach.
    """
    print("\n" + "="*50)
    print("PHASE 1: Training Simple Dataset Classifier")
    print("="*50)
    
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    # Create datasets with simple labels
    all_train_datasets = []
    all_train_labels = []
    all_val_datasets = []
    all_val_labels = []
    
    for dataset_name in ['CSAW', 'DMID', 'DDSM']:
        print(f"Loading {dataset_name} data...")
        
        # Train data
        train_dataset, train_labels = create_dataset_with_labels(
            config, image_processor, dataset_name, 'train', epoch
        )
        all_train_datasets.append(train_dataset)
        all_train_labels.extend(train_labels)
        print(f"{dataset_name} train: {len(train_dataset)} samples")
        
        # Val data
        val_dataset, val_labels = create_dataset_with_labels(
            config, image_processor, dataset_name, 'val', epoch
        )
        all_val_datasets.append(val_dataset)
        all_val_labels.extend(val_labels)
        print(f"{dataset_name} val: {len(val_dataset)} samples")
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_val_dataset = ConcatDataset(all_val_datasets)
    
    print(f"Combined train: {len(combined_train_dataset)} samples")
    print(f"Combined val: {len(combined_val_dataset)} samples")
    
    # Count distribution
    train_counts = np.bincount(all_train_labels, minlength=3)
    print(f"Train distribution: CSAW={train_counts[0]}, DMID={train_counts[1]}, DDSM={train_counts[2]}")
    
    # Create classifier
    classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = epoch if epoch else 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Custom dataset class for labels
    class LabeledDataset(Dataset):
        def __init__(self, base_dataset, labels):
            self.base_dataset = base_dataset
            self.labels = labels
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            item = self.base_dataset[idx]
            item['dataset_label'] = self.labels[idx]
            return item
    
    # Custom collate function to handle dataset_label
    def custom_collate_fn(batch):
        # Extract dataset labels
        dataset_labels = [item['dataset_label'] for item in batch]
        
        # Remove dataset_label from items before using standard collate
        for item in batch:
            del item['dataset_label']
        
        # Use standard collate function
        collated = collate_fn(batch)
        
        # Add dataset labels back
        collated['dataset_label'] = dataset_labels
        
        return collated
    
    # Create labeled datasets
    train_labeled = LabeledDataset(combined_train_dataset, all_train_labels)
    val_labeled = LabeledDataset(combined_val_dataset, all_val_labels)
    
    # Training loop
    for epoch_idx in range(num_epochs):
        print(f"\nEpoch {epoch_idx + 1}/{num_epochs}")
        
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loader = DataLoader(train_labeled, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
        
        for batch in tqdm(train_loader, desc="Training"):
            pixel_values = batch['pixel_values'].to(device)
            dataset_labels = torch.tensor(batch['dataset_label'], device=device)
            
            optimizer.zero_grad()
            outputs = classifier(pixel_values)
            loss = criterion(outputs, dataset_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == dataset_labels).sum().item()
            train_total += dataset_labels.size(0)
        
        scheduler.step()
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_loader = DataLoader(val_labeled, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(device)
                dataset_labels = torch.tensor(batch['dataset_label'], device=device)
                
                outputs = classifier(pixel_values)
                loss = criterion(outputs, dataset_labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == dataset_labels).sum().item()
                val_total += dataset_labels.size(0)
        
        # Calculate accuracies
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch_idx + 1
            
            # Save classifier in moe_MOMO directory
            moe_save_dir = os.path.join(weight_dir, 'moe_MOMO')
            os.makedirs(moe_save_dir, exist_ok=True)
            classifier_save_path = os.path.join(moe_save_dir, 'classifier_best.pth')
            torch.save(classifier.state_dict(), classifier_save_path)
            print(f"Best classifier saved to {classifier_save_path} (acc: {val_acc:.4f})")
        
        # Test MoE after each epoch
        print(f"\n=== Testing MoE with Current Classifier (Epoch {epoch_idx + 1}) ===")
        test_moe_with_classifier(config, classifier, device, epoch_idx + 1, weight_dir)
    
    print(f"\nBest classifier accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # Load best classifier from moe_MOMO path
    moe_save_dir = os.path.join(weight_dir, 'moe_MOMO')
    best_path = os.path.join(moe_save_dir, 'classifier_best.pth')
    classifier.load_state_dict(torch.load(best_path, map_location=device))
    classifier.eval()
    
    return classifier

def test_moe_with_classifier(config, classifier, device, epoch_num, weight_dir, dataset_name=None):
    """Test MoE with current classifier on specific dataset - like train_moe2.py evaluation."""
    try:
        # Load expert models
        expert_models, _ = load_expert_models(weight_dir, device)
        
        # Create MoE
        moe_model = SimpleMoE(expert_models, classifier, device).to(device)
        moe_model.eval()
        
        # Create test dataset - use specific dataset if provided
        MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        
        if dataset_name:
            # Test on specific dataset only
            test_dataset, _ = create_dataset_with_labels(config, image_processor, dataset_name, 'test')
            print(f"Testing MoE on {dataset_name} dataset: {len(test_dataset)} samples")
        else:
            # Test on combined datasets (original behavior)
            test_datasets = []
            for ds_name in ['CSAW', 'DMID', 'DDSM']:
                test_dataset, _ = create_dataset_with_labels(config, image_processor, ds_name, 'test')
                test_datasets.append(test_dataset)
            
            from torch.utils.data import ConcatDataset
            test_dataset = ConcatDataset(test_datasets)
            print(f"Testing MoE on combined datasets: {len(test_dataset)} samples")
        
        # Quick evaluation
        training_args = TrainingArguments(
            output_dir='./temp_eval',
            per_device_eval_batch_size=8,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
        )
        
        eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
        
        trainer = Trainer(
            model=moe_model,
            args=training_args,
            processing_class=image_processor,
            data_collator=collate_fn,
            compute_metrics=eval_compute_metrics_fn,
        )
        
        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
        
        test_name = dataset_name if dataset_name else "Combined"
        print(f"Epoch {epoch_num} MoE Results on {test_name}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # Print routing stats
        stats = moe_model.get_routing_stats()
        print(f"Routing: CSAW={stats['CSAW']*100:.1f}%, DMID={stats['DMID']*100:.1f}%, DDSM={stats['DDSM']*100:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"MoE test failed: {e}")
        return None

def run_moe_like_train_moe2(config, classifier, device, dataset_name, expert_weights_dir, epoch=None):
    """Run MoE training/testing exactly like train_moe2.py but with our simple classifier."""
    print(f"\n=== Running MoE Training/Testing like train_moe2.py on {dataset_name} ===")
    
    # Load expert models (same as train_moe2.py)
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    image_processor = expert_processors[0]
    
    # Create MoE model (same structure as train_moe2.py)
    model = SimpleMoE(expert_models, classifier, device).to(device)
    
    # Create datasets exactly like train_moe2.py
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    print(f"Creating datasets for {dataset_name}...")
    train_dataset = BreastCancerDataset(
        split='train',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    
    val_dataset = BreastCancerDataset(
        split='val',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    
    print(f'Train: {len(train_dataset)} samples')
    print(f'Val: {len(val_dataset)} samples') 
    print(f'Test: {len(test_dataset)} samples')
    
    # Training setup exactly like train_moe2.py - save to moe_MOMO directory
    output_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"SimpleMoE_{dataset_name}_{date_str}"
    
    # Use exact same training arguments as train_moe2.py
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=epoch if epoch else 10,
        per_device_train_batch_size=16,  # Same as train_moe2.py
        per_device_eval_batch_size=16,   # Same as train_moe2.py
        learning_rate=1e-3,  # Same as train_moe2.py
        weight_decay=0.0,    # Same as train_moe2.py
        warmup_ratio=0.0,    # Same as train_moe2.py
        lr_scheduler_type='constant',  # Same as train_moe2.py
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model='eval_map_50',
        greater_is_better=True,
        fp16=False,  # Same as train_moe2.py
        bf16=False,  # Same as train_moe2.py
        dataloader_num_workers=0,
        gradient_accumulation_steps=1,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        save_safetensors=True,
    )
    
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    print(f"\n=== Training SimpleMoE on {dataset_name} ===")
    trainer.train()
    
    print(f"\n=== Testing SimpleMoE on {dataset_name} test set ===")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Print test results exactly like train_moe2.py
    print(f"\n=== Test Results on {dataset_name} ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Print final routing statistics (same as train_moe2.py)
    print(f"\n=== Final Routing Statistics on {dataset_name} ===")
    final_stats = model.get_routing_stats()
    for expert_name, usage in final_stats.items():
        print(f"{expert_name}: {usage*100:.1f}%")
    
    # Save final results in moe_MOMO directory
    moe_save_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    os.makedirs(moe_save_dir, exist_ok=True)
    
    # Save final classifier
    final_classifier_path = os.path.join(moe_save_dir, 'classifier_final.pth')
    torch.save(classifier.state_dict(), final_classifier_path)
    
    # Save final results
    final_results_path = os.path.join(moe_save_dir, 'results_final.json')
    with open(final_results_path, 'w') as f:
        json_results = {}
        for k, v in test_results.items():
            if isinstance(v, (np.integer, np.floating, np.ndarray)):
                json_results[k] = float(v)
            else:
                json_results[k] = v
        json_results['routing_stats'] = final_stats
        json.dump(json_results, f, indent=2)
    
    print(f"Final MoE model and results saved to {moe_save_dir}")
    print(f"Classifier saved to: {final_classifier_path}")
    print(f"Results saved to: {final_results_path}")
    
    return test_results

def test_only_mode(config, device, dataset_name, expert_weights_dir):
    """Test-only mode: Load classifier and test MoE without any training."""
    print("\n" + "="*60)
    print("TEST-ONLY MODE: LOADING CLASSIFIER AND TESTING MoE")
    print("="*60)
    
    # Load classifier from moe_MOMO path
    moe_save_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    classifier_path = os.path.join(moe_save_dir, 'classifier_best.pth')
    
    if not os.path.exists(classifier_path):
        # Try to find classifier_final.pth as backup
        classifier_final_path = os.path.join(moe_save_dir, 'classifier_final.pth')
        if os.path.exists(classifier_final_path):
            classifier_path = classifier_final_path
            print(f"Using final classifier: {classifier_path}")
        else:
            raise FileNotFoundError(f"No classifier found in {moe_save_dir}. Run training first.")
    
    classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()
    print(f"Loaded classifier from: {classifier_path}")
    
    # Load expert models
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    image_processor = expert_processors[0]
    
    # Create MoE model
    model = SimpleMoE(expert_models, classifier, device).to(device)
    model.eval()
    print("Created SimpleMoE with loaded classifier")
    
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
    
    print(f'Test dataset ({dataset_name}): {len(test_dataset)} samples')
    
    # Quick evaluation setup (same as train_moe2.py test mode)
    training_args = TrainingArguments(
        output_dir='./temp_test',
        per_device_eval_batch_size=16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        fp16=False,
        bf16=False,
        eval_do_concat_batches=False,
        disable_tqdm=False,
    )
    
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    print(f"\n=== Testing SimpleMoE on {dataset_name} test set ===")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Print test results exactly like train_moe2.py
    print(f"\n=== Test Results on {dataset_name} ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Print routing statistics
    print(f"\n=== Routing Statistics on {dataset_name} ===")
    final_stats = model.get_routing_stats()
    for expert_name, usage in final_stats.items():
        print(f"{expert_name}: {usage*100:.1f}%")
    
    # Save test results
    test_results_path = os.path.join(moe_save_dir, f'test_results_{dataset_name}.json')
    with open(test_results_path, 'w') as f:
        json_results = {}
        for k, v in test_results.items():
            if isinstance(v, (np.integer, np.floating, np.ndarray)):
                json_results[k] = float(v)
            else:
                json_results[k] = v
        json_results['routing_stats'] = final_stats
        json_results['dataset'] = dataset_name
        json.dump(json_results, f, indent=2)
    
    print(f"Test results saved to: {test_results_path}")
    
    return test_results

def main(config_path, epoch=None, dataset=None, weight_dir=None, phase=None, test=False):
    """Main function - automatically runs both phases if not specified."""
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
    
    # Test-only mode
    if test:
        target_dataset = dataset if dataset else 'CSAW'
        print(f"Running test-only mode on dataset: {target_dataset}")
        results = test_only_mode(config, device, target_dataset, expert_weights_dir)
        print("\n" + "="*60)
        print("TEST-ONLY MODE COMPLETED!")
        print("="*60)
        return results
    
    # If no phase specified, run both phases automatically
    if phase is None:
        print("No phase specified - running both Phase 1 (training classifier) and Phase 2 (MoE like train_moe2)")
        run_phase_1 = True
        run_phase_2 = True
    elif phase == "1":
        run_phase_1 = True
        run_phase_2 = False
    elif phase == "2":
        run_phase_1 = False
        run_phase_2 = True
    
    # Phase 1: Train dataset classifier
    if run_phase_1:
        print("\n" + "="*60)
        print("STARTING PHASE 1: TRAINING DATASET CLASSIFIER")
        print("="*60)
        classifier = train_dataset_classifier(config, device, epoch, expert_weights_dir)
        print("Phase 1 completed successfully!")
        
        if not run_phase_2:
            print("\nTraining completed. Use --phase 2 to test the MoE model.")
            return
    
    # Phase 2: Load classifier and run MoE like train_moe2.py
    if run_phase_2:
        print("\n" + "="*60)
        print("STARTING PHASE 2: MoE TRAINING/TESTING LIKE TRAIN_MOE2.PY")
        print("="*60)
        
        # Load classifier from moe_MOMO path
        moe_save_dir = os.path.join(weight_dir, 'moe_MOMO')
        classifier_path = os.path.join(moe_save_dir, 'classifier_best.pth')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found at {classifier_path}. Run Phase 1 first.")
        
        classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        print(f"Loaded best classifier from: {classifier_path}")
        
        # Use specified dataset or default to CSAW
        target_dataset = dataset if dataset else 'CSAW'
        print(f"Running MoE training/testing on dataset: {target_dataset}")
        
        # Run MoE exactly like train_moe2.py
        results = run_moe_like_train_moe2(config, classifier, device, target_dataset, expert_weights_dir, epoch)
        
        print("Phase 2 completed successfully!")
        print(f"All files saved in: {os.path.join(expert_weights_dir, 'moe_MOMO')}")
    
    print("\n" + "="*60)
    print("ALL PHASES COMPLETED SUCCESSFULLY!")
    print("="*60)

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