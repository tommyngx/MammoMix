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
    Much cleaner than the complex router approach.
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
        pass
    
    def gradient_checkpointing_disable(self):
        pass
    
    def forward(self, pixel_values, labels=None, return_routing=False):
        """Forward pass with dataset classification routing."""
        batch_size = pixel_values.shape[0]
        
        # Get dataset predictions
        with torch.no_grad():
            dataset_logits = self.classifier(pixel_values)
            dataset_probs = F.softmax(dataset_logits, dim=1)
            expert_choices = torch.argmax(dataset_probs, dim=1)
        
        # Update statistics
        if not self.training:
            for choice in expert_choices:
                self.routing_counts[choice] += 1
            self.total_routed += batch_size
        
        # Group by expert
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # Print routing stats
        if not self.training and self.total_routed % 100 == 0:
            usage_pct = (self.routing_counts / self.total_routed * 100).cpu().numpy()
            print(f"Routing: CSAW={usage_pct[0]:.1f}%, DMID={usage_pct[1]:.1f}%, DDSM={usage_pct[2]:.1f}%")
        
        # Get expert outputs
        batch_outputs = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for expert_idx, sample_indices in expert_groups.items():
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            # Get expert output
            with torch.no_grad():
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
            
            batch_outputs[expert_idx] = {
                'output': expert_output,
                'indices': sample_indices
            }
            
            # Accumulate loss
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                weight = len(sample_indices) / batch_size
                total_loss = total_loss + expert_output.loss * weight
        
        # Reconstruct batch outputs
        if not batch_outputs:
            raise RuntimeError("No expert outputs generated")
        
        first_expert_idx = list(batch_outputs.keys())[0]
        reference_output = batch_outputs[first_expert_idx]['output']
        
        # Initialize batch tensors
        num_queries = reference_output.logits.shape[1]
        num_classes = reference_output.logits.shape[2]
        
        batch_logits = torch.zeros(batch_size, num_queries, num_classes, 
                                 device=pixel_values.device, dtype=reference_output.logits.dtype)
        batch_pred_boxes = torch.zeros(batch_size, num_queries, 4, 
                                     device=pixel_values.device, dtype=reference_output.pred_boxes.dtype)
        
        # Fill batch tensors
        for expert_idx, group_data in batch_outputs.items():
            expert_output = group_data['output']
            sample_indices = group_data['indices']
            
            batch_logits[sample_indices] = expert_output.logits
            batch_pred_boxes[sample_indices] = expert_output.pred_boxes[..., :4]
        
        # Create output
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        combined_output = YolosObjectDetectionOutput(
            loss=total_loss,
            logits=batch_logits,
            pred_boxes=batch_pred_boxes,
            last_hidden_state=None
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
    
    def get_routing_stats(self):
        """Get routing statistics."""
        if self.total_routed == 0:
            return {name: 0.0 for name in self.expert_names}
        
        stats = {}
        usage = (self.routing_counts / self.total_routed).cpu().numpy()
        for i, name in enumerate(self.expert_names):
            stats[name] = float(usage[i])
        return stats

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
            
            # Save classifier in organized directory structure
            moe_save_dir = os.path.join(weight_dir, 'moe_combined')
            os.makedirs(moe_save_dir, exist_ok=True)
            classifier_save_path = os.path.join(moe_save_dir, 'classifier_best.pth')
            torch.save(classifier.state_dict(), classifier_save_path)
            print(f"Best classifier saved to {classifier_save_path} (acc: {val_acc:.4f})")
        
        # Test MoE after each epoch
        print(f"\n=== Testing MoE with Current Classifier (Epoch {epoch_idx + 1}) ===")
        test_moe_with_classifier(config, classifier, device, epoch_idx + 1, weight_dir)
    
    print(f"\nBest classifier accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # Load best classifier from new path
    moe_save_dir = os.path.join(weight_dir, 'moe_combined')
    best_path = os.path.join(moe_save_dir, 'classifier_best.pth')
    classifier.load_state_dict(torch.load(best_path, map_location=device))
    classifier.eval()
    
    return classifier

def test_moe_with_classifier(config, classifier, device, epoch_num, weight_dir, dataset_name=None):
    """Test MoE with current classifier on specific dataset."""
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
        
        # Save MoE model for specific dataset tests
        if dataset_name and epoch_num != "FINAL":
            moe_save_dir = os.path.join(weight_dir, f'moe_{dataset_name}')
            os.makedirs(moe_save_dir, exist_ok=True)
            
            # Save classifier
            classifier_path = os.path.join(moe_save_dir, f'classifier_epoch{epoch_num}.pth')
            torch.save(classifier.state_dict(), classifier_path)
            
            # Save MoE results
            results_path = os.path.join(moe_save_dir, f'results_epoch{epoch_num}.json')
            with open(results_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {}
                for k, v in results.items():
                    if isinstance(v, (np.integer, np.floating, np.ndarray)):
                        json_results[k] = float(v)
                    else:
                        json_results[k] = v
                json_results['routing_stats'] = stats
                json.dump(json_results, f, indent=2)
            
            print(f"Saved MoE model and results to {moe_save_dir}")
        
        return results
        
    except Exception as e:
        print(f"MoE test failed: {e}")
        return None

def main(config_path, epoch=None, dataset=None, weight_dir=None, phase=None):
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
    
    # If no phase specified, run both phases automatically
    if phase is None:
        print("No phase specified - running both Phase 1 (training) and Phase 2 (testing)")
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
            print("\nTraining completed. Use --phase 2 to test the model.")
            return
    
    # Phase 2: Test with best classifier
    if run_phase_2:
        print("\n" + "="*60)
        print("STARTING PHASE 2: TESTING WITH BEST CLASSIFIER")
        print("="*60)
        
        # Load classifier from organized path
        moe_save_dir = os.path.join(expert_weights_dir, 'moe_combined')
        classifier_path = os.path.join(moe_save_dir, 'classifier_best.pth')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier not found at {classifier_path}. Run Phase 1 first.")
        
        classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.eval()
        print(f"Loaded best classifier from: {classifier_path}")
        
        # Test on specified dataset and save final results
        results = test_moe_with_classifier(config, classifier, device, "FINAL", expert_weights_dir, dataset)
        
        # Save final MoE model for the tested dataset
        if dataset and results:
            moe_save_dir = os.path.join(expert_weights_dir, f'moe_{dataset}')
            os.makedirs(moe_save_dir, exist_ok=True)
            
            # Save final classifier
            final_classifier_path = os.path.join(moe_save_dir, 'classifier_final.pth')
            torch.save(classifier.state_dict(), final_classifier_path)
            
            # Save final results
            final_results_path = os.path.join(moe_save_dir, 'results_final.json')
            with open(final_results_path, 'w') as f:
                json_results = {}
                for k, v in results.items():
                    if isinstance(v, (np.integer, np.floating, np.ndarray)):
                        json_results[k] = float(v)
                    else:
                        json_results[k] = v
                json.dump(json_results, f, indent=2)
            
            print(f"Final MoE model saved to {moe_save_dir}")
        
        print("Phase 2 completed successfully!")
    
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
    args = parser.parse_args()

    main(args.config, args.epoch, args.dataset, args.weight_dir, args.phase)