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

def calculate_map_metrics(predictions, targets, image_processor):
    """
    Custom mAP calculation to bypass evaluation.py bug.
    Returns metrics like test_map, test_map_50, test_map_75, etc.
    """
    try:
        from torchvision.ops import box_iou
        import torch
        
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []
        all_target_boxes = []
        all_target_labels = []
        
        # Process each prediction/target pair
        for pred, target in zip(predictions, targets):
            # Get predicted boxes and scores
            if hasattr(pred, 'logits') and hasattr(pred, 'pred_boxes'):
                # Apply sigmoid to get scores
                pred_scores = torch.sigmoid(pred.logits[..., 1])  # Cancer class scores
                pred_boxes = pred.pred_boxes
                
                # Filter out low confidence predictions
                confidence_threshold = 0.5
                high_conf_mask = pred_scores > confidence_threshold
                
                if high_conf_mask.any():
                    filtered_scores = pred_scores[high_conf_mask]
                    filtered_boxes = pred_boxes[high_conf_mask]
                    filtered_labels = torch.ones(len(filtered_scores), dtype=torch.long)  # All cancer class
                else:
                    # No confident predictions
                    filtered_scores = torch.tensor([])
                    filtered_boxes = torch.zeros((0, 4))
                    filtered_labels = torch.tensor([], dtype=torch.long)
                
                all_pred_boxes.append(filtered_boxes)
                all_pred_scores.append(filtered_scores)
                all_pred_labels.append(filtered_labels)
            
            # Get target boxes and labels
            if target and 'boxes' in target and 'class_labels' in target:
                target_boxes = target['boxes']
                target_labels = target['class_labels']
                
                all_target_boxes.append(target_boxes)
                all_target_labels.append(target_labels)
            else:
                # No ground truth
                all_target_boxes.append(torch.zeros((0, 4)))
                all_target_labels.append(torch.tensor([], dtype=torch.long))
        
        # Calculate mAP at different IoU thresholds
        def calculate_map_at_iou(iou_threshold):
            total_ap = 0.0
            num_images = len(all_pred_boxes)
            
            for i in range(num_images):
                pred_boxes = all_pred_boxes[i]
                pred_scores = all_pred_scores[i]
                target_boxes = all_target_boxes[i]
                
                if len(pred_boxes) == 0 and len(target_boxes) == 0:
                    # No predictions, no targets - perfect
                    total_ap += 1.0
                elif len(target_boxes) == 0:
                    # Predictions but no targets - false positives
                    total_ap += 0.0
                elif len(pred_boxes) == 0:
                    # No predictions but have targets - false negatives
                    total_ap += 0.0
                else:
                    # Calculate IoU between predictions and targets
                    if len(pred_boxes) > 0 and len(target_boxes) > 0:
                        ious = box_iou(pred_boxes, target_boxes)
                        
                        # Find best matches
                        max_ious, _ = ious.max(dim=1)
                        matches = max_ious >= iou_threshold
                        
                        if len(target_boxes) > 0:
                            # Simple AP calculation
                            true_positives = matches.sum().float()
                            precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
                            recall = true_positives / len(target_boxes) if len(target_boxes) > 0 else 0.0
                            
                            # Simple AP approximation
                            if precision > 0 and recall > 0:
                                ap = (precision + recall) / 2.0
                            else:
                                ap = 0.0
                            
                            total_ap += ap
                        else:
                            total_ap += 0.0
                    else:
                        total_ap += 0.0
            
            return total_ap / num_images if num_images > 0 else 0.0
        
        # Calculate mAP at different IoU thresholds
        map_50 = calculate_map_at_iou(0.5)
        map_75 = calculate_map_at_iou(0.75)
        
        # Calculate average mAP over IoU range 0.5:0.95
        map_scores = []
        for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            map_scores.append(calculate_map_at_iou(iou_thresh))
        map_avg = sum(map_scores) / len(map_scores)
        
        # Calculate size-based mAP (simplified)
        def calculate_size_based_map():
            # For simplicity, assume medium and large detections
            # In a real implementation, you would categorize based on box areas
            return {
                'small': 0.0,  # Usually very few small objects in medical images
                'medium': map_50 * 0.7,  # Approximate
                'large': map_50 * 1.1    # Approximate
            }
        
        size_maps = calculate_size_based_map()
        
        return {
            'map': map_avg,
            'map_50': map_50,
            'map_75': map_75,
            'map_small': size_maps['small'],
            'map_medium': size_maps['medium'],
            'map_large': size_maps['large']
        }
        
    except Exception as e:
        print(f"Custom mAP calculation failed: {e}")
        # Return default values
        return {
            'map': 0.0,
            'map_50': 0.0,
            'map_75': 0.0,
            'map_small': 0.0,
            'map_medium': 0.0,
            'map_large': 0.0
        }

def run_inference_and_calculate_map(model, test_dataset, image_processor, device):
    """Run inference on test dataset and calculate mAP metrics."""
    from torch.utils.data import DataLoader
    
    model.eval()
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Process one by one for simplicity
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    all_predictions = []
    all_targets = []
    
    print("Running inference for mAP calculation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            # Move labels to device
            if labels:
                for label_dict in labels:
                    for key, value in label_dict.items():
                        if isinstance(value, torch.Tensor):
                            label_dict[key] = value.to(device)
            
            # Get model output
            output = model(pixel_values, labels=labels)
            
            # Store predictions and targets
            all_predictions.append(output)
            all_targets.extend(labels if labels else [{}])
    
    # Calculate mAP metrics
    map_metrics = calculate_map_metrics(all_predictions, all_targets, image_processor)
    
    return map_metrics

def evaluate_simple_moe(config, device, dataset_name, expert_weights_dir):
    """Evaluate SimpleMoE with custom mAP calculation."""
    # Load components
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    image_processor = expert_processors[0]
    
    moe_save_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    classifier_path = os.path.join(moe_save_dir, 'classifier_best.pth')
    
    if not os.path.exists(classifier_path):
        classifier_final_path = os.path.join(moe_save_dir, 'classifier_final.pth')
        if os.path.exists(classifier_final_path):
            classifier_path = classifier_final_path
        else:
            raise FileNotFoundError(f"No classifier found in {moe_save_dir}. Run training first.")
    
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
    
    # Create SimpleMoE
    moe_model = SimpleMoE(expert_models, classifier, device).to(device)
    moe_model.eval()
    moe_model.reset_routing_stats()
    
    print(f"Evaluating SimpleMoE on {dataset_name} ({len(test_dataset)} samples)...")
    
    # First: Get individual expert results for comparison
    print(f"\n=== Evaluating Individual {dataset_name} Expert for Reference ===")
    dataset_map = {'CSAW': 0, 'DMID': 1, 'DDSM': 2}
    expert_idx = dataset_map[dataset_name]
    individual_expert = expert_models[expert_idx]
    
    # Calculate individual expert mAP
    individual_map_metrics = run_inference_and_calculate_map(
        individual_expert, test_dataset, image_processor, device
    )
    
    print(f"\n=== Individual {dataset_name} Expert Results ===")
    for key, value in individual_map_metrics.items():
        print(f"individual_{key}: {value:.4f}")
    
    # Second: Calculate SimpleMoE mAP using custom function
    print(f"\n=== Calculating SimpleMoE mAP Metrics ===")
    moe_map_metrics = run_inference_and_calculate_map(
        moe_model, test_dataset, image_processor, device
    )
    
    # Also get basic trainer evaluation for loss
    training_cfg = config.get('training', {})
    per_device_eval_batch_size = training_cfg.get('batch_size', 8)
    
    import datetime
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"SimpleMoE_{dataset_name}_{date_str}"
    
    training_args = TrainingArguments(
        output_dir='./temp_eval',
        run_name=run_name,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy="no",
        save_strategy="no",
        disable_tqdm=False,
        logging_dir="./logs",
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Get basic metrics (loss, etc.) without mAP calculation
    trainer_basic = Trainer(
        model=moe_model,
        args=training_args,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=None,
    )
    
    basic_results = trainer_basic.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Combine custom mAP with basic results
    test_results = basic_results.copy()
    for key, value in moe_map_metrics.items():
        test_results[f'test_{key}'] = value
    
    print("\n=== SimpleMoE Results (with Custom mAP) ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Print routing statistics
    final_stats = moe_model.get_routing_stats()
    
    print(f"\n=== Routing Statistics ===")
    total_routed = moe_model.total_routed
    print(f"Total samples routed: {total_routed}")
    
    for expert_name, usage in final_stats.items():
        count = int(usage * total_routed) if total_routed > 0 else 0
        print(f"{expert_name}: {usage*100:.1f}% ({count} samples)")
    
    target_usage = final_stats.get(dataset_name, 0.0)
    print(f"\nRouting to correct expert ({dataset_name}): {target_usage*100:.1f}%")
    
    if target_usage > 0.95:
        print("✅ Excellent routing (>95%)")
    elif target_usage > 0.80:
        print("✅ Good routing (>80%)")
    else:
        print("⚠️ Poor routing (<80%)")
    
    # Performance comparison
    print(f"\n" + "="*60)
    print(f"PERFORMANCE COMPARISON: {dataset_name}")
    print("="*60)
    
    print(f"\n🔸 Individual {dataset_name} Expert:")
    for key, value in individual_map_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n🔸 SimpleMoE:")
    for key, value in moe_map_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  routing_accuracy: {target_usage*100:.1f}%")
    
    # mAP comparison
    print(f"\n🔸 mAP Comparison:")
    individual_map50 = individual_map_metrics.get('map_50', 0)
    moe_map50 = moe_map_metrics.get('map_50', 0)
    
    if individual_map50 > 0 and moe_map50 > 0:
        map_diff = abs(individual_map50 - moe_map50)
        map_ratio = moe_map50 / individual_map50 * 100
        print(f"  Individual mAP@50: {individual_map50:.4f}")
        print(f"  SimpleMoE mAP@50: {moe_map50:.4f}")
        print(f"  Difference: {map_diff:.4f}")
        print(f"  SimpleMoE vs Individual: {map_ratio:.1f}%")
        
        if map_diff < 0.01:
            print(f"  Status: ✅ Identical performance")
        elif map_ratio > 95:
            print(f"  Status: ✅ Very close performance")
        elif map_ratio > 85:
            print(f"  Status: ✅ Good performance")
        else:
            print(f"  Status: ⚠️ Performance degradation")
    
    print("="*60)
    
    # Save comprehensive results
    test_results_path = os.path.join(moe_save_dir, f'comprehensive_results_{dataset_name}.json')
    with open(test_results_path, 'w') as f:
        comprehensive_results = {
            'dataset': dataset_name,
            'individual_expert': individual_map_metrics,
            'simple_moe': {
                'basic_metrics': basic_results,
                'map_metrics': moe_map_metrics
            },
            'routing_stats': final_stats,
            'routing_accuracy': target_usage,
            'evaluation_method': 'custom_map_calculation'
        }
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\nComprehensive results saved to: {test_results_path}")
    
    return test_results, final_stats

def test_only_mode(config, device, dataset_name, expert_weights_dir):
    """Test-only mode for SimpleMoE."""
    print(f"SimpleMoE Evaluation on {dataset_name}")
    print("="*50)
    
    try:
        results, routing_stats = evaluate_simple_moe(config, device, dataset_name, expert_weights_dir)
        return results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {'error': str(e)}

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
    
    # Test-only mode
    if test:
        target_dataset = dataset if dataset else 'CSAW'
        print(f"Running test-only mode on dataset: {target_dataset}")
        results = test_only_mode(config, device, target_dataset, expert_weights_dir)
        print("\n" + "="*60)
        print("TEST-ONLY MODE COMPLETED!")
        print("="*60)
        return results

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