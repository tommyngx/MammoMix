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
    Đơn giản: classifier chọn expert → lấy output expert → đưa ra kết quả
    Không phức tạp, freeze expert, evaluate như test.py
    """
    def __init__(self, expert_models, dataset_classifier, device):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)
        self.classifier = dataset_classifier
        self.device = device
        self.expert_names = ['CSAW', 'DMID', 'DDSM']
        
        # Freeze ALL experts - không train gì cả
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
            expert.eval()  # Luôn eval mode
        
        # Freeze classifier luôn - chỉ dùng để route
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.eval()  # Luôn eval mode
        
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
        
        # 1. Classifier chọn expert - LUÔN no_grad
        with torch.no_grad():
            dataset_logits = self.classifier(pixel_values)
            expert_choices = torch.argmax(dataset_logits, dim=1)
        
        # Statistics
        if not self.training:
            for choice in expert_choices:
                self.routing_counts[choice] += 1
            self.total_routed += batch_size
        
        # 2. Group by expert
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # 3. Get expert outputs
        sample_to_output = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for expert_idx, sample_indices in expert_groups.items():
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            with torch.no_grad():
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
            
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                weight = len(sample_indices) / batch_size
                total_loss = total_loss + expert_output.loss * weight
            
            for i, sample_idx in enumerate(sample_indices):
                sample_to_output[sample_idx] = {
                    'logits': expert_output.logits[i],
                    'pred_boxes': expert_output.pred_boxes[i],
                }
        
        # 4. Reconstruct batch
        batch_logits = []
        batch_pred_boxes = []
        
        for i in range(batch_size):
            if i in sample_to_output:
                batch_logits.append(sample_to_output[i]['logits'])
                batch_pred_boxes.append(sample_to_output[i]['pred_boxes'])
            else:
                dummy_logits = torch.zeros((100, 2), device=pixel_values.device)
                dummy_boxes = torch.zeros((100, 4), device=pixel_values.device)
                batch_logits.append(dummy_logits)
                batch_pred_boxes.append(dummy_boxes)
        
        batch_logits = torch.stack(batch_logits, dim=0)
        batch_pred_boxes = torch.stack(batch_pred_boxes, dim=0)
        batch_pred_boxes = batch_pred_boxes[..., :4].contiguous()
        
        # 5. Create output
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
        self.routing_counts.zero_()
        self.total_routed = 0

def load_expert_models(weight_dir, device):
    """Load expert models silently."""
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

def simple_moe_evaluation(config, device, dataset_name, expert_weights_dir):
    """Complete evaluation of SimpleMoE using trainer.evaluate exactly like test.py."""
    # Load components silently
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
    
    # Create test dataset exactly like test.py
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
    
    # Setup training args exactly like test.py but for evaluation only
    training_cfg = config.get('training', {})
    
    # Extract all training arguments from config like test.py
    per_device_eval_batch_size = training_cfg.get('batch_size', 8)
    dataloader_num_workers = training_cfg.get('num_workers', 2)
    gradient_accumulation_steps = training_cfg.get('gradient_accumulation_steps', 2)
    remove_unused_columns = training_cfg.get('remove_unused_columns', False)
    
    # Create run_name exactly like test.py
    import datetime
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"SimpleMoE_{dataset_name}_{date_str}"
    
    # Use TrainingArguments for evaluation only - fix the eval_strategy issue
    training_args = TrainingArguments(
        output_dir='./temp_eval',
        run_name=run_name,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_strategy="no",  # Set to "no" since we're only doing evaluation
        save_strategy="no",  # Set to "no" since we're only doing evaluation
        disable_tqdm=False,
        logging_dir="./logs",
        report_to=[],  # Disable wandb and all external loggers
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=remove_unused_columns,
    )
    
    # Use evaluation function from evaluation.py exactly like test.py
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    # Create trainer exactly like test.py - without eval_dataset since we're only evaluating
    trainer = Trainer(
        model=moe_model,
        args=training_args,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    # Run evaluation exactly like test.py
    print(f'Test loader: {len(test_dataset)} samples')
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Print results exactly like test.py
    print("\n=== Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Get routing statistics
    final_stats = moe_model.get_routing_stats()
    
    # Print routing statistics
    print(f"\n=== Routing Statistics ===")
    total_routed = moe_model.total_routed
    print(f"Total samples routed: {total_routed}")
    
    for expert_name, usage in final_stats.items():
        count = int(usage * total_routed) if total_routed > 0 else 0
        print(f"{expert_name}: {usage*100:.1f}% ({count} samples)")
    
    # Check routing accuracy for the target dataset
    target_usage = final_stats.get(dataset_name, 0.0)
    print(f"\nRouting to correct expert ({dataset_name}): {target_usage*100:.1f}%")
    
    if target_usage > 0.95:
        print("✓ Excellent routing (>95%)")
    elif target_usage > 0.80:
        print("✓ Good routing (>80%)")
    elif target_usage > 0.60:
        print("⚠ Fair routing (>60%)")
    else:
        print("✗ Poor routing (<60%)")
    
    # Save results
    test_results_path = os.path.join(moe_save_dir, f'moe_results_{dataset_name}.json')
    with open(test_results_path, 'w') as f:
        json_results = {}
        for k, v in test_results.items():
            if isinstance(v, (np.integer, np.floating, np.ndarray)):
                json_results[k] = float(v)
            else:
                json_results[k] = v
        json_results['routing_stats'] = final_stats
        json_results['dataset'] = dataset_name
        json_results['total_routed'] = total_routed
        json_results['routing_accuracy'] = target_usage
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {test_results_path}")
    
    return test_results, final_stats

def test_only_mode(config, device, dataset_name, expert_weights_dir):
    """Test-only mode: Clean evaluation with detailed routing stats."""
    print(f"SimpleMoE Evaluation on {dataset_name}")
    print("="*50)
    
    try:
        results, routing_stats = simple_moe_evaluation(config, device, dataset_name, expert_weights_dir)
        return results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

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