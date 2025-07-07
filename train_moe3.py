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
    FIXED: Đảm bảo routing statistics chính xác và output filtering
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
        
        # Statistics - FIXED: Reset properly for each evaluation
        self.routing_counts = torch.zeros(3, device=device)
        self.total_routed = 0

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass
    
    def gradient_checkpointing_disable(self):
        pass
    
    def forward(self, pixel_values, labels=None, return_routing=False):
        """
        Forward pass with dataset classification routing.
        FIXED: Đảm bảo routing chính xác và output giống như individual expert.
        """
        batch_size = pixel_values.shape[0]
        
        # 1. Classifier chọn expert - LUÔN no_grad và eval mode
        with torch.no_grad():
            # Đảm bảo classifier ở eval mode
            self.classifier.eval()
            dataset_logits = self.classifier(pixel_values)
            expert_choices = torch.argmax(dataset_logits, dim=1)
        
        # 2. FIXED: Chỉ cập nhật statistics khi eval và đảm bảo đúng batch size
        if not self.training:
            # Cập nhật routing counts chính xác
            for choice in expert_choices:
                self.routing_counts[choice] += 1
            self.total_routed += batch_size
        
        # 3. Group by expert - CHÍNH XÁC
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # 4. FIXED: Lấy expert outputs với device handling chính xác
        sample_to_output = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for expert_idx, sample_indices in expert_groups.items():
            # Lấy pixel values cho group này
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            # CRITICAL: Đảm bảo expert ở eval mode và xử lý device đúng
            with torch.no_grad():
                self.experts[expert_idx].eval()
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
            
            # Accumulate loss with proper weighting
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                weight = len(sample_indices) / batch_size
                total_loss = total_loss + expert_output.loss * weight
            
            # FIXED: Lưu output với index mapping chính xác
            for i, sample_idx in enumerate(sample_indices):
                sample_to_output[sample_idx] = {
                    'logits': expert_output.logits[i].clone(),  # Clone để tránh reference issues
                    'pred_boxes': expert_output.pred_boxes[i].clone(),
                }
        
        # 5. FIXED: Reconstruct batch với dimension checking
        batch_logits = []
        batch_pred_boxes = []
        
        # Lấy reference shape từ first expert output
        first_sample_idx = list(sample_to_output.keys())[0]
        ref_logits_shape = sample_to_output[first_sample_idx]['logits'].shape
        ref_boxes_shape = sample_to_output[first_sample_idx]['pred_boxes'].shape
        
        for i in range(batch_size):
            if i in sample_to_output:
                batch_logits.append(sample_to_output[i]['logits'])
                batch_pred_boxes.append(sample_to_output[i]['pred_boxes'])
            else:
                # KHÔNG NÊN XẢY RA - tạo dummy với đúng shape
                print(f"WARNING: No output for sample {i} - this should not happen!")
                dummy_logits = torch.zeros(ref_logits_shape, device=pixel_values.device)
                dummy_boxes = torch.zeros(ref_boxes_shape, device=pixel_values.device)
                batch_logits.append(dummy_logits)
                batch_pred_boxes.append(dummy_boxes)
        
        # Stack thành batch tensors với error checking
        try:
            batch_logits = torch.stack(batch_logits, dim=0)
            batch_pred_boxes = torch.stack(batch_pred_boxes, dim=0)
        except Exception as e:
            print(f"Error stacking outputs: {e}")
            print(f"Batch size: {batch_size}, Outputs: {len(batch_logits)}")
            raise e
        
        # CRITICAL: Đảm bảo pred_boxes có đúng 4 dimensions
        if batch_pred_boxes.shape[-1] != 4:
            print(f"Warning: pred_boxes has {batch_pred_boxes.shape[-1]} dims, truncating to 4")
            batch_pred_boxes = batch_pred_boxes[..., :4]
        batch_pred_boxes = batch_pred_boxes.contiguous()
        
        # 6. Tạo output với format chính xác
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
        """Get routing statistics - FIXED: Đảm bảo tính toán chính xác."""
        if self.total_routed == 0:
            return {name: 0.0 for name in self.expert_names}
        
        stats = {}
        usage = (self.routing_counts / self.total_routed).cpu().numpy()
        for i, name in enumerate(self.expert_names):
            stats[name] = float(usage[i])
        return stats
    
    def reset_routing_stats(self):
        """Reset routing statistics - FIXED: Đảm bảo reset hoàn toàn."""
        self.routing_counts = torch.zeros(3, device=self.device)
        self.total_routed = 0
        print(f"Routing statistics reset - total_routed: {self.total_routed}")

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

def debug_moe_vs_individual(config, device, dataset_name, expert_weights_dir, num_samples=5):
    """Debug function to compare SimpleMoE vs Individual Expert outputs on same samples."""
    print(f"\n=== DEBUG: SimpleMoE vs Individual Expert on {dataset_name} ===")
    
    # Load expert models
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    image_processor = expert_processors[0]
    
    # Load classifier
    moe_save_dir = os.path.join(expert_weights_dir, 'moe_MOMO')
    classifier_path = os.path.join(moe_save_dir, 'classifier_best.pth')
    
    if not os.path.exists(classifier_path):
        classifier_final_path = os.path.join(moe_save_dir, 'classifier_final.pth')
        if os.path.exists(classifier_final_path):
            classifier_path = classifier_final_path
        else:
            raise FileNotFoundError(f"No classifier found in {moe_save_dir}")
    
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
    
    # Get the correct expert for this dataset
    dataset_map = {'CSAW': 0, 'DMID': 1, 'DDSM': 2}
    expert_idx = dataset_map[dataset_name]
    individual_expert = expert_models[expert_idx]
    
    # Create SimpleMoE
    moe_model = SimpleMoE(expert_models, classifier, device).to(device)
    moe_model.eval()
    moe_model.reset_routing_stats()
    
    # Helper function to move labels to device
    def move_labels_to_device(labels_list, target_device):
        if labels_list is None:
            return None
        moved_labels = []
        for label_dict in labels_list:
            moved_dict = {}
            for key, value in label_dict.items():
                if isinstance(value, torch.Tensor):
                    moved_dict[key] = value.to(target_device)
                else:
                    moved_dict[key] = value
            moved_labels.append(moved_dict)
        return moved_labels
    
    print(f"Testing on {num_samples} samples from {dataset_name}...")
    
    # Test on first few samples
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    sample_count = 0
    total_differences = {
        'logits_diff': 0.0,
        'boxes_diff': 0.0,
        'loss_diff': 0.0,
        'routing_correct': 0
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if sample_count >= num_samples:
                break
                
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            labels = move_labels_to_device(labels, device)
            
            print(f"\n--- Sample {sample_count + 1} ---")
            print(f"Pixel values shape: {pixel_values.shape}")
            
            # 1. Individual Expert Forward
            individual_expert.eval()
            individual_output = individual_expert(pixel_values, labels=labels)
            
            # 2. SimpleMoE Forward 
            moe_model.eval()
            moe_output, routing_info = moe_model(pixel_values, labels=labels, return_routing=True)
            
            # 3. Analyze routing
            expert_choice = routing_info['choices'][0].item()
            routing_correct = (expert_choice == expert_idx)
            total_differences['routing_correct'] += int(routing_correct)
            
            print(f"Expected expert: {expert_idx} ({dataset_name})")
            print(f"Routed to expert: {expert_choice} ({moe_model.expert_names[expert_choice]})")
            print(f"Routing correct: {routing_correct}")
            
            # 4. Compare outputs
            if routing_correct:
                # Compare logits
                logits_diff = torch.abs(individual_output.logits - moe_output.logits).mean().item()
                total_differences['logits_diff'] += logits_diff
                print(f"Logits difference (mean abs): {logits_diff:.6f}")
                
                # Compare pred_boxes
                boxes_diff = torch.abs(individual_output.pred_boxes - moe_output.pred_boxes).mean().item()
                total_differences['boxes_diff'] += boxes_diff
                print(f"Pred_boxes difference (mean abs): {boxes_diff:.6f}")
                
                # Compare loss
                if individual_output.loss is not None and moe_output.loss is not None:
                    loss_diff = abs(individual_output.loss.item() - moe_output.loss.item())
                    total_differences['loss_diff'] += loss_diff
                    print(f"Individual loss: {individual_output.loss.item():.6f}")
                    print(f"MoE loss: {moe_output.loss.item():.6f}")
                    print(f"Loss difference: {loss_diff:.6f}")
                
                # Check tensor properties
                print(f"Individual logits shape: {individual_output.logits.shape}")
                print(f"MoE logits shape: {moe_output.logits.shape}")
                print(f"Individual boxes shape: {individual_output.pred_boxes.shape}")
                print(f"MoE boxes shape: {moe_output.pred_boxes.shape}")
                
                # Check if tensors are identical (should be for correct routing)
                logits_identical = torch.allclose(individual_output.logits, moe_output.logits, atol=1e-6)
                boxes_identical = torch.allclose(individual_output.pred_boxes, moe_output.pred_boxes, atol=1e-6)
                print(f"Logits identical (atol=1e-6): {logits_identical}")
                print(f"Boxes identical (atol=1e-6): {boxes_identical}")
                
                if not logits_identical or not boxes_identical:
                    print("⚠️  WARNING: Outputs should be identical when routing is correct!")
            else:
                print("❌ Routing incorrect - cannot compare outputs meaningfully")
            
            sample_count += 1
    
    # Print summary
    print(f"\n=== DEBUG Summary ({num_samples} samples) ===")
    print(f"Routing accuracy: {total_differences['routing_correct']}/{num_samples} = {total_differences['routing_correct']/num_samples*100:.1f}%")
    
    if total_differences['routing_correct'] > 0:
        correct_samples = total_differences['routing_correct']
        print(f"Average differences (for correctly routed samples):")
        print(f"  Logits difference: {total_differences['logits_diff']/correct_samples:.8f}")
        print(f"  Boxes difference: {total_differences['boxes_diff']/correct_samples:.8f}")
        print(f"  Loss difference: {total_differences['loss_diff']/correct_samples:.8f}")
        
        if total_differences['logits_diff']/correct_samples < 1e-6 and total_differences['boxes_diff']/correct_samples < 1e-6:
            print("✅ Outputs are virtually identical - SimpleMoE working correctly!")
        else:
            print("⚠️  Outputs differ - there may be an issue with SimpleMoE implementation")
    else:
        print("❌ No correctly routed samples - classifier needs debugging")
    
    # Final routing stats
    final_stats = moe_model.get_routing_stats()
    print(f"\nFinal routing distribution:")
    for expert_name, usage in final_stats.items():
        print(f"  {expert_name}: {usage*100:.1f}%")

def simple_moe_evaluation(config, device, dataset_name, expert_weights_dir):
    """Complete evaluation of SimpleMoE using trainer.evaluate exactly like test.py."""
    # FIRST: Run debug to check for issues
    print("\n" + "="*60)
    print("DEBUGGING SimpleMoE vs Individual Expert")
    print("="*60)
    try:
        debug_moe_vs_individual(config, device, dataset_name, expert_weights_dir, num_samples=10)
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
    
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
    
    # FIXED: Reset routing statistics BEFORE evaluation
    moe_model.reset_routing_stats()
    print(f"SimpleMoE reset - starting fresh evaluation on {len(test_dataset)} samples")
    
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
    
    # Try evaluation with compute_metrics first, fallback if it fails
    test_results = {}
    
    try:
        # Use TrainingArguments for evaluation with compute_metrics
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
            dataloader_num_workers=dataloader_num_workers,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=remove_unused_columns,
        )
        
        # Use evaluation function from evaluation.py exactly like test.py
        eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
        
        # Create trainer exactly like test.py
        trainer = Trainer(
            model=moe_model,
            args=training_args,
            processing_class=image_processor,
            data_collator=collate_fn,
            compute_metrics=eval_compute_metrics_fn,
        )
        
        # FIXED: Reset routing stats RIGHT BEFORE evaluation
        moe_model.reset_routing_stats()
        
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
        
    except Exception as e:
        print(f"mAP evaluation failed due to evaluation.py bug: {e}")
        print("Falling back to basic evaluation without mAP metrics...")
        
        # FIXED: Reset routing stats before fallback evaluation too
        moe_model.reset_routing_stats()
        
        # Fallback: Basic evaluation without compute_metrics to avoid the bug
        training_args_basic = TrainingArguments(
            output_dir='./temp_eval_basic',
            run_name=f"{run_name}_basic",
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_strategy="no",
            save_strategy="no",
            disable_tqdm=False,
            logging_dir="./logs",
            report_to=[],
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=dataloader_num_workers,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=remove_unused_columns,
        )
        
        # Create trainer without compute_metrics to avoid the bug
        trainer_basic = Trainer(
            model=moe_model,
            args=training_args_basic,
            processing_class=image_processor,
            data_collator=collate_fn,
            compute_metrics=None,  # No compute_metrics to avoid the bug
        )
        
        # Run basic evaluation
        print(f'Test loader (basic): {len(test_dataset)} samples')
        test_results = trainer_basic.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
        
        # Print basic results
        print("\n=== Test Results (Basic) ===")
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        print("\nNote: mAP metrics unavailable due to evaluation.py bug")
        print("Loss and basic metrics shown above")
    
    # Get routing statistics - SHOULD MATCH TEST DATASET SIZE
    final_stats = moe_model.get_routing_stats()
    
    # VERIFICATION: Check if routing stats match expected test dataset size
    expected_samples = len(test_dataset)
    actual_samples = moe_model.total_routed
    
    print(f"\n=== Routing Verification ===")
    print(f"Expected samples (test dataset): {expected_samples}")
    print(f"Actual samples routed: {actual_samples}")
    
    if actual_samples != expected_samples:
        print(f"⚠️  WARNING: Routing count mismatch! Expected {expected_samples}, got {actual_samples}")
        print("This indicates a problem with routing statistics calculation")
    else:
        print("✓ Routing count matches test dataset size")
    
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
    
    # Save results with verification info
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
        json_results['expected_samples'] = expected_samples
        json_results['routing_accuracy'] = target_usage
        json_results['routing_count_correct'] = (actual_samples == expected_samples)
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