import argparse
import os

# Suppress TensorFlow and CUDA warnings (same as train.py)
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    """Load configuration from YAML file (same as train.py)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class ImageRouterMoE(nn.Module):
    """
    Router-based MoE that learns to route images to the best expert.
    Uses a minimal CNN to classify which expert should handle each image.
    """
    def __init__(self, expert_models, device):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)  # [CSAW, DMID, DDSM]
        self.device = device
        
        # Minimal router network: very simple CNN + classifier
        self.router = nn.Sequential(
            # Simple feature extractor - much smaller
            nn.Conv2d(3, 16, 7, stride=4, padding=3),  # 640->160
            nn.ReLU(),
            nn.MaxPool2d(4),  # 160->40
            nn.Conv2d(16, 32, 5, stride=2, padding=2),  # 40->20
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling -> 1x1
            nn.Flatten(),
            
            # Simple classifier (3 classes: CSAW=0, DMID=1, DDSM=2)
            nn.Linear(32, 3),  # Direct to 3 experts, no hidden layers
        )
        
        # Freeze expert models (similar to train.py approach)
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        print(f"Router MoE initialized with {len(self.experts)} frozen experts")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters: {trainable_params / 1e3:.2f}K (router only)')
    
    def forward(self, pixel_values, labels=None, return_routing=False):
        """
        Forward pass: route each image to best expert and return expert's output.
        NO COMBINATION - just return the chosen expert's output directly.
        """
        batch_size = pixel_values.shape[0]
        
        # Get routing probabilities
        routing_logits = self.router(pixel_values)  # Shape: (batch_size, 3)
        routing_probs = F.softmax(routing_logits, dim=1)
        expert_choices = torch.argmax(routing_probs, dim=1)  # Shape: (batch_size,)
        
        # For training: compute routing loss to learn better routing
        routing_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if labels is not None:
            # Get outputs from all experts to compare losses (for routing learning)
            expert_losses = []
            for expert_idx in range(len(self.experts)):
                with torch.no_grad():
                    expert_output = self.experts[expert_idx](pixel_values, labels=labels)
                    if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                        expert_losses.append(expert_output.loss)
            
            if len(expert_losses) > 0:
                # Find best expert (lowest loss) for routing supervision
                expert_losses_tensor = torch.stack(expert_losses)
                best_expert_indices = torch.argmin(expert_losses_tensor.unsqueeze(0).expand(batch_size, -1), dim=1)
                
                # Routing loss: encourage router to choose the best expert
                routing_loss = F.cross_entropy(routing_logits, best_expert_indices)
        
        # For batch processing, we need to group by expert choice
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # Get the first expert output to determine correct tensor shapes and dtypes
        first_expert_idx = list(expert_groups.keys())[0]
        first_sample_indices = expert_groups[first_expert_idx]
        first_expert_pixel_values = pixel_values[first_sample_indices]
        first_expert_labels = None
        if labels is not None:
            first_expert_labels = [labels[i] for i in first_sample_indices]
        
        with torch.no_grad():
            reference_output = self.experts[first_expert_idx](first_expert_pixel_values, labels=first_expert_labels)
        
        # CRITICAL: Fix pred_boxes dimensions immediately from reference output
        if reference_output.pred_boxes.shape[-1] != 4:
            print(f"WARNING: Reference expert pred_boxes has {reference_output.pred_boxes.shape[-1]} dims, fixing to 4")
            reference_output = type(reference_output)(
                loss=reference_output.loss,
                logits=reference_output.logits,
                pred_boxes=reference_output.pred_boxes[..., :4].contiguous(),
                last_hidden_state=reference_output.last_hidden_state if hasattr(reference_output, 'last_hidden_state') else None
            )
        
        # Initialize output tensors with correct dtypes and shapes from reference
        num_queries = reference_output.logits.shape[1]
        num_classes = reference_output.logits.shape[2]
        batch_logits = torch.zeros(batch_size, num_queries, num_classes, 
                                 device=pixel_values.device, dtype=reference_output.logits.dtype)
        batch_pred_boxes = torch.zeros(batch_size, num_queries, 4, 
                                     device=pixel_values.device, dtype=reference_output.pred_boxes.dtype)
        batch_loss = torch.tensor(0.0, device=pixel_values.device, requires_grad=True)
        batch_last_hidden_state = None
        
        # Use the reference output for the first group
        batch_logits[first_sample_indices] = reference_output.logits
        batch_pred_boxes[first_sample_indices] = reference_output.pred_boxes  # Already fixed to 4D above
        
        if hasattr(reference_output, 'loss') and reference_output.loss is not None:
            batch_loss = batch_loss + reference_output.loss * len(first_sample_indices) / batch_size
        
        if hasattr(reference_output, 'last_hidden_state') and reference_output.last_hidden_state is not None:
            batch_last_hidden_state = torch.zeros(batch_size, reference_output.last_hidden_state.shape[1], 
                                                 reference_output.last_hidden_state.shape[2], 
                                                 device=pixel_values.device, dtype=reference_output.last_hidden_state.dtype)
            batch_last_hidden_state[first_sample_indices] = reference_output.last_hidden_state
        
        # Process remaining expert groups (skip the first one we already processed)
        remaining_groups = {k: v for k, v in expert_groups.items() if k != first_expert_idx}
        for expert_idx, sample_indices in remaining_groups.items():
            # Get inputs for this expert
            expert_pixel_values = pixel_values[sample_indices]
            
            # Fix: Handle labels properly - labels is a list, so we need to extract individual items
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            # Get expert output
            with torch.no_grad():
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
            
            # CRITICAL: Fix pred_boxes dimensions before using the output
            if expert_output.pred_boxes.shape[-1] != 4:
                print(f"WARNING: Expert {expert_idx} pred_boxes has {expert_output.pred_boxes.shape[-1]} dims, fixing to 4")
                expert_output = type(expert_output)(
                    loss=expert_output.loss,
                    logits=expert_output.logits,
                    pred_boxes=expert_output.pred_boxes[..., :4].contiguous(),
                    last_hidden_state=expert_output.last_hidden_state if hasattr(expert_output, 'last_hidden_state') else None
                )
            
            # Place expert outputs back into batch positions
            batch_logits[sample_indices] = expert_output.logits
            batch_pred_boxes[sample_indices] = expert_output.pred_boxes  # Now guaranteed to be 4D
            
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                batch_loss = batch_loss + expert_output.loss * len(sample_indices) / batch_size
            
            if batch_last_hidden_state is not None and hasattr(expert_output, 'last_hidden_state'):
                batch_last_hidden_state[sample_indices] = expert_output.last_hidden_state
        
        # Add routing loss to total loss (if training)
        if labels is not None and routing_loss.item() > 0:
            batch_loss = batch_loss + 0.1 * routing_loss
        
        # Final safety check - ensure ALL outputs have exactly 4 dimensions
        assert batch_pred_boxes.shape[-1] == 4, f"FINAL CHECK FAILED: pred_boxes has {batch_pred_boxes.shape[-1]} dims, expected 4"
        
        # Additional safety: force contiguous and ensure exact 4D for all outputs
        batch_pred_boxes = batch_pred_boxes[..., :4].contiguous()
        
        # Create output exactly like a single YOLOS model
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        combined_output = YolosObjectDetectionOutput(
            loss=batch_loss,
            logits=batch_logits,
            pred_boxes=batch_pred_boxes,  # Now guaranteed to be exactly 4D
            last_hidden_state=batch_last_hidden_state
        )
        
        if return_routing:
            return combined_output, routing_probs, expert_choices
        else:
            return combined_output
    
    def _combine_batch_outputs(self, batch_outputs):
        """This method is no longer needed - we process by expert groups instead."""
        pass

def load_expert_models(weight_dir, device):
    """Load the 3 expert models (similar to train.py model loading)."""
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

def create_routing_dataset(config, image_processor, split='train', dataset_name='CSAW', epoch=None):
    """
    Create dataset with routing labels based on the dataset name.
    Uses standard dataset from argument/config - NO additional routing labels needed.
    The routing will be learned from object detection loss, like YOLOS.
    """
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    # Create single dataset based on argument/config (exactly like train.py)
    dataset = BreastCancerDataset(
        split=split,
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    
    print(f"{dataset_name} {split}: {len(dataset)} samples")
    
    # Return the dataset directly (no routing labels needed)
    # The router will learn from object detection performance
    return dataset

def main(config_path, epoch=None, dataset=None):
    """Main training function following train.py structure exactly."""
    config = load_config(config_path)
    
    # Use dataset from argument or config (same as train.py)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)
    
    # Add wandb folder support (same as train.py)
    wandb_dir = None
    if 'wandb' in config and 'wandb_dir' in config['wandb']:
        wandb_dir = config['wandb']['wandb_dir']
        print(f"Wandb directory: {wandb_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get weight_dir from config (instead of argument) - same pattern as train.py
    weight_dir = config.get('moe', {}).get('expert_weights_dir', '/content/Weights')
    if not os.path.exists(weight_dir):
        raise ValueError(f"Expert weights directory not found: {weight_dir}. Please add 'moe.expert_weights_dir' to your config.")
    
    print("Loading expert models...")
    expert_models, expert_processors = load_expert_models(weight_dir, device)
    
    # Use first processor for consistency (same as train.py approach)
    image_processor = expert_processors[0]
    
    # Create standard datasets (exactly like train.py dataset creation)
    train_dataset = create_routing_dataset(config, image_processor, 'train', DATASET_NAME, epoch)
    val_dataset = create_routing_dataset(config, image_processor, 'val', DATASET_NAME, epoch)
    
    # Create data loaders (same as train.py)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f'Train loader: {len(train_dataset)} samples')
    print(f'Val loader: {len(val_dataset)} samples')
    print(f"Train loader batches: {len(train_loader)}")
    
    # Create Router MoE model
    model = ImageRouterMoE(expert_models, device).to(device)
    
    # Load training arguments from config (exactly like train.py)
    training_cfg = config.get('training', {})
    output_dir = training_cfg.get('output_dir', '/tmp')
    num_train_epochs = epoch if epoch is not None else training_cfg.get('epochs', 20)
    per_device_train_batch_size = training_cfg.get('batch_size', 8)
    per_device_eval_batch_size = training_cfg.get('batch_size', 8)
    learning_rate = training_cfg.get('learning_rate', 5e-5)
    weight_decay = training_cfg.get('weight_decay', 1e-4)
    warmup_ratio = training_cfg.get('warmup_ratio', 0.05)
    lr_scheduler_type = training_cfg.get('lr_scheduler_type', 'cosine_with_restarts')
    lr_scheduler_kwargs = training_cfg.get('lr_scheduler_kwargs', dict(num_cycles=1))
    eval_do_concat_batches = training_cfg.get('eval_do_concat_batches', False)
    evaluation_strategy = training_cfg.get('evaluation_strategy', 'epoch')
    save_strategy = training_cfg.get('save_strategy', 'epoch')
    save_total_limit = training_cfg.get('save_total_limit', 1)
    logging_strategy = training_cfg.get('logging_strategy', 'epoch')
    load_best_model_at_end = training_cfg.get('load_best_model_at_end', True)
    metric_for_best_model = training_cfg.get('metric_for_best_model', 'eval_map_50')
    greater_is_better = training_cfg.get('greater_is_better', True)
    dataloader_num_workers = training_cfg.get('num_workers', 2)
    gradient_accumulation_steps = training_cfg.get('gradient_accumulation_steps', 2)
    remove_unused_columns = training_cfg.get('remove_unused_columns', False)
    
    print(f"Using epoch (num_train_epochs): {num_train_epochs}")
    
    # Set unique run_name (same as train.py)
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"RouterMoE_{DATASET_NAME}_{date_str}"
    
    # Training arguments (exactly like train.py)
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        eval_do_concat_batches=eval_do_concat_batches,
        disable_tqdm=False,
        logging_dir=wandb_dir if wandb_dir else "./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        logging_strategy="epoch",
        report_to="all",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=remove_unused_columns,
    )
    
    # Evaluation metrics (same as train.py)
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    # Training (exactly like train.py)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )
    
    print("\n=== Training now ===")
    trainer.train()
    
    # Save model (same as train.py)
    date_str = datetime.datetime.now().strftime("%d%m%y")
    save_path = f'../router_moe_{DATASET_NAME}_{date_str}'
    trainer.save_model(save_path)
    print(f"Model saved to: {save_path}")
    
    # Evaluate on test dataset (same as train.py structure)
    print("\n=== Evaluating on test dataset ===")
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    print(f'Test dataset: {len(test_dataset)} samples')
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Print test results (exactly like train.py and test.py)
    print("\n=== Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to use (overrides config)')
    args = parser.parse_args()

    main(args.config, args.epoch, args.dataset)