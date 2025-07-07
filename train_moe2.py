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
    SIMPLIFIED: Much simpler architecture for faster training
    """
    def __init__(self, expert_models, device):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)  # [CSAW, DMID, DDSM]
        self.device = device
        
        # ULTRA SIMPLE router network: just 2 layers
        self.router = nn.Sequential(
            # Very simple feature extractor
            nn.AdaptiveAvgPool2d(1),  # Global average pooling: 640x640x3 -> 1x1x3
            nn.Flatten(),             # 3 features
            
            # Direct classification - no hidden layers
            nn.Linear(3, 3),          # RGB -> 3 experts directly
        )
        
        # Simple initialization
        nn.init.xavier_uniform_(self.router[2].weight)
        nn.init.zeros_(self.router[2].bias)
        
        # Freeze expert models (similar to train.py approach)
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        print(f"Router MoE initialized with {len(self.experts)} frozen experts")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters: {trainable_params:.0f} (router only - ULTRA SIMPLE)')
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for compatibility with Transformers Trainer."""
        pass
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for compatibility with Transformers Trainer."""
        pass
    
    def forward(self, pixel_values, labels=None, return_routing=False):
        """
        Forward pass: route each image to best expert and return expert's output.
        SIMPLIFIED: Much simpler routing logic
        """
        batch_size = pixel_values.shape[0]
        
        # Get routing probabilities - SUPER SIMPLE
        routing_logits = self.router(pixel_values)  # Shape: (batch_size, 3)
        expert_choices = torch.argmax(routing_logits, dim=1)  # Shape: (batch_size,)
        
        # SIMPLIFIED routing loss computation - only during training
        routing_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if labels is not None:
            # Simple approach: just encourage uniform distribution
            routing_probs = F.softmax(routing_logits, dim=1)
            expert_usage = routing_probs.mean(dim=0)  # Average usage per expert
            target_usage = torch.ones_like(expert_usage) / len(self.experts)  # Uniform
            routing_loss = F.mse_loss(expert_usage, target_usage)
        
        # For batch processing, group by expert choice
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # Get the first expert output to determine tensor shapes
        first_expert_idx = list(expert_groups.keys())[0]
        first_sample_indices = expert_groups[first_expert_idx]
        first_expert_pixel_values = pixel_values[first_sample_indices]
        first_expert_labels = None
        if labels is not None:
            first_expert_labels = [labels[i] for i in first_sample_indices]
        
        with torch.no_grad():
            reference_output = self.experts[first_expert_idx](first_expert_pixel_values, labels=first_expert_labels)
            reference_output = self._fix_expert_output_dimensions(reference_output)
        
        # Initialize output tensors
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
        batch_pred_boxes[first_sample_indices] = reference_output.pred_boxes
        
        if hasattr(reference_output, 'loss') and reference_output.loss is not None:
            batch_loss = batch_loss + reference_output.loss * len(first_sample_indices) / batch_size
        
        if hasattr(reference_output, 'last_hidden_state') and reference_output.last_hidden_state is not None:
            batch_last_hidden_state = torch.zeros(batch_size, reference_output.last_hidden_state.shape[1], 
                                                 reference_output.last_hidden_state.shape[2], 
                                                 device=pixel_values.device, dtype=reference_output.last_hidden_state.dtype)
            batch_last_hidden_state[first_sample_indices] = reference_output.last_hidden_state
        
        # Process remaining expert groups
        remaining_groups = {k: v for k, v in expert_groups.items() if k != first_expert_idx}
        for expert_idx, sample_indices in remaining_groups.items():
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            with torch.no_grad():
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
                expert_output = self._fix_expert_output_dimensions(expert_output)
            
            batch_logits[sample_indices] = expert_output.logits
            batch_pred_boxes[sample_indices] = expert_output.pred_boxes
            
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                batch_loss = batch_loss + expert_output.loss * len(sample_indices) / batch_size
            
            if batch_last_hidden_state is not None and hasattr(expert_output, 'last_hidden_state'):
                batch_last_hidden_state[sample_indices] = expert_output.last_hidden_state
        
        # Add VERY SMALL routing loss - just for diversity
        if labels is not None and routing_loss.item() > 0:
            batch_loss = batch_loss + 0.01 * routing_loss  # Very small weight
        
        # Final safety check
        assert batch_pred_boxes.shape[-1] == 4, f"pred_boxes has {batch_pred_boxes.shape[-1]} dims, expected 4"
        batch_pred_boxes = batch_pred_boxes[..., :4].contiguous()
        
        # Create output
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        combined_output = YolosObjectDetectionOutput(
            loss=batch_loss,
            logits=batch_logits,
            pred_boxes=batch_pred_boxes,
            last_hidden_state=batch_last_hidden_state
        )
        
        if return_routing:
            return combined_output, F.softmax(routing_logits, dim=1), expert_choices
        else:
            return combined_output
    
    def _fix_expert_output_dimensions(self, expert_output):
        """Fix expert output dimensions to ensure pred_boxes has exactly 4 dimensions."""
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

def test_saved_moe_model(config_path, model_path, dataset=None, epoch=None, weight_dir=None):
    """Test a saved Router MoE model."""
    config = load_config(config_path)
    
    # Use dataset from argument or config (same as train.py)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get weight_dir from argument or config - prioritize argument
    if weight_dir is not None:
        expert_weights_dir = weight_dir
        print(f"Using expert weights from argument: {expert_weights_dir}")
    else:
        expert_weights_dir = config.get('moe', {}).get('expert_weights_dir', '/content/Weights')
        print(f"Using expert weights from config: {expert_weights_dir}")
    
    if not os.path.exists(expert_weights_dir):
        raise ValueError(f"Expert weights directory not found: {expert_weights_dir}")
    
    print("Loading expert models...")
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    
    # Use first processor for consistency
    image_processor = expert_processors[0]
    
    # Create Router MoE model
    model = ImageRouterMoE(expert_models, device).to(device)
    
    # Handle both directory and file paths (same as test.py pattern)
    if os.path.isdir(model_path):
        # Directory path - look for pytorch_model.bin or model.safetensors
        pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
        safetensors_path = os.path.join(model_path, "model.safetensors")
        
        if os.path.exists(pytorch_model_path):
            model_file = pytorch_model_path
            print(f"Loading saved model from: {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=device))
        elif os.path.exists(safetensors_path):
            model_file = safetensors_path
            print(f"Loading saved model from: {model_file}")
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No model file found in directory: {model_path}")
    else:
        # Direct file path
        print(f"Loading saved model from: {model_path}")
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    # Create test dataset
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    print(f'Test dataset: {len(test_dataset)} samples')
    
    # Setup evaluation (same as test.py)
    import datetime
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"RouterMoE_Test_{DATASET_NAME}_{date_str}"
    
    training_args = TrainingArguments(
        output_dir='./temp_test_output',
        run_name=run_name,
        per_device_eval_batch_size=8,
        eval_do_concat_batches=False,
        disable_tqdm=False,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        report_to=[],  # Disable external loggers for testing
        load_best_model_at_end=True,
        metric_for_best_model='eval_map_50',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
    )
    
    # Use the same evaluation function as train.py
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    print("\n=== Testing Router MoE model ===")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Print test results (exactly like train.py and test.py)
    print("\n=== Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return test_results

def main(config_path, epoch=None, dataset=None, weight_moe2=None, weight_dir=None):
    """Main training function following train.py structure exactly."""
    
    # If weight_moe2 is provided, only run testing
    if weight_moe2 is not None:
        print(f"Testing mode: Loading saved Router MoE model from {weight_moe2}")
        return test_saved_moe_model(config_path, weight_moe2, dataset, epoch, weight_dir)
    
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
    
    # Get weight_dir from argument or config - prioritize argument
    if weight_dir is not None:
        expert_weights_dir = weight_dir
        print(f"Using expert weights from argument: {expert_weights_dir}")
    else:
        expert_weights_dir = config.get('moe', {}).get('expert_weights_dir', '/content/Weights')
        print(f"Using expert weights from config: {expert_weights_dir}")
    
    if not os.path.exists(expert_weights_dir):
        raise ValueError(f"Expert weights directory not found: {expert_weights_dir}")
    
    print("Loading expert models...")
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    
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
    
    # ULTRA SIMPLE training arguments for very fast convergence
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=1e-3,  # HIGH learning rate for simple router
        weight_decay=0.0,    # NO weight decay for simple model
        warmup_ratio=0.0,    # NO warmup needed
        lr_scheduler_type='constant',  # CONSTANT learning rate
        lr_scheduler_kwargs={},
        eval_do_concat_batches=eval_do_concat_batches,
        disable_tqdm=False,  # Keep progress bars enabled
        logging_dir=wandb_dir if wandb_dir else "./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        logging_strategy="epoch",  # Log only at epoch level
        report_to="all",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # REDUCED to 0 to avoid tqdm conflicts
        dataloader_pin_memory=False,  # Disable pin memory to fix tqdm
        gradient_accumulation_steps=1,  # NO accumulation
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
    
    # SAVE MODEL BEFORE TRAINING STARTS to avoid losing progress
    date_str = datetime.datetime.now().strftime("%d%m%y")
    save_path = f'../moe_{DATASET_NAME}_{date_str}'
    print(f"Model will be saved to: {save_path}")
    
    try:
        trainer.train()
        
        # Save model after successful training (same as train.py)
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
    
    except Exception as e:
        print(f"Training/Evaluation failed: {e}")
        # Still save the model if training completed but evaluation failed
        trainer.save_model(save_path)
        print(f"Model saved to: {save_path} (despite evaluation error)")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to use (overrides config)')
    parser.add_argument('--weight_moe2', type=str, default=None, help='Path to saved Router MoE model for testing only')
    parser.add_argument('--weight_dir', type=str, default=None, help='Path to directory containing expert model weights (overrides config)')
    args = parser.parse_args()

    main(args.config, args.epoch, args.dataset, args.weight_moe2, args.weight_dir)