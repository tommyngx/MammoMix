import os
import argparse
import warnings

# Suppress TensorFlow, CUDA, absl, and XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'  # Suppress absl logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Redirect absl and XLA warnings to /dev/null (works on Linux/Unix)
import sys
import logging
import contextlib

class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

# Suppress absl and XLA warnings at runtime
sys.stderr = DevNull()

warnings.filterwarnings("ignore")

import torch
import pickle
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou

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
from train_moe import IntegratedMoE, MoEObjectDetectionModel, get_yolos_model

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt log của TensorFlow (0=verbose, 1=info, 2=warning, 3=error)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Chỉ định GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Suppress Albumentations update warnings



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_moe_model(moe_model_path, expert_dir, test_dataset, image_processor):
    """Test MoE model and print sample outputs"""
    
    print(f"DEBUG: Starting MoE model testing...")
    #print(f"DEBUG: MoE model path: {moe_model_path}")
    #print(f"DEBUG: Expert directory: {expert_dir}")
    
    # Load all expert models for MoE
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
    expert_paths = [os.path.join(expert_dir, name) for name in expert_names]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEBUG: Using device: {device}")
    
    # Load expert models
    models = []
    image_processors = []
    
    for path in expert_paths:
        if os.path.exists(path):
            print(f"DEBUG: Loading expert from {path}")
            processor = AutoImageProcessor.from_pretrained(path)
            model = get_yolos_model(path, processor, 'yolos').to(device)
            models.append(model)
            image_processors.append(processor)
            print(f"Loaded expert: {os.path.basename(path)}")
        else:
            print(f"DEBUG: Expert path not found: {path}")
    
    if not models:
        print("No expert models found for MoE!")
        return None
    
    print(f"DEBUG: Total experts loaded: {len(models)}")
    
    # Create MoE model
    print(f"DEBUG: Creating IntegratedMoE...")
    integrated_moe = IntegratedMoE(models, n_models=len(models), top_k=2)
    
    print(f"DEBUG: Loading MoE state dict...")
    integrated_moe.load_state_dict(torch.load(moe_model_path, map_location=device))
    integrated_moe.eval()
    
    # Move entire MoE model to device
    integrated_moe = integrated_moe.to(device)
    print(f"DEBUG: MoE model loaded and moved to {device}")
    
    # Wrap for compatibility
    print(f"DEBUG: Wrapping MoE for object detection...")
    moe_detector = MoEObjectDetectionModel(integrated_moe)
    moe_detector = moe_detector.to(device)  # Ensure wrapper is also on device
    
    # Setup evaluation
    print(f"DEBUG: Setting up evaluation function...")
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    # Test with a single sample first
    print(f"DEBUG: Testing with single sample...")
    try:
        sample = test_dataset[0]
        print(f"DEBUG: Sample keys: {sample.keys()}")
        print(f"DEBUG: Sample pixel_values shape: {sample['pixel_values'].shape}")
        print(f"DEBUG: Sample labels type: {type(sample['labels'])}")
        
        # Test MoE forward pass
        pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
        print(f"DEBUG: Input shape: {pixel_values.shape}")
        
        with torch.no_grad():
            output = moe_detector(pixel_values)
            print(f"DEBUG: MoE output type: {type(output)}")
            print(f"DEBUG: MoE output has logits: {hasattr(output, 'logits')}")
            if hasattr(output, 'logits'):
                print(f"DEBUG: MoE logits shape: {output.logits.shape}")
    except Exception as e:
        print(f"DEBUG: Single sample test failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create trainer
    print(f"DEBUG: Creating trainer...")
    training_args = TrainingArguments(
        output_dir='./temp_output',
        per_device_eval_batch_size=4,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
    )
    
    trainer = Trainer(
        model=moe_detector,
        args=training_args,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    print(f"DEBUG: Starting trainer evaluation...")
    print(f"\n=== Testing MoE Model ===")
    
    try:
        # Add debug to see what's being passed to evaluation
        print(f"DEBUG: Test dataset type: {type(test_dataset)}")
        print(f"DEBUG: Test dataset length: {len(test_dataset)}")
        
        # Test a single batch to see the data structure
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)
        sample_batch = next(iter(test_loader))
        
        print(f"DEBUG: Sample batch keys: {sample_batch.keys()}")
        print(f"DEBUG: Sample batch labels type: {type(sample_batch['labels'])}")
        print(f"DEBUG: Sample batch labels length: {len(sample_batch['labels'])}")
        if len(sample_batch['labels']) > 0:
            print(f"DEBUG: First label type: {type(sample_batch['labels'][0])}")
        
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='moe')
        
        print("\n=== MoE Test Results ===")
        for key, value in test_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        return test_results
    
    except Exception as e:
        print(f"DEBUG: Trainer evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(config_path, epoch=None, dataset=None, weight_dir=None, num_samples=8, moe_model=None):
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)

    # Add wandb folder support (optional, for consistency)
    wandb_dir = None
    if 'wandb' in config and 'wandb_dir' in config['wandb']:
        wandb_dir = config['wandb']['wandb_dir']

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

    # Fix: define run_name for TrainingArguments
    import datetime
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"{MODEL_NAME.replace('/', '_')}_{DATASET_NAME}_{date_str}"

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
        report_to=[],  # Disable wandb and all external loggers for testing
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=remove_unused_columns,
    )

    # Determine model weight directory
    if weight_dir is not None:
        model_dir = weight_dir
    else:
        model_dir = f'./yolos_{DATASET_NAME}'

    # Load model and processor from the specified directory
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(
        model_dir,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
    )

    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)

    train_dataset = BreastCancerDataset(
        split='train',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    val_dataset = BreastCancerDataset(
        split='val',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    
    # Limit to small sample size for testing
    original_size = len(test_dataset)
    if len(test_dataset) > num_samples:
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f'Original test dataset: {original_size} samples')
    print(f'Limited to: {len(test_dataset)} samples')
    
    # Print individual sample details
    print(f"\n=== Sample Details (First 2) ===")
    for i in range(min(2, len(test_dataset))):
        try:
            sample = test_dataset[i]
            labels = sample['labels']
            print(f"Sample {i}:")
            print(f"  Image shape: {sample['pixel_values'].shape}")
            print(f"  Image dtype: {sample['pixel_values'].dtype}")
            print(f"  Labels type: {type(labels)}")
            
            if isinstance(labels, dict):
                print(f"  Image ID: {labels.get('image_id', 'N/A')}")
                boxes = labels.get('boxes', [])
                print(f"  Boxes shape: {np.array(boxes).shape if len(boxes) > 0 else 'Empty'}")
                print(f"  Boxes: {boxes}")
                classes = labels.get('class_labels', [])
                print(f"  Classes: {classes}")
            print("-" * 50)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
    
    # Test dataset-specific expert model only once
    print(f"\n=== Testing Dataset-Specific Expert: {DATASET_NAME} ===")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    print(f"\n=== Expert ({DATASET_NAME}) Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Test MoE model if provided
    moe_results = None
    if moe_model and os.path.exists(moe_model):
        if weight_dir:
            expert_dir = os.path.dirname(weight_dir)  # Parent directory containing all experts
            try:
                moe_results = test_moe_model(moe_model, expert_dir, test_dataset, image_processor)
            except Exception as e:
                print(f"MoE testing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Warning: weight_dir required for MoE testing (to find expert models)")
    elif moe_model:
        print(f"MoE model not found: {moe_model}")
    
    # Summary comparison
    print(f"\n=== Summary Comparison ===")
    if test_results:
        expert_map = test_results.get('test_map', 'N/A')
        print(f"Expert ({DATASET_NAME}) mAP: {expert_map}")
    
    if moe_results:
        moe_map = moe_results.get('moe_map', 'N/A')
        print(f"MoE (all experts) mAP: {moe_map}")
    else:
        print("MoE: Not tested or failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to use (overrides config)')
    parser.add_argument('--weight_dir', type=str, default=None, help='Path to model folder containing config.json, model.safetensors, preprocessor_config.json')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of test samples to use')
    parser.add_argument('--moe_model', type=str, default=None, help='Path to trained MoE model file')
    args = parser.parse_args()
    main(args.config, args.epoch, args.dataset, args.weight_dir, args.num_samples, args.moe_model)
