import argparse
import os

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")

import pickle
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET
import yaml
from PIL import Image
from functools import partial
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from pathlib import Path
import datetime
from tqdm.auto import tqdm  # Add this import

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    IntervalStrategy,  # Add this to imports
)

from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def is_deformable_detr(model_name):
    """Check if the model is Deformable DETR."""
    return 'deformable-detr' in model_name.lower()

def get_model_config_for_deformable_detr(config):
    """Get specific configuration for Deformable DETR."""
    deformable_config = {}
    
    # Check if there are Deformable DETR specific settings in config
    if 'deformable_detr' in config:
        detr_settings = config['deformable_detr']
        deformable_config.update({
            'num_queries': detr_settings.get('num_queries', 300),
            'num_feature_levels': detr_settings.get('num_feature_levels', 4),
            'dec_n_points': detr_settings.get('dec_n_points', 4),
            'enc_n_points': detr_settings.get('enc_n_points', 4),
            'with_box_refine': detr_settings.get('with_box_refine', True),
            'two_stage': detr_settings.get('two_stage', True),
        })
    
    return deformable_config

def main(config_path, epoch=None, dataset=None):
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    
    # Consistently get splits_dir from dataset section
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)

    # Add wandb folder support
    wandb_dir = None
    if 'wandb' in config and 'wandb_dir' in config['wandb']:
        wandb_dir = config['wandb']['wandb_dir']
        print(f"Wandb directory: {wandb_dir}")

    image_processor = get_image_processor(MODEL_NAME, MAX_SIZE)

    # Get data directories from config - use the same SPLITS_DIR for consistency
    data_config = config['data']
    train_dir = SPLITS_DIR / data_config['train_dir']
    val_dir = SPLITS_DIR / data_config['val_dir']

    train_dataset = BreastCancerDataset(
        split='train',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch  # <-- renamed argument
    )
    val_dataset = BreastCancerDataset(
        split='val',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch  # <-- renamed argument
    )

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

    # Model initialization with Deformable DETR support
    model_kwargs = {
        'id2label': {0: 'cancer'},
        'label2id': {'cancer': 0},
        'auxiliary_loss': False,
        'ignore_mismatched_sizes': True,
    }
    
    # Add Deformable DETR specific configurations
    if is_deformable_detr(MODEL_NAME):
        print("🔧 Configuring Deformable DETR model...")
        deformable_config = get_model_config_for_deformable_detr(config)
        model_kwargs.update(deformable_config)
        
        # Print Deformable DETR configuration
        print("Deformable DETR Configuration:")
        for key, value in deformable_config.items():
            print(f"  {key}: {value}")
    else:
        # Regular YOLOS/DETR configuration
        # model_kwargs['num_queries'] = 8  # Uncomment if needed
        pass

    try:
        model = AutoModelForObjectDetection.from_pretrained(
            MODEL_NAME,
            **model_kwargs
        )
    except Exception as e:
        print(f"⚠️ Failed to load {MODEL_NAME} with custom config: {e}")
        print("🔄 Trying with basic configuration...")
        # Fallback to basic configuration
        model = AutoModelForObjectDetection.from_pretrained(
            MODEL_NAME,
            id2label={0: 'cancer'},
            label2id={'cancer': 0},
            auxiliary_loss=False,
            ignore_mismatched_sizes=True,
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model initialized with {total_params / 1e6:.2f}M parameters')

    # Load training arguments from config with Deformable DETR optimizations
    training_cfg = config.get('training', {})
    output_dir = training_cfg.get('output_dir', '/tmp')
    num_train_epochs = epoch if epoch is not None else training_cfg.get('epochs', 20)
    
    # Adjust batch sizes for Deformable DETR
    if is_deformable_detr(MODEL_NAME):
        # Use smaller batch sizes for Deformable DETR due to memory requirements
        per_device_train_batch_size = min(training_cfg.get('batch_size', 2), 2)
        per_device_eval_batch_size = min(training_cfg.get('batch_size', 2), 2)
        gradient_accumulation_steps = max(training_cfg.get('gradient_accumulation_steps', 16), 16)
        print(f"🔧 Deformable DETR: Using batch_size={per_device_train_batch_size}, grad_accum={gradient_accumulation_steps}")
    else:
        per_device_train_batch_size = training_cfg.get('batch_size', 8)
        per_device_eval_batch_size = training_cfg.get('batch_size', 8)
        gradient_accumulation_steps = training_cfg.get('gradient_accumulation_steps', 2)
    
    learning_rate = training_cfg.get('learning_rate', 5e-5)
    weight_decay = training_cfg.get('weight_decay', 1e-4)
    warmup_ratio = training_cfg.get('warmup_ratio', 0.05)
    lr_scheduler_type = training_cfg.get('lr_scheduler_type', 'cosine_with_restarts')
    lr_scheduler_kwargs = training_cfg.get('lr_scheduler_kwargs', dict(num_cycles=1))
    
    # Additional training arguments for stability
    fp16 = training_cfg.get('fp16', torch.cuda.is_available() and not is_deformable_detr(MODEL_NAME))
    gradient_checkpointing = training_cfg.get('gradient_checkpointing', False)
    max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
    dataloader_num_workers = training_cfg.get('dataloader_num_workers', 2)
    
    # Disable FP16 for Deformable DETR for stability
    if is_deformable_detr(MODEL_NAME):
        fp16 = False
        dataloader_num_workers = 0  # Avoid multiprocessing issues
        print("🔧 Deformable DETR: Disabled FP16 and multiprocessing for stability")

    print(f"Using epoch (num_train_epochs): {num_train_epochs}")

    # Set a unique run_name for wandb
    date_str = datetime.datetime.now().strftime("%d%m%y")
    model_short_name = MODEL_NAME.replace('/', '_').replace('SenseTime_', '').replace('facebook_', '')
    run_name = f"{model_short_name}_{DATASET_NAME}_{date_str}"

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
        eval_do_concat_batches=training_cfg.get('eval_do_concat_batches', False),
        disable_tqdm=False,
        logging_dir=wandb_dir if wandb_dir else "./logs",
        eval_strategy=training_cfg.get('evaluation_strategy', 'epoch'),
        save_strategy=training_cfg.get('save_strategy', 'epoch'),
        save_total_limit=training_cfg.get('save_total_limit', 1),
        logging_strategy=training_cfg.get('logging_strategy', 'epoch'),
        logging_steps=training_cfg.get('logging_steps', 50),
        eval_steps=training_cfg.get('eval_steps', None),
        save_steps=training_cfg.get('save_steps', None),
        report_to="all",
        load_best_model_at_end=training_cfg.get('load_best_model_at_end', True),
        metric_for_best_model=training_cfg.get('metric_for_best_model', 'eval_map_50'),
        greater_is_better=training_cfg.get('greater_is_better', True),
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=training_cfg.get('remove_unused_columns', False),
        dataloader_pin_memory=False,  # Disable for stability
        dataloader_drop_last=False,
    )

    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)

    # Training
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
    
    # Add model type to save path
    date_str = datetime.datetime.now().strftime("%d%m%y")
    if is_deformable_detr(MODEL_NAME):
        save_path = f'../deformable_detr_{DATASET_NAME}_{date_str}'
    else:
        save_path = f'../yolos_{DATASET_NAME}_{date_str}'
    
    trainer.save_model(save_path)
    print(f"✅ Model saved to: {save_path}")

    # Evaluate on test dataset
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
    
    # Print test results
    print("\n=== Test Results ===")
    for key, value in test_results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to use (overrides config)')
    args = parser.parse_args()
    main(args.config, args.epoch, args.dataset)
