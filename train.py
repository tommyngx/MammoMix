import argparse
import os
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
)

from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path, epoch=None, dataset=None):
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('DATASET_NAME', 'CSAW')
    SPLITS_DIR = config.get('SPLITS_DIR', '/content/dataset')
    MODEL_NAME = config.get('MODEL_NAME', 'hustvl/yolos-base')
    MAX_SIZE = config.get('MAX_SIZE', 640)

    # Add wandb folder support
    wandb_dir = None
    if 'wandb' in config and 'wandb_dir' in config['wandb']:
        wandb_dir = config['wandb']['wandb_dir']
        print(f"Wandb directory: {wandb_dir}")

    image_processor = get_image_processor(MODEL_NAME, MAX_SIZE)

    # Get data directories from config
    data_config = config['data']
    splits_dir = Path(config['dataset']['splits_dir'])
    train_dir = splits_dir / data_config['train_dir']
    val_dir = splits_dir / data_config['val_dir']

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

    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        #num_queries=8,
        auxiliary_loss=False,
        #use_pretrained_backbone=False,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model initialized with {total_params / 1e6:.2f}M parameters')

    # Load training arguments from config
    training_cfg = config.get('training', {})
    output_dir = training_cfg.get('output_dir', '/tmp')
    # Use CLI epoch if provided, else from config
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

    # Set a unique run_name for wandb
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
        eval_steps="epoch",  # Changed from evaluation_strategy
        save_steps="epoch",  # Changed from save_strategy
        save_total_limit=save_total_limit,
        logging_steps="epoch",  # Changed from logging_strategy
        report_to="all",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=remove_unused_columns,
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
    trainer.train()
    # Add DDMMYY to the save path
    date_str = datetime.datetime.now().strftime("%d%m%y")
    trainer.save_model(f'../yolos_{DATASET_NAME}_{date_str}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name to use (overrides config)')
    args = parser.parse_args()
    main(args.config, args.epoch, args.dataset)
