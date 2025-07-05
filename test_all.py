import os
import argparse
import warnings
import sys
import datetime
from pathlib import Path
import pandas as pd

# Suppress TensorFlow, CUDA, absl, and XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

sys.stderr = DevNull()
warnings.filterwarnings("ignore")

import torch
import pickle
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET
import yaml
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

def run_test(config_path, dataset_name, model_dir, epoch=None):
    config = load_config(config_path)
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
    per_device_train_batch_size = training_cfg.get('batch_size', 4)
    per_device_eval_batch_size = training_cfg.get('batch_size', 4)
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

    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"{Path(model_dir).name}_{dataset_name}_{date_str}"

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
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    test_results_fmt = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in test_results.items()}
    return test_results_fmt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--datasets', type=str, required=True, help='Comma-separated list of dataset names (e.g. CSAW,DDSM,DMID)')
    parser.add_argument('--weight_dir', type=str, required=True, help='Path to directory containing model subfolders')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated list of model subfolder names (e.g. yolos_CSAW,yolos_DDSM)')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--output_csv', type=str, default='test_all_results.csv', help='Output CSV file for merged results')
    args = parser.parse_args()

    dataset_list = [d.strip() for d in args.datasets.split(',')]
    model_list = [m.strip() for m in args.models.split(',')]

    all_results = []
    for model_name in model_list:
        model_dir = os.path.join(args.weight_dir, model_name)
        model_dir = os.path.abspath(model_dir)
        # Test on all datasets if model_name does not match any dataset name (case-insensitive)
        datasets_to_test = []
        for ds in dataset_list:
            if model_name.lower().endswith(ds.lower()):
                datasets_to_test = [ds]
                break
        if not datasets_to_test:
            datasets_to_test = dataset_list
        for dataset_name in datasets_to_test:
            print(f"\n=== Testing model '{model_dir}' on dataset '{dataset_name}' ===")
            try:
                result = run_test(args.config, dataset_name, model_dir, args.epoch)
                result_row = {'dataset': dataset_name, 'model': model_name}
                result_row.update(result)
                all_results.append(result_row)
            except Exception as e:
                print(f"Error testing model '{model_dir}' on dataset '{dataset_name}': {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n=== Merged Results Table ===")
        df = df.set_index(['dataset', 'model']).T  # Reverse: metrics as rows, dataset/model as columns
        print(df.to_string())
        df.to_csv(args.output_csv)
        print(f"\nResults saved to {args.output_csv}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
