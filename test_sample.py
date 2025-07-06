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
    
    # Load all expert models for MoE
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
    expert_paths = [os.path.join(expert_dir, name) for name in expert_names]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load expert models
    models = []
    image_processors = []
    
    for path in expert_paths:
        if os.path.exists(path):
            processor = AutoImageProcessor.from_pretrained(path)
            model = get_yolos_model(path, processor, 'yolos').to(device)
            models.append(model)
            image_processors.append(processor)
            #print(f"Loaded expert: {os.path.basename(path)}")
    
    if not models:
        print("No expert models found for MoE!")
        return None
    
    # Create MoE model
    integrated_moe = IntegratedMoE(models, n_models=len(models), top_k=2)
    integrated_moe.load_state_dict(torch.load(moe_model_path, map_location=device))
    integrated_moe.eval()
    
    # Move entire MoE model to device
    integrated_moe = integrated_moe.to(device)
    
    # Wrap for compatibility
    moe_detector = MoEObjectDetectionModel(integrated_moe)
    moe_detector = moe_detector.to(device)
    
    # Setup evaluation
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    # Create trainer
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
    
    print(f"\n=== Testing MoE Model ===")
    
    try:
        # Test a single batch to see the data structure
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)
        sample_batch = next(iter(test_loader))
        
        print(f"DEBUG: Test dataset length for MoE: {len(test_dataset)}")
        print(f"DEBUG: Sample batch labels length: {len(sample_batch['labels'])}")
        
        # Test MoE model output on this batch
        pixel_values = sample_batch['pixel_values'].to(device)
        
        with torch.no_grad():
            moe_output = moe_detector(pixel_values)
            print(f"DEBUG: MoE logits shape: {moe_output.logits.shape}")
            print(f"DEBUG: MoE logits sample: {moe_output.logits[0, :3, :]}")
        
        # Skip trainer evaluation and use manual evaluation
        print(f"\nDEBUG: Attempting manual evaluation...")
        
        # Collect all predictions manually using the SAME test_dataset that expert used
        all_predictions = []
        all_targets = []
        
        eval_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
        
        for batch_idx, batch in enumerate(eval_loader):
            pixel_values_batch = batch['pixel_values'].to(device)
            labels_batch = batch['labels']
            
            with torch.no_grad():
                outputs = moe_detector(pixel_values_batch)
                all_predictions.append(outputs.logits.cpu())
                all_targets.extend(labels_batch)
                
            #print(f"DEBUG: Batch {batch_idx}, predictions shape: {outputs.logits.shape}, targets: {len(labels_batch)}")
        
        print(f"DEBUG: Total predictions collected: {len(all_predictions)}")
        print(f"DEBUG: Total targets collected: {len(all_targets)}")
        
        # Format predictions exactly like the trainer does
        predictions_formatted = []
        for i in range(len(all_predictions)):
            batch_logits = all_predictions[i].numpy()
            batch_pred_boxes = torch.zeros(batch_logits.shape[0], batch_logits.shape[1], 4).numpy()
            predictions_formatted.append((None, batch_logits, batch_pred_boxes))
        
        # Format targets as list of batches
        targets_formatted = []
        batch_size = 2
        for i in range(0, len(all_targets), batch_size):
            batch_targets = all_targets[i:i+batch_size]
            targets_formatted.append(batch_targets)
            
        print(f"DEBUG: Formatted predictions: {len(predictions_formatted)} batches")
        print(f"DEBUG: Formatted targets: {len(targets_formatted)} batches")
        
        # Create evaluation object
        from types import SimpleNamespace
        eval_pred = SimpleNamespace()
        eval_pred.predictions = predictions_formatted
        eval_pred.label_ids = targets_formatted
        
        # Add debugging before calling evaluation
        print(f"DEBUG: First prediction batch shape: {predictions_formatted[0][1].shape}")
        print(f"DEBUG: First prediction values sample: {predictions_formatted[0][1][0, :3, :]}")
        print(f"DEBUG: First target batch length: {len(targets_formatted[0])}")
        if len(targets_formatted[0]) > 0:
            print(f"DEBUG: First target sample keys: {targets_formatted[0][0].keys()}")
            print(f"DEBUG: First target boxes: {targets_formatted[0][0].get('boxes', [])}")
        
        # Call evaluation function directly
        metrics = eval_compute_metrics_fn(eval_pred)
        
        print("\n=== MoE Test Results (Manual) ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"moe_{key}: {value:.4f}")
            else:
                print(f"moe_{key}: {value}")
        
        return {f"moe_{k}": v for k, v in metrics.items()}
    
    except Exception as e:
        print(f"DEBUG: Manual evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main(config_path, epoch=None, dataset=None, weight_dir=None, num_samples=8, moe_model=None, one_test=False):
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
    
    # If one_test is True, only do random sample comparison
    if one_test:
        print(f"\n=== One Test Mode: Random Sample Comparison ===")
        import random
        random_idx = random.randint(0, len(test_dataset) - 1)
        random_sample = test_dataset[random_idx]
        
        print(f"Random sample index: {random_idx}")
        print(f"Ground truth:")
        gt_labels = random_sample['labels']
        if isinstance(gt_labels, dict):
            print(f"  Image ID: {gt_labels.get('image_id', 'N/A')}")
            gt_boxes = gt_labels.get('boxes', [])
            print(f"  GT Boxes: {gt_boxes}")
            print(f"  GT Classes: {gt_labels.get('class_labels', [])}")
        
        # Get predictions from expert
        pixel_values_single = random_sample['pixel_values'].unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        with torch.no_grad():
            expert_pred = model(pixel_values_single)
            print(f"\nExpert predictions:")
            print(f"  Logits shape: {expert_pred.logits.shape}")
            print(f"  Top 3 predictions (confidence scores):")
            expert_probs = torch.softmax(expert_pred.logits[0], dim=-1)
            top_expert = torch.topk(expert_probs[:, 1], 3)  # Get top 3 for class 1 (cancer)
            for i, (score, idx) in enumerate(zip(top_expert.values, top_expert.indices)):
                print(f"    Query {idx}: {score:.4f}")
            print(f"  Pred boxes (top 3): {expert_pred.pred_boxes[0, top_expert.indices[:3], :]}")
        
        # Test MoE model if provided
        if moe_model and os.path.exists(moe_model) and weight_dir:
            try:
                print(f"\nMoE predictions for same sample:")
                expert_dir = os.path.dirname(weight_dir)
                expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
                expert_paths = [os.path.join(expert_dir, name) for name in expert_names if os.path.exists(os.path.join(expert_dir, name))]
                
                if expert_paths:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    models = []
                    for path in expert_paths[:4]:  # Load all available experts
                        processor = AutoImageProcessor.from_pretrained(path)
                        expert_model = get_yolos_model(path, processor, 'yolos').to(device)
                        models.append(expert_model)
                    
                    if len(models) >= 2:
                        integrated_moe = IntegratedMoE(models, n_models=len(models), top_k=2)
                        integrated_moe.load_state_dict(torch.load(moe_model, map_location=device))
                        integrated_moe.eval().to(device)
                        moe_detector = MoEObjectDetectionModel(integrated_moe).to(device)
                        
                        with torch.no_grad():
                            moe_pred = moe_detector(pixel_values_single)
                            print(f"  Logits shape: {moe_pred.logits.shape}")
                            print(f"  Top 3 predictions (confidence scores):")
                            moe_probs = torch.softmax(moe_pred.logits[0], dim=-1)
                            top_moe = torch.topk(moe_probs[:, 1], 3)  # Get top 3 for class 1 (cancer)
                            for i, (score, idx) in enumerate(zip(top_moe.values, top_moe.indices)):
                                print(f"    Query {idx}: {score:.4f}")
                            print(f"  Pred boxes (top 3): {moe_pred.pred_boxes[0, top_moe.indices[:3], :]}")
                        
                        print(f"\nComparison:")
                        print(f"  Expert top confidence: {top_expert.values[0]:.4f}")
                        print(f"  MoE top confidence: {top_moe.values[0]:.4f}")
                        print(f"  Difference: {(top_moe.values[0] - top_expert.values[0]):.4f}")
            except Exception as e:
                print(f"MoE one test failed: {e}")
        
        return  # Exit early, skip all other tests
    
    # Continue with normal testing flow if not one_test mode
    # Limit to small sample size for testing
    original_size = len(test_dataset)
    print(f'DEBUG: num_samples value: {num_samples}, type: {type(num_samples)}')
    print(f'DEBUG: original dataset size: {original_size}')
    
    if num_samples != 'all' and len(test_dataset) > num_samples:
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f'DEBUG: Dataset limited from {original_size} to {len(test_dataset)}')
    else:
        print(f'DEBUG: Using full dataset of {len(test_dataset)} samples')
    
    print(f'Original test dataset: {original_size} samples')
    print(f'Limited to: {len(test_dataset)} samples')
    
    # Test dataset-specific expert model only once
    print(f"\n=== Testing Dataset-Specific Expert: {DATASET_NAME} ===")
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    print(f"\n=== Expert ({DATASET_NAME}) Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Show expert model output for comparison
    print(f"\n=== Expert Model Output Analysis ===")
    test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)
    sample_batch = next(iter(test_loader))
    pixel_values = sample_batch['pixel_values'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    with torch.no_grad():
        expert_output = model(pixel_values)
        print(f"Expert output type: {type(expert_output)}")
        print(f"Expert logits shape: {expert_output.logits.shape}")
        print(f"Expert pred_boxes shape: {expert_output.pred_boxes.shape}")
        print(f"Expert logits sample (first 3 queries):")
        print(expert_output.logits[0, :3, :])
    
    # Random sample comparison between Expert, MoE and Ground Truth
    print(f"\n=== Random Sample Comparison ===")
    import random
    random_idx = random.randint(0, len(test_dataset) - 1)
    random_sample = test_dataset[random_idx]
    
    print(f"Random sample index: {random_idx}")
    print(f"Ground truth:")
    gt_labels = random_sample['labels']
    if isinstance(gt_labels, dict):
        print(f"  Image ID: {gt_labels.get('image_id', 'N/A')}")
        gt_boxes = gt_labels.get('boxes', [])
        print(f"  GT Boxes: {gt_boxes}")
        print(f"  GT Classes: {gt_labels.get('class_labels', [])}")
    
    # Get predictions from expert
    pixel_values_single = random_sample['pixel_values'].unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    with torch.no_grad():
        expert_pred = model(pixel_values_single)
        print(f"\nExpert predictions:")
        print(f"  Logits shape: {expert_pred.logits.shape}")
        print(f"  Top 3 predictions (confidence scores):")
        expert_probs = torch.softmax(expert_pred.logits[0], dim=-1)
        top_expert = torch.topk(expert_probs[:, 1], 3)  # Get top 3 for class 1 (cancer)
        for i, (score, idx) in enumerate(zip(top_expert.values, top_expert.indices)):
            print(f"    Query {idx}: {score:.4f}")
        print(f"  Pred boxes (top 3): {expert_pred.pred_boxes[0, top_expert.indices[:3], :]}")
        
    # Test MoE model if provided
    moe_results = None
    if moe_model and os.path.exists(moe_model):
        if weight_dir:
            expert_dir = os.path.dirname(weight_dir)  # Parent directory containing all experts
            try:
                # First get MoE prediction for the same sample
                print(f"\nMoE predictions for same sample:")
                
                # Load MoE components quickly for single prediction
                expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
                expert_paths = [os.path.join(expert_dir, name) for name in expert_names if os.path.exists(os.path.join(expert_dir, name))]
                
                if expert_paths:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    models = []
                    for path in expert_paths[:2]:  # Just load 2 for quick test
                        processor = AutoImageProcessor.from_pretrained(path)
                        expert_model = get_yolos_model(path, processor, 'yolos').to(device)
                        models.append(expert_model)
                    
                    if len(models) >= 2:
                        integrated_moe = IntegratedMoE(models, n_models=len(models), top_k=2)
                        integrated_moe.load_state_dict(torch.load(moe_model, map_location=device))
                        integrated_moe.eval().to(device)
                        moe_detector = MoEObjectDetectionModel(integrated_moe).to(device)
                        
                        with torch.no_grad():
                            moe_pred = moe_detector(pixel_values_single)
                            print(f"  Logits shape: {moe_pred.logits.shape}")
                            print(f"  Top 3 predictions (confidence scores):")
                            moe_probs = torch.softmax(moe_pred.logits[0], dim=-1)
                            top_moe = torch.topk(moe_probs[:, 1], 3)  # Get top 3 for class 1 (cancer)
                            for i, (score, idx) in enumerate(zip(top_moe.values, top_moe.indices)):
                                print(f"    Query {idx}: {score:.4f}")
                            print(f"  Pred boxes (top 3): {moe_pred.pred_boxes[0, top_moe.indices[:3], :]}")
                        
                        print(f"\nComparison:")
                        print(f"  Expert top confidence: {top_expert.values[0]:.4f}")
                        print(f"  MoE top confidence: {top_moe.values[0]:.4f}")
                        print(f"  Difference: {(top_moe.values[0] - top_expert.values[0]):.4f}")
                
                # Now run full MoE evaluation
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
    parser.add_argument('--num_samples', default=8, help='Number of test samples to use (or "all" for full dataset)')
    parser.add_argument('--moe_model', type=str, default=None, help='Path to trained MoE model file')
    parser.add_argument('--one_test', action='store_true', help='Only run single sample comparison test')
    args = parser.parse_args()
    
    # Convert num_samples to int if it's not "all"
    if args.num_samples != 'all':
        args.num_samples = int(args.num_samples)
    
    main(args.config, args.epoch, args.dataset, args.weight_dir, args.num_samples, args.moe_model, args.one_test)
