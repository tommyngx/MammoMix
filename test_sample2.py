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
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

warnings.filterwarnings("ignore")

import torch
import yaml
from pathlib import Path
import random

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn
from train_moe import IntegratedMoE, MoEObjectDetectionModel, get_yolos_model

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path, epoch=None, dataset=None, weight_dir=None, num_samples=8, moe_model=None, one_testing=False):
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')

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
    
    # Move model to device and set to eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )

    # If one_testing is True, only do random sample comparison
    if one_testing:
        print(f"\n=== One Testing Mode: Random Sample Comparison ===")
        random_idx = random.randint(0, len(test_dataset) - 1)
        random_sample = test_dataset[random_idx]
        
        print(f"Dataset: {DATASET_NAME}")
        print(f"Random sample index: {random_idx}")
        print(f"Total test samples: {len(test_dataset)}")
        
        # Print Ground Truth
        print(f"\n=== Ground Truth ===")
        gt_labels = random_sample['labels']
        if hasattr(gt_labels, 'data'):
            print(f"  Labels type: BatchFeature")
            for key, value in gt_labels.data.items():
                print(f"    {key}: {value}")
        elif isinstance(gt_labels, dict):
            print(f"  Labels type: dict")
            for key, value in gt_labels.items():
                print(f"    {key}: {value}")
        else:
            print(f"  Labels type: {type(gt_labels)}")
            print(f"  Labels content: {gt_labels}")
        
        # Get predictions from expert
        pixel_values_single = random_sample['pixel_values'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            expert_pred = model(pixel_values_single)
            print(f"\n=== Expert ({DATASET_NAME}) Predictions ===")
            print(f"  Logits shape: {expert_pred.logits.shape}")
            print(f"  Pred boxes shape: {expert_pred.pred_boxes.shape}")
            
            # Get top 3 predictions
            expert_probs = torch.softmax(expert_pred.logits[0], dim=-1)
            top_expert = torch.topk(expert_probs[:, 1], 3)  # Get top 3 for class 1 (cancer)
            print(f"  Top 3 predictions (confidence scores):")
            for i, (score, idx) in enumerate(zip(top_expert.values, top_expert.indices)):
                print(f"    Query {idx}: {score:.4f}")
            print(f"  Top 3 pred boxes:")
            for i, idx in enumerate(top_expert.indices):
                print(f"    Query {idx}: {expert_pred.pred_boxes[0, idx, :]}")
        
        # Test MoE model if provided
        if moe_model and os.path.exists(moe_model) and weight_dir:
            try:
                print(f"\n=== MoE Predictions ===")
                expert_dir = os.path.dirname(weight_dir)
                expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
                expert_paths = [os.path.join(expert_dir, name) for name in expert_names if os.path.exists(os.path.join(expert_dir, name))]
                
                if expert_paths:
                    models_list = []
                    for path in expert_paths:
                        processor = AutoImageProcessor.from_pretrained(path)
                        expert_model = get_yolos_model(path, processor, 'yolos').to(device)
                        expert_model.eval()
                        models_list.append(expert_model)
                    
                    if len(models_list) >= 2:
                        integrated_moe = IntegratedMoE(models_list, n_models=len(models_list), top_k=2)
                        # Load with strict=False to ignore mismatched keys
                        state_dict = torch.load(moe_model, map_location=device)
                        missing, unexpected = integrated_moe.load_state_dict(state_dict, strict=False)
                        integrated_moe.eval().to(device)
                        moe_detector = MoEObjectDetectionModel(integrated_moe).to(device)
                        
                        with torch.no_grad():
                            moe_pred = moe_detector(pixel_values_single)
                            print(f"  Logits shape: {moe_pred.logits.shape}")
                            print(f"  Pred boxes shape: {moe_pred.pred_boxes.shape}")
                            
                            # Get top 3 predictions
                            moe_probs = torch.softmax(moe_pred.logits[0], dim=-1)
                            top_moe = torch.topk(moe_probs[:, 1], 3)  # Get top 3 for class 1 (cancer)
                            print(f"  Top 3 predictions (confidence scores):")
                            for i, (score, idx) in enumerate(zip(top_moe.values, top_moe.indices)):
                                print(f"    Query {idx}: {score:.4f}")
                            print(f"  Top 3 pred boxes:")
                            for i, idx in enumerate(top_moe.indices):
                                print(f"    Query {idx}: {moe_pred.pred_boxes[0, idx, :]}")
                        
                        print(f"\n=== Comparison ===")
                        print(f"  Expert top confidence: {top_expert.values[0]:.4f}")
                        print(f"  MoE top confidence: {top_moe.values[0]:.4f}")
                        print(f"  Confidence difference (MoE - Expert): {(top_moe.values[0] - top_expert.values[0]):.4f}")
                        
                        # Compare top prediction boxes
                        expert_top_box = expert_pred.pred_boxes[0, top_expert.indices[0], :]
                        moe_top_box = moe_pred.pred_boxes[0, top_moe.indices[0], :]
                        box_diff = moe_top_box - expert_top_box
                        print(f"  Top box difference (MoE - Expert): {box_diff}")
                        
                    else:
                        print("  Error: Need at least 2 expert models for MoE")
                else:
                    print("  Error: No expert models found")
                    
            except Exception as e:
                print(f"  MoE testing failed: {e}")
                import traceback
                traceback.print_exc()
        elif moe_model:
            print(f"\n=== MoE Error ===")
            if not os.path.exists(moe_model):
                print(f"  MoE model not found: {moe_model}")
            elif not weight_dir:
                print(f"  weight_dir required for MoE testing")
        else:
            print(f"\n=== MoE ===")
            print(f"  No MoE model provided")
        
        return  # Exit early, skip all other tests
    
    # Normal testing flow - full dataset evaluation
    print(f"\n=== Full Dataset Evaluation Mode ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total test samples: {len(test_dataset)}")
    
    # Limit dataset if num_samples is specified
    if num_samples != 'all' and len(test_dataset) > num_samples:
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f"Limited to: {len(test_dataset)} samples")
    else:
        print(f"Using all {len(test_dataset)} samples")
    
    # Setup trainer for expert evaluation
    training_cfg = config.get('training', {})
    per_device_eval_batch_size = training_cfg.get('batch_size', 8)
    
    training_args = TrainingArguments(
        output_dir='./temp_output',
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        fp16=torch.cuda.is_available(),
        eval_do_concat_batches=False,  # Add this to prevent batch concatenation issues
    )
    
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    expert_trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    # Evaluate expert
    print(f"\n=== Testing Expert ({DATASET_NAME}) ===")
    try:
        expert_results = expert_trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='expert')
        print(f"\n=== Expert ({DATASET_NAME}) Test Results ===")
        for key, value in expert_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Expert evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        expert_results = None
    
    # Test MoE model if provided
    moe_results = None
    if moe_model and os.path.exists(moe_model) and weight_dir:
        try:
            print(f"\n=== Testing MoE Model ===")
            expert_dir = os.path.dirname(weight_dir)
            expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
            expert_paths = [os.path.join(expert_dir, name) for name in expert_names if os.path.exists(os.path.join(expert_dir, name))]
            
            if expert_paths:
                models_list = []
                for path in expert_paths:
                    processor = AutoImageProcessor.from_pretrained(path)
                    expert_model = get_yolos_model(path, processor, 'yolos').to(device)
                    expert_model.eval()
                    models_list.append(expert_model)
                
                if len(models_list) >= 2:
                    integrated_moe = IntegratedMoE(models_list, n_models=len(models_list), top_k=2)
                    # Load with strict=False to ignore mismatched keys
                    state_dict = torch.load(moe_model, map_location=device)
                    missing, unexpected = integrated_moe.load_state_dict(state_dict, strict=False)
                    integrated_moe.eval().to(device)
                    moe_detector = MoEObjectDetectionModel(integrated_moe).to(device)
                    
                    # Create MoE trainer with same settings
                    moe_training_args = TrainingArguments(
                        output_dir='./temp_output',
                        per_device_eval_batch_size=1,  # Use batch size 1 for MoE to avoid target size mismatch
                        dataloader_num_workers=0,
                        remove_unused_columns=False,
                        report_to=[],
                        fp16=torch.cuda.is_available(),
                        eval_do_concat_batches=False,  # Add this to prevent batch concatenation issues
                    )
                    
                    moe_trainer = Trainer(
                        model=moe_detector,
                        args=moe_training_args,
                        processing_class=image_processor,
                        data_collator=collate_fn,
                        compute_metrics=eval_compute_metrics_fn,
                    )
                    
                    # Evaluate MoE
                    try:
                        moe_results = moe_trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='moe')
                        print(f"\n=== MoE Test Results ===")
                        for key, value in moe_results.items():
                            if isinstance(value, float):
                                print(f"{key}: {value:.4f}")
                            else:
                                print(f"{key}: {value}")
                    except Exception as e:
                        print(f"MoE evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        moe_results = None
                    
                else:
                    print("Error: Need at least 2 expert models for MoE")
            else:
                print("Error: No expert models found")
                
        except Exception as e:
            print(f"MoE testing failed: {e}")
            import traceback
            traceback.print_exc()
    elif moe_model:
        print(f"\n=== MoE Error ===")
        if not os.path.exists(moe_model):
            print(f"MoE model not found: {moe_model}")
        elif not weight_dir:
            print(f"weight_dir required for MoE testing")
    else:
        print(f"\n=== MoE Not Provided ===")
    
    # Summary comparison
    print(f"\n=== Summary Comparison ===")
    if expert_results:
        expert_map = expert_results.get('expert_map', 'N/A')
        expert_map_50 = expert_results.get('expert_map_50', 'N/A')
        print(f"Expert ({DATASET_NAME}) mAP: {expert_map}")
        print(f"Expert ({DATASET_NAME}) mAP@50: {expert_map_50}")
    else:
        print(f"Expert ({DATASET_NAME}): Evaluation failed")
    
    if moe_results:
        moe_map = moe_results.get('moe_map', 'N/A')
        moe_map_50 = moe_results.get('moe_map_50', 'N/A')
        print(f"MoE (all experts) mAP: {moe_map}")
        print(f"MoE (all experts) mAP@50: {moe_map_50}")
        
        # Calculate improvement
        if expert_results and isinstance(expert_map, float) and isinstance(moe_map, float):
            improvement = moe_map - expert_map
            improvement_pct = (improvement / expert_map) * 100 if expert_map != 0 else 0
            print(f"mAP Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)")
        
        if expert_results and isinstance(expert_map_50, float) and isinstance(moe_map_50, float):
            improvement_50 = moe_map_50 - expert_map_50
            improvement_50_pct = (improvement_50 / expert_map_50) * 100 if expert_map_50 != 0 else 0
            print(f"mAP@50 Improvement: {improvement_50:.4f} ({improvement_50_pct:+.2f}%)")
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
    parser.add_argument('--one_testing', action='store_true', help='Only run single random sample comparison test')
    args = parser.parse_args()
    
    # Convert num_samples to int if it's not "all"
    if args.num_samples != 'all':
        args.num_samples = int(args.num_samples)
    
    main(args.config, args.epoch, args.dataset, args.weight_dir, args.num_samples, args.moe_model, args.one_testing)
