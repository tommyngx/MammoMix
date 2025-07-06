import argparse
import os
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Import MoE components from train_moe.py
from train_moe import IntegratedMoE, MoEObjectDetectionModel, get_yolos_model

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_individual_expert(expert_model_path, test_loader, image_processor, device, expert_name="Expert"):
    """Test individual expert model and return results - following test.py approach."""
    print(f"\n=== Testing {expert_name} Model ===")
    
    # Load the expert model
    expert_model = AutoModelForObjectDetection.from_pretrained(
        expert_model_path,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
        ignore_mismatched_sizes=True,
    ).to(device)
    
    expert_model.eval()
    
    # Setup evaluation function
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    all_predictions = []
    all_targets = []
    
    print(f"Evaluating {expert_name} on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            # Get model predictions
            outputs = expert_model(pixel_values)
            
            # Store predictions and targets for metrics calculation
            all_predictions.append(outputs.logits.cpu())
            all_targets.extend(labels)
            
            if batch_idx == 0:
                print(f"  Batch 0 - Images: {pixel_values.shape[0]}, Labels: {len(labels)}")
    
    try:
        # Compute metrics using the same approach as test.py
        predictions = torch.cat(all_predictions, dim=0)
        
        # Create EvalPrediction-like object that evaluation function expects
        class EvalPrediction:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids
        
        eval_pred = EvalPrediction(predictions, all_targets)
        
        metrics = eval_compute_metrics_fn(eval_pred)
        
        print(f"{expert_name} Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
        return metrics
    
    except Exception as e:
        print(f"{expert_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_moe_model(moe_model_path, expert_models, test_loader, image_processors, device):
    """Test MoE model and return results - following test.py approach."""
    print(f"\n=== Testing MoE Model ===")
    
    # Create MoE model
    integrated_moe = IntegratedMoE(expert_models, n_models=len(expert_models), top_k=2)
    integrated_moe.load_state_dict(torch.load(moe_model_path, map_location=device))
    integrated_moe.eval()
    
    # Wrap for object detection compatibility
    moe_detector = MoEObjectDetectionModel(integrated_moe)
    
    # Setup evaluation function
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processors[0])
    
    all_predictions = []
    all_targets = []
    
    print(f"Evaluating MoE on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            # Get MoE predictions
            outputs = moe_detector(pixel_values)
            
            # Store predictions and targets for metrics calculation
            all_predictions.append(outputs.logits.cpu())
            all_targets.extend(labels)
            
            if batch_idx == 0:
                print(f"  Batch 0 - Images: {pixel_values.shape[0]}, Labels: {len(labels)}")
    
    try:
        # Compute metrics using the same approach as test.py
        predictions = torch.cat(all_predictions, dim=0)
        
        # Create EvalPrediction-like object that evaluation function expects
        class EvalPrediction:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids
        
        eval_pred = EvalPrediction(predictions, all_targets)
        
        metrics = eval_compute_metrics_fn(eval_pred)
        
        print("MoE Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
        return metrics
    
    except Exception as e:
        print(f"MoE evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Test sample data with expert models and MoE")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', 
                       help='Path to config yaml')
    parser.add_argument('--expert_dir', type=str, required=True,
                       help='Path to directory containing expert model folders')
    parser.add_argument('--moe_model', type=str, required=True,
                       help='Path to trained MoE model file')
    parser.add_argument('--dataset', type=str, default='CSAW',
                       help='Dataset name to use')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Dataset epoch value')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of test samples to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup expert model paths - only test the expert that matches the dataset
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
    all_expert_paths = [os.path.join(args.expert_dir, name) for name in expert_names]
    
    # Find the expert that matches the current dataset
    dataset_expert_name = f"yolos_{args.dataset}"
    matching_expert_path = os.path.join(args.expert_dir, dataset_expert_name)
    
    # Check if the matching expert exists
    if os.path.exists(matching_expert_path):
        print(f"Found matching expert for dataset {args.dataset}: {dataset_expert_name}")
        existing_paths = [matching_expert_path]
        existing_names = [dataset_expert_name]
    else:
        print(f"Warning: No expert model found for dataset {args.dataset}")
        print(f"Looking for: {matching_expert_path}")
        print(f"Available experts in {args.expert_dir}:")
        if os.path.exists(args.expert_dir):
            for item in os.listdir(args.expert_dir):
                if os.path.isdir(os.path.join(args.expert_dir, item)):
                    print(f"  - {item}")
        print("Will load all available experts for MoE testing...")
        
        # Fallback: load all available experts for MoE
        existing_paths = []
        existing_names = []
        for path, name in zip(all_expert_paths, expert_names):
            if os.path.exists(path):
                existing_paths.append(path)
                existing_names.append(name)
    
    if not existing_paths:
        print("No expert models found! Exiting.")
        return
    
    # Setup image processors for existing models
    image_processors = []
    for path in existing_paths:
        image_processors.append(AutoImageProcessor.from_pretrained(path))
    
    # Create test dataset using first available image processor
    print(f"Creating test dataset for {args.dataset} with {args.num_samples} samples...")
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=args.dataset,
        image_processor=image_processors[0],
        model_type='yolos',
        dataset_epoch=args.epoch
    )
    
    # Limit the dataset to specified number of samples
    if len(test_dataset) > args.num_samples:
        indices = list(range(min(args.num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create DataLoader following test.py approach
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,  # Small batch size for testing
        num_workers=0,  # Reduced workers for stability
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Test loader: {len(test_dataset)} samples")
    
    # Test only the matching expert model (if found)
    expert_results = []
    expert_models = []
    
    if len(existing_paths) == 1 and existing_names[0] == dataset_expert_name:
        # Test only the matching expert
        expert_path = existing_paths[0]
        expert_name = existing_names[0]
        
        print(f"\nTesting expert {expert_name} on {args.dataset} dataset...")
        
        # Load expert model
        expert_model = get_yolos_model(expert_path, image_processors[0], 'yolos').to(device)
        expert_models.append(expert_model)
        
        # Test the expert
        result = test_individual_expert(
            expert_path, 
            test_loader, 
            image_processors[0], 
            device, 
            expert_name
        )
        expert_results.append(result)
        
        # Load all experts for MoE comparison
        print(f"\nLoading all experts for MoE model...")
        for path, name in zip(all_expert_paths, expert_names):
            if os.path.exists(path) and path != expert_path:  # Don't reload the same expert
                additional_processor = AutoImageProcessor.from_pretrained(path)
                additional_model = get_yolos_model(path, additional_processor, 'yolos').to(device)
                expert_models.append(additional_model)
                image_processors.append(additional_processor)
                print(f"  Loaded additional expert: {name}")
    else:
        # Load all available experts
        print(f"\nLoading all available experts...")
        for i, (expert_path, expert_name) in enumerate(zip(existing_paths, existing_names)):
            expert_model = get_yolos_model(expert_path, image_processors[i], 'yolos').to(device)
            expert_models.append(expert_model)
            print(f"  Loaded expert: {expert_name}")
    
    # Test MoE model if we have expert models loaded
    if expert_models and os.path.exists(args.moe_model):
        moe_result = test_moe_model(
            args.moe_model,
            expert_models,
            test_loader,
            image_processors,
            device
        )
    else:
        print("Cannot test MoE model - missing expert models or MoE model file")
        moe_result = None
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {device}")
    print(f"Experts loaded for MoE: {len(expert_models)}")
    
    # Show individual expert result (only the matching one)
    if expert_results and len(expert_results) > 0:
        result = expert_results[0]
        name = existing_names[0] if existing_names else "Expert"
        if result:
            map_score = result.get('map', 'N/A')
            print(f"{name} (dataset-specific) mAP: {map_score}")
        else:
            print(f"{name} (dataset-specific): Failed")
    else:
        print("No dataset-specific expert tested")
    
    if moe_result:
        moe_map = moe_result.get('map', 'N/A')
        print(f"MoE (all experts) mAP: {moe_map}")
    else:
        print("MoE: Failed/Skipped")

if __name__ == "__main__":
    main()
