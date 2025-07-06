import argparse
import os
from pathlib import Path
import numpy as np
import yaml
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainingArguments
from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn

# Import MoE components from train_moe.py
from train_moe import IntegratedMoE, MoEObjectDetectionModel, get_yolos_model

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_individual_expert(expert_model_path, test_dataset, image_processor, device, expert_name="Expert"):
    """Test individual expert model and return results."""
    print(f"\n=== Testing {expert_name} Model ===")
    
    # Load the expert model
    expert_model = AutoModelForObjectDetection.from_pretrained(
        expert_model_path,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
        ignore_mismatched_sizes=True,
    ).to(device)
    
    # Setup evaluation
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=expert_model,
        args=TrainingArguments(
            output_dir='./temp_output',
            per_device_eval_batch_size=4,  # Smaller batch size for testing
            dataloader_num_workers=0,      # Reduced workers for stability
            remove_unused_columns=False,
            report_to=[],
        ),
        eval_dataset=test_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    try:
        # Evaluate the model
        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix=f'{expert_name.lower()}')
        
        print(f"{expert_name} Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
        return results
    
    except Exception as e:
        print(f"{expert_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_moe_model(moe_model_path, expert_models, test_dataset, image_processors, device):
    """Test MoE model and return results."""
    print(f"\n=== Testing MoE Model ===")
    
    # Create MoE model
    integrated_moe = IntegratedMoE(expert_models, n_models=len(expert_models), top_k=2)
    integrated_moe.load_state_dict(torch.load(moe_model_path, map_location=device))
    integrated_moe.eval()
    
    # Wrap for object detection compatibility
    moe_detector = MoEObjectDetectionModel(integrated_moe)
    
    # Setup evaluation
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processors[0])
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=moe_detector,
        args=TrainingArguments(
            output_dir='./temp_output',
            per_device_eval_batch_size=4,  # Smaller batch size for testing
            dataloader_num_workers=0,      # Reduced workers for stability
            remove_unused_columns=False,
            report_to=[],
        ),
        eval_dataset=test_dataset,
        processing_class=image_processors[0],
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    try:
        # Evaluate the model
        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='moe')
        
        print("MoE Results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
        return results
    
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
    
    # Setup expert model paths
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
    expert_paths = [os.path.join(args.expert_dir, name) for name in expert_names]
    
    # Check if expert paths exist
    for i, path in enumerate(expert_paths):
        if not os.path.exists(path):
            print(f"Warning: Expert model path does not exist: {path}")
            print(f"Available directories in {args.expert_dir}:")
            if os.path.exists(args.expert_dir):
                for item in os.listdir(args.expert_dir):
                    if os.path.isdir(os.path.join(args.expert_dir, item)):
                        print(f"  - {item}")
    
    # Setup image processors
    image_processors = []
    for path in expert_paths:
        if os.path.exists(path):
            image_processors.append(AutoImageProcessor.from_pretrained(path))
        else:
            # Fallback to default if path doesn't exist
            image_processors.append(get_image_processor('hustvl/yolos-base', MAX_SIZE))
    
    # Create test dataset (limited samples)
    print(f"Creating test dataset with {args.num_samples} samples...")
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
        # Create a subset
        indices = list(range(min(args.num_samples, len(test_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test individual expert models
    expert_results = []
    expert_models = []
    
    for i, (expert_path, expert_name) in enumerate(zip(expert_paths, expert_names)):
        if os.path.exists(expert_path):
            # Load expert model
            expert_model = get_yolos_model(expert_path, image_processors[i], 'yolos').to(device)
            expert_models.append(expert_model)
            
            # Test the expert
            result = test_individual_expert(
                expert_path, 
                test_dataset, 
                image_processors[i], 
                device, 
                expert_name
            )
            expert_results.append(result)
        else:
            print(f"Skipping {expert_name} - path does not exist: {expert_path}")
            expert_results.append(None)
    
    # Test MoE model if we have expert models loaded
    if expert_models and os.path.exists(args.moe_model):
        moe_result = test_moe_model(
            args.moe_model,
            expert_models,
            test_dataset,
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
    
    for i, (name, result) in enumerate(zip(expert_names, expert_results)):
        if result:
            map_score = result.get(f'{name.lower()}_map', 'N/A')
            print(f"{name} mAP: {map_score}")
        else:
            print(f"{name}: Failed/Skipped")
    
    if moe_result:
        moe_map = moe_result.get('moe_map', 'N/A')
        print(f"MoE mAP: {moe_map}")
    else:
        print("MoE: Failed/Skipped")

if __name__ == "__main__":
    main()
