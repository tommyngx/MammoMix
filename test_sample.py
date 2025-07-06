import argparse
import os
from pathlib import Path
import numpy as np
import yaml
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainingArguments

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Import everything from test.py that we need
from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn

# Import MoE components
from train_moe import IntegratedMoE, MoEObjectDetectionModel, get_yolos_model

def test_single_expert(expert_path, dataset_name, splits_dir, num_samples=16, epoch=None):
    """Test a single expert model - copied from test.py structure"""
    
    # Load model and processor exactly like test.py
    image_processor = AutoImageProcessor.from_pretrained(expert_path)
    model = AutoModelForObjectDetection.from_pretrained(
        expert_path,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
        ignore_mismatched_sizes=True,
    )
    
    # Create dataset exactly like test.py
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=splits_dir,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type='yolos',
        dataset_epoch=epoch
    )
    
    # Limit samples if needed
    if len(test_dataset) > num_samples:
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"Test loader: {len(test_dataset)} samples")
    
    # Setup evaluation exactly like test.py
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    # Create trainer exactly like test.py
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='./temp_output',
            per_device_eval_batch_size=8,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=[],
        ),
        eval_dataset=test_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    # Evaluate exactly like test.py
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    print("\n=== Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return test_results

def test_moe_model(moe_model_path, expert_dir, dataset_name, splits_dir, num_samples=16, epoch=None):
    """Test MoE model - adapted from test.py structure"""
    
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
            print(f"Loaded expert: {os.path.basename(path)}")
    
    if not models:
        print("No expert models found for MoE!")
        return None
    
    # Create MoE model
    integrated_moe = IntegratedMoE(models, n_models=len(models), top_k=2)
    integrated_moe.load_state_dict(torch.load(moe_model_path, map_location=device))
    integrated_moe.eval()
    
    # Wrap for compatibility
    moe_detector = MoEObjectDetectionModel(integrated_moe)
    
    # Create dataset exactly like test.py
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=splits_dir,
        dataset_name=dataset_name,
        image_processor=image_processors[0],  # Use first processor
        model_type='yolos',
        dataset_epoch=epoch
    )
    
    # Limit samples if needed
    if len(test_dataset) > num_samples:
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    print(f"MoE Test loader: {len(test_dataset)} samples")
    
    # Setup evaluation exactly like test.py
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processors[0])
    
    # Create trainer exactly like test.py
    trainer = Trainer(
        model=moe_detector,
        args=TrainingArguments(
            output_dir='./temp_output',
            per_device_eval_batch_size=8,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=[],
        ),
        eval_dataset=test_dataset,
        processing_class=image_processors[0],
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    # Evaluate exactly like test.py
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='moe')
    
    print("\n=== MoE Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return test_results

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
    splits_dir = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    
    print(f"Testing dataset: {args.dataset}")
    print(f"Splits directory: {splits_dir}")
    print(f"Expert directory: {args.expert_dir}")
    print(f"MoE model: {args.moe_model}")
    
    # Test dataset-specific expert
    dataset_expert_name = f"yolos_{args.dataset}"
    expert_path = os.path.join(args.expert_dir, dataset_expert_name)
    
    expert_result = None
    if os.path.exists(expert_path):
        print(f"\n=== Testing Expert: {dataset_expert_name} ===")
        try:
            expert_result = test_single_expert(
                expert_path, 
                args.dataset, 
                splits_dir, 
                args.num_samples, 
                args.epoch
            )
        except Exception as e:
            print(f"Expert test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Expert model not found: {expert_path}")
    
    # Test MoE model
    moe_result = None
    if os.path.exists(args.moe_model):
        print(f"\n=== Testing MoE Model ===")
        try:
            moe_result = test_moe_model(
                args.moe_model,
                args.expert_dir,
                args.dataset,
                splits_dir,
                args.num_samples,
                args.epoch
            )
        except Exception as e:
            print(f"MoE test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"MoE model not found: {args.moe_model}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Test samples: {args.num_samples}")
    
    if expert_result:
        expert_map = expert_result.get('test_map', 'N/A')
        print(f"{dataset_expert_name} mAP: {expert_map}")
    else:
        print(f"{dataset_expert_name}: Failed/Not Found")
    
    if moe_result:
        moe_map = moe_result.get('moe_map', 'N/A')
        print(f"MoE mAP: {moe_map}")
    else:
        print("MoE: Failed/Not Found")

if __name__ == "__main__":
    main()
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
