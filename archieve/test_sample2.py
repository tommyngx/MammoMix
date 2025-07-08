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
from torch.utils.data import DataLoader

from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn
from train_moe import IntegratedMoE, MoEObjectDetectionModel, get_yolos_model

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_expert_model(model_dir, device):
    """Load expert model and processor."""
    print(f"Loading expert model from: {model_dir}")
    
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(
        model_dir,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
    )
    
    model = model.to(device)
    model.eval()
    
    return model, image_processor

def create_test_dataset(splits_dir, dataset_name, image_processor, model_name, epoch=None):
    """Create test dataset."""
    return BreastCancerDataset(
        split='test',
        splits_dir=splits_dir,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(model_name),
        dataset_epoch=epoch
    )

def load_moe_experts(expert_dir, device):
    """Load all available expert models for MoE."""
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM', 'yolos_MOMO']
    expert_paths = [os.path.join(expert_dir, name) for name in expert_names 
                   if os.path.exists(os.path.join(expert_dir, name))]
    
    print(f"Found expert paths: {expert_paths}")
    
    models_list = []
    processors_list = []
    
    for path in expert_paths:
        try:
            processor = AutoImageProcessor.from_pretrained(path)
            expert_model = get_yolos_model(path, processor, 'yolos').to(device)
            expert_model.eval()
            models_list.append(expert_model)
            processors_list.append(processor)
            print(f"Loaded expert from: {path}")
        except Exception as e:
            print(f"Failed to load expert from {path}: {e}")
            continue
    
    return models_list, processors_list

def create_moe_model(models_list, moe_model_path, device):
    """Create and load MoE model."""
    if len(models_list) < 2:
        raise ValueError(f"Need at least 2 expert models for MoE, found {len(models_list)}")
    
    # Create MoE
    integrated_moe = IntegratedMoE(models_list, n_models=len(models_list), top_k=min(2, len(models_list)))
    
    # Load MoE weights
    try:
        state_dict = torch.load(moe_model_path, map_location=device)
        model_keys = set(integrated_moe.state_dict().keys())
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        missing, unexpected = integrated_moe.load_state_dict(filtered_state_dict, strict=False)
        
        if missing:
            print(f"Missing keys in MoE model: {missing[:5]}...")
        if unexpected:
            print(f"Unexpected keys in MoE model: {unexpected[:5]}...")
            
        integrated_moe.eval().to(device)
        print("MoE model loaded successfully")
        
        return integrated_moe
        
    except Exception as e:
        raise RuntimeError(f"Failed to load MoE weights: {e}")

class ValidatedMoEObjectDetectionModel(torch.nn.Module):
    """Wrapper for MoE model that ensures output compatibility."""
    
    def __init__(self, moe_model, reference_processor):
        super().__init__()
        self.moe_model = moe_model
        self.reference_processor = reference_processor
        
    def forward(self, pixel_values, labels=None):
        # Get MoE output - this should now return the corrected YolosObjectDetectionOutput
        moe_output = self.moe_model(pixel_values, labels=labels)
        
        # TEMPORARY FIX: Always ensure loss is present regardless of labels
        # The Trainer evaluation expects loss to always be present
        moe_output.loss = torch.tensor(0.0, device=pixel_values.device, requires_grad=False)
        
        # CRITICAL FIX: Ensure pred_boxes has exactly 4 dimensions for post-processing
        if moe_output.pred_boxes.shape[-1] != 4:
            print(f"WARNING: Fixing pred_boxes from {moe_output.pred_boxes.shape} to 4 dims")
            if moe_output.pred_boxes.shape[-1] > 4:
                moe_output.pred_boxes = moe_output.pred_boxes[..., :4]
            else:
                # Pad with zeros if less than 4
                batch_size, num_queries = moe_output.pred_boxes.shape[:2]
                padding_size = 4 - moe_output.pred_boxes.shape[-1]
                padding = torch.zeros(batch_size, num_queries, padding_size,
                                    device=moe_output.pred_boxes.device, 
                                    dtype=moe_output.pred_boxes.dtype)
                moe_output.pred_boxes = torch.cat([moe_output.pred_boxes, padding], dim=-1)
        
        # Ensure tensors are contiguous
        moe_output.pred_boxes = moe_output.pred_boxes.contiguous()
        moe_output.logits = moe_output.logits.contiguous()
        
        return moe_output

def test_model_output_format(model, test_dataset, device, model_name="Model"):
    """Test and debug model output format."""
    print(f"Testing {model_name} output format...")
    
    debug_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    debug_batch = next(iter(debug_loader))
    
    with torch.no_grad():
        output = model(debug_batch['pixel_values'].to(device))
        print(f"{model_name} output:")
        print(f"  Type: {type(output)}")
        print(f"  Logits shape: {output.logits.shape}")
        print(f"  Pred boxes shape: {output.pred_boxes.shape}")
        print(f"  Pred boxes dtype: {output.pred_boxes.dtype}")
        print(f"  Pred boxes last dim: {output.pred_boxes.shape[-1]}")
        print(f"  Has loss: {hasattr(output, 'loss')}")
        print(f"  Has last_hidden_state: {hasattr(output, 'last_hidden_state')}")
        
    return output

def test_post_processing(model, test_dataset, image_processor, device):
    """Test post-processing compatibility."""
    try:
        debug_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        debug_batch = next(iter(debug_loader))
        
        with torch.no_grad():
            output = model(debug_batch['pixel_values'].to(device))
            test_post_process = image_processor.post_process_object_detection(
                output, 
                threshold=0.1, 
                target_sizes=torch.tensor([[512, 512]])
            )
            print(f"  Post-processing test: SUCCESS")
            return True
    except Exception as pp_error:
        print(f"  Post-processing test: FAILED - {pp_error}")
        return False

def evaluate_model(model, test_dataset, image_processor, config, device, model_name="Model"):
    """Evaluate model and return results."""
    print(f"\n=== Testing {model_name} ===")
    
    try:
        # Test single batch first
        debug_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        debug_batch = next(iter(debug_loader))
        
        with torch.no_grad():
            output = model(debug_batch['pixel_values'].to(device))
            if output.pred_boxes.shape[-1] != 4:
                print(f"ISSUE: {model_name} pred_boxes wrong shape: {output.pred_boxes.shape}")
            else:
                print(f"{model_name} output validation: pred_boxes shape OK {output.pred_boxes.shape}")
        
        # Setup trainer
        training_cfg = config.get('training', {})
        per_device_eval_batch_size = training_cfg.get('batch_size', 8)
        
        # Use batch size 1 for MoE to avoid concatenation issues
        if "moe" in model_name.lower():
            per_device_eval_batch_size = 1
            print(f"Using batch size 1 for MoE evaluation")
        
        training_args = TrainingArguments(
            output_dir=f'./temp_{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_output',
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
            fp16=torch.cuda.is_available(),
            eval_do_concat_batches=False,
        )
        
        eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=image_processor,
            data_collator=collate_fn,
            compute_metrics=eval_compute_metrics_fn,
        )
        
        # Evaluate
        print(f"Starting evaluation of {model_name}...")
        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix=model_name.lower().replace(" ", "_").replace("(", "").replace(")", ""))
        
        print(f"\n=== {model_name} Test Results ===")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
        return results
        
    except Exception as e:
        print(f"{model_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_one_testing_mode(expert_model, test_dataset, moe_model_path, weight_dir, dataset_name, device, image_processor):
    """Run single sample comparison test."""
    print(f"\n=== One Testing Mode: Random Sample Comparison ===")
    
    random_idx = random.randint(0, len(test_dataset) - 1)
    random_sample = test_dataset[random_idx]
    
    print(f"Dataset: {dataset_name}")
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
    
    # Get expert predictions
    pixel_values_single = random_sample['pixel_values'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        expert_pred = expert_model(pixel_values_single)
        print(f"\n=== Expert ({dataset_name}) Predictions ===")
        print(f"  Logits shape: {expert_pred.logits.shape}")
        print(f"  Pred boxes shape: {expert_pred.pred_boxes.shape}")
        
        # Get top 3 predictions
        expert_probs = torch.softmax(expert_pred.logits[0], dim=-1)
        top_expert = torch.topk(expert_probs[:, 1], 3)
        print(f"  Top 3 predictions (confidence scores):")
        for i, (score, idx) in enumerate(zip(top_expert.values, top_expert.indices)):
            print(f"    Query {idx}: {score:.4f}")
        print(f"  Top 3 pred boxes:")
        for i, idx in enumerate(top_expert.indices):
            print(f"    Query {idx}: {expert_pred.pred_boxes[0, idx, :]}")
    
    # Test MoE if provided
    if moe_model_path and os.path.exists(moe_model_path) and weight_dir:
        try:
            print(f"\n=== MoE Predictions ===")
            expert_dir = os.path.dirname(weight_dir)
            models_list, _ = load_moe_experts(expert_dir, device)
            
            if len(models_list) >= 2:
                integrated_moe = create_moe_model(models_list, moe_model_path, device)
                
                # Create MoE - use direct MoEObjectDetectionModel without additional wrapper
                moe_detector = MoEObjectDetectionModel(integrated_moe).to(device)
                
                # Quick validation test
                debug_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
                debug_batch = next(iter(debug_loader))
                
                with torch.no_grad():
                    test_output = moe_detector(debug_batch['pixel_values'].to(device))
                    if test_output.pred_boxes.shape[-1] != 4:
                        print(f"CRITICAL ERROR: MoE still outputs {test_output.pred_boxes.shape[-1]} dimensions!")
                        return
                    else:
                        print(f"✅ MoE validation passed: pred_boxes shape {test_output.pred_boxes.shape}")
                
                # Test post-processing
                try:
                    test_post_process = image_processor.post_process_object_detection(
                        test_output, threshold=0.1, target_sizes=torch.tensor([[640, 640]])
                    )
                    print("✅ Post-processing test: SUCCESS")
                except Exception as e:
                    print(f"❌ Post-processing failed: {e}")
                    return
                
                # Use minimal TrainingArguments for MoE evaluation (no need for complex config)
                training_args = TrainingArguments(
                    output_dir='./temp_moe_output',
                    per_device_eval_batch_size=1,  # Use batch size 1 for MoE
                    eval_do_concat_batches=False,
                    disable_tqdm=False,
                    report_to=[],  # Disable external loggers
                    fp16=torch.cuda.is_available(),
                    dataloader_num_workers=0,
                    remove_unused_columns=False,
                )
                
                eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
                
                trainer = Trainer(
                    model=moe_detector,
                    args=training_args,
                    processing_class=image_processor,
                    data_collator=collate_fn,
                    compute_metrics=eval_compute_metrics_fn,
                )
                
                # Simple evaluation without full config dependency
                print("Running MoE evaluation...")
                try:
                    moe_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='moe')
                    print("\n=== MoE Results ===")
                    for key, value in moe_results.items():
                        if isinstance(value, float):
                            print(f"{key}: {value:.4f}")
                        else:
                            print(f"{key}: {value}")
                except Exception as eval_error:
                    print(f"MoE evaluation failed: {eval_error}")

            else:
                print("  Error: Need at least 2 expert models for MoE")
                
        except Exception as e:
            print(f"  MoE testing failed: {e}")
            import traceback
            traceback.print_exc()
    elif moe_model_path:
        print(f"\n=== MoE Error ===")
        if not os.path.exists(moe_model_path):
            print(f"  MoE model not found: {moe_model_path}")
        elif not weight_dir:
            print(f"  weight_dir required for MoE testing")
    else:
        print(f"\n=== MoE ===")
        print(f"  No MoE model provided")

def print_comparison_summary(expert_results, moe_results, dataset_name):
    """Print comparison summary."""
    print(f"\n=== Summary Comparison ===")
    
    if expert_results:
        expert_map = expert_results.get('expert_map', 'N/A')
        expert_map_50 = expert_results.get('expert_map_50', 'N/A')
        print(f"Expert ({dataset_name}) mAP: {expert_map}")
        print(f"Expert ({dataset_name}) mAP@50: {expert_map_50}")
    else:
        print(f"Expert ({dataset_name}): Evaluation failed")
    
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

def main(config_path, epoch=None, dataset=None, weight_dir=None, num_samples=8, moe_model=None, one_testing=False):
    """Main function."""
    # Load configuration
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    # Determine model directory
    model_dir = weight_dir if weight_dir is not None else f'./yolos_{DATASET_NAME}'
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load expert model
    expert_model, image_processor = load_expert_model(model_dir, device)
    
    # Create test dataset
    test_dataset = create_test_dataset(SPLITS_DIR, DATASET_NAME, image_processor, MODEL_NAME, epoch)
    
    # Handle one testing mode - pass the config as well
    if one_testing:
        run_one_testing_mode(expert_model, test_dataset, moe_model, weight_dir, DATASET_NAME, device, image_processor)
        return
    
    # Normal testing flow
    print(f"\n=== Full Dataset Evaluation Mode ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total test samples: {len(test_dataset)}")
    
    # Limit dataset if specified
    if num_samples != 'all' and len(test_dataset) > num_samples:
        indices = list(range(num_samples))
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f"Limited to: {len(test_dataset)} samples")
    else:
        print(f"Using all {len(test_dataset)} samples")
    
    # Evaluate expert
    expert_results = evaluate_model(expert_model, test_dataset, image_processor, config, device, f"Expert ({DATASET_NAME})")
    
    # Test MoE model
    moe_results = None
    if moe_model and os.path.exists(moe_model) and weight_dir:
        try:
            print(f"\n=== Testing MoE Model ===")
            expert_dir = os.path.dirname(weight_dir)
            models_list, _ = load_moe_experts(expert_dir, device)
            
            if len(models_list) >= 2:
                # Create MoE
                integrated_moe = create_moe_model(models_list, moe_model, device)
                
                # Debug: Compare raw outputs before wrapping
                print(f"\n=== Raw Output Comparison ===")
                debug_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
                debug_batch = next(iter(debug_loader))
                
                with torch.no_grad():
                    # Raw MoE output
                    raw_moe_detector = MoEObjectDetectionModel(integrated_moe).to(device)
                    raw_moe_output = raw_moe_detector(debug_batch['pixel_values'].to(device))
                    
                    # Check if the issue is in the raw MoE output
                    if raw_moe_output.pred_boxes.shape[-1] != 4:
                        print(f"CRITICAL ERROR: Raw MoE pred_boxes has {raw_moe_output.pred_boxes.shape[-1]} dimensions instead of 4")
                        print(f"This should not happen with the fixed train_moe.py")
                        return
                    else:
                        print(f"Raw MoE output: pred_boxes shape OK {raw_moe_output.pred_boxes.shape}")
                
                # Create MoE - use direct MoEObjectDetectionModel without additional wrapper
                integrated_moe = create_moe_model(models_list, moe_model, device)
                moe_detector = MoEObjectDetectionModel(integrated_moe).to(device)
                
                # Quick validation test
                debug_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
                debug_batch = next(iter(debug_loader))
                
                with torch.no_grad():
                    test_output = moe_detector(debug_batch['pixel_values'].to(device))
                    if test_output.pred_boxes.shape[-1] != 4:
                        print(f"CRITICAL ERROR: MoE still outputs {test_output.pred_boxes.shape[-1]} dimensions!")
                        return
                    else:
                        print(f"✅ MoE validation passed: pred_boxes shape {test_output.pred_boxes.shape}")
                
                # Test post-processing
                try:
                    test_post_process = image_processor.post_process_object_detection(
                        test_output, threshold=0.1, target_sizes=torch.tensor([[640, 640]])
                    )
                    print("✅ Post-processing test: SUCCESS")
                except Exception as e:
                    print(f"❌ Post-processing failed: {e}")
                    return
                
                # Use EXACT same evaluation setup as test.py
                training_cfg = config.get('training', {})
                
                # Copy exact TrainingArguments from test.py
                import datetime
                date_str = datetime.datetime.now().strftime("%d%m%y")
                run_name = f"MoE_{DATASET_NAME}_{date_str}"
                
                training_args = TrainingArguments(
                    output_dir='./temp_moe_output',
                    run_name=run_name,
                    per_device_eval_batch_size=training_cfg.get('batch_size', 8),
                    eval_do_concat_batches=training_cfg.get('eval_do_concat_batches', False),
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
                    dataloader_num_workers=0,  # Use 0 for testing to avoid issues
                    gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 2),
                    remove_unused_columns=training_cfg.get('remove_unused_columns', False),
                )
                
                eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
                
                trainer = Trainer(
                    model=moe_detector,
                    args=training_args,
                    processing_class=image_processor,
                    data_collator=collate_fn,  # SAME as test.py
                    compute_metrics=eval_compute_metrics_fn,
                )
                
                # Evaluate MoE
                moe_results = evaluate_model(moe_detector, test_dataset, image_processor, config, device, "MoE")

            else:
                print(f"Error: Need at least 2 expert models for MoE, found {len(models_list)}")
                
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
    
    # Print comparison summary
    print_comparison_summary(expert_results, moe_results, DATASET_NAME)

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