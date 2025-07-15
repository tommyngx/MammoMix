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
import yaml
from functools import partial

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)

# Add import for custom MoE model
try:
    from models.moe_model import SimpleMoEModel  # Adjust import path as needed
except ImportError:
    SimpleMoEModel = None

# Import SimpleMoE classes from train_moe3.py
try:
    from train_moe3 import SimpleMoE, SimpleDatasetClassifier, load_expert_models
except ImportError:
    SimpleMoE = None
    SimpleDatasetClassifier = None
    load_expert_models = None

from loader import BreastCancerDataset, collate_fn
from utils import load_config, get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn, run_model_inference_with_map

def get_all_metrics(model, test_dataset, image_processor, device, model_name):
    """Get all metrics (mAP + basic) for any model consistently."""
    # Get mAP metrics using custom function
    map_metrics = run_model_inference_with_map(model, test_dataset, image_processor, device)
    
    # Get basic metrics (loss, runtime, etc.) using Trainer
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"{model_name}_{date_str}"
    
    training_args = TrainingArguments(
        output_dir='./temp_eval',
        run_name=run_name,
        per_device_eval_batch_size=8,
        eval_strategy="no",
        save_strategy="no",
        disable_tqdm=False,
        logging_dir="./logs",
        report_to=[],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=None,
    )
    
    basic_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    # Combine all metrics
    all_metrics = basic_results.copy()
    for key, value in map_metrics.items():
        all_metrics[f'test_{key}'] = value
    
    return all_metrics

def load_model_safe(model_dir, device):
    """Safely load model, handling both standard and custom MoE models."""
    model_dir = Path(model_dir)
    
    # Check if config.json exists to determine model type
    config_path = model_dir / "config.json"
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config.get('model_type', '')
        
        if model_type == 'simple_moe' or 'moe' in str(model_dir).lower():
            print(f"Detected SimpleMoE model, loading with custom loader: {model_dir}")
            
            if SimpleMoE is not None and SimpleDatasetClassifier is not None and load_expert_models is not None:
                try:
                    # Load expert models from parent directory (same as train_moe3.py)
                    expert_weights_dir = model_dir.parent
                    expert_models, expert_processors = load_expert_models(str(expert_weights_dir), device)
                    
                    # Load trained classifier
                    classifier_path = model_dir / 'classifier_best.pth'
                    if not classifier_path.exists():
                        classifier_path = model_dir / 'classifier_final.pth'
                    
                    if not classifier_path.exists():
                        raise FileNotFoundError(f"No classifier found in {model_dir}")
                    
                    # Initialize and load classifier
                    classifier = SimpleDatasetClassifier(num_classes=3, device=device).to(device)
                    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
                    classifier.eval()
                    
                    # Create SimpleMoE model
                    moe_model = SimpleMoE(expert_models, classifier, device).to(device)
                    moe_model.eval()
                    
                    print(f"Successfully loaded SimpleMoE model from {model_dir}")
                    return moe_model
                    
                except Exception as e:
                    print(f"SimpleMoE loading failed: {e}")
                    # Fall through to standard loading
            else:
                print("SimpleMoE classes not available, falling back to standard loading")
    
    # Fallback to standard transformers loading
    try:
        print(f"Loading as standard transformers model: {model_dir}")
        model = AutoModelForObjectDetection.from_pretrained(
            str(model_dir),
            id2label={0: 'cancer'},
            label2id={'cancer': 0},
            auxiliary_loss=False,
            trust_remote_code=True,  # Allow custom code
        )
        return model.to(device)
    except Exception as e:
        print(f"Standard loading failed: {e}")
        
        # Last resort: try with local_files_only=False and trust_remote_code=True
        try:
            model = AutoModelForObjectDetection.from_pretrained(
                str(model_dir),
                id2label={0: 'cancer'},
                label2id={'cancer': 0},
                auxiliary_loss=False,
                trust_remote_code=True,
                local_files_only=False,
            )
            return model.to(device)
        except Exception as e2:
            raise Exception(f"All model loading methods failed. Original error: {e}, Fallback error: {e2}")

def run_test_with_new_metrics(config_path, dataset_name, model_dir, epoch=None):
    """Test model using new metrics function (like test_all.py but with new metrics)."""
    config = load_config(config_path)
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and processor from the specified directory
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = load_model_safe(model_dir, device)
    model.eval()

    # Create test dataset (same as test_all.py)
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )

    # Use get_all_metrics function instead of trainer.evaluate
    all_metrics = get_all_metrics(model, test_dataset, image_processor, device, f"{Path(model_dir).name}_{dataset_name}")
    
    # Format results like test_all.py
    test_results_fmt = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in all_metrics.items()}
    return test_results_fmt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--datasets', type=str, required=True, help='Comma-separated list of dataset names (e.g. CSAW,DDSM,DMID)')
    parser.add_argument('--weight_dir', type=str, required=True, help='Path to directory containing model subfolders')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated list of model subfolder names (e.g. yolos_CSAW,yolos_DDSM,moe_MOMO)')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--output_csv', type=str, default='test_moe_results.csv', help='Output CSV file for merged results')
    args = parser.parse_args()

    dataset_list = [d.strip() for d in args.datasets.split(',')]
    model_list = [m.strip() for m in args.models.split(',')]

    all_results = []
    
    # Test all models (individual experts + MoE) - same pattern as test_all.py
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
                result = run_test_with_new_metrics(args.config, dataset_name, model_dir, args.epoch)
                result_row = {'dataset': dataset_name, 'model': model_name}
                result_row.update(result)
                all_results.append(result_row)
            except Exception as e:
                print(f"Error testing model '{model_dir}' on dataset '{dataset_name}': {e}")

    # Create and display results (same as test_all.py)
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
