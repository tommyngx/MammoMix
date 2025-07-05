import os
import argparse
import warnings
import sys
import yaml
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
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
from dataset import BreastCancerDataset, collate_fn
from utils import load_config, get_model_type
from evaluation import get_eval_compute_metrics_fn

def run_test(config_path, dataset_name, model_dir, epoch=None):
    config = load_config(config_path)
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MAX_SIZE = config.get('dataset', {}).get('max_size', 640)

    training_cfg = config.get('training', {})
    output_dir = training_cfg.get('output_dir', '/tmp')
    per_device_eval_batch_size = training_cfg.get('batch_size', 8)
    dataloader_num_workers = training_cfg.get('num_workers', 2)
    remove_unused_columns = training_cfg.get('remove_unused_columns', False)

    # Load processor and model from model_dir
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(
        model_dir,
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        auxiliary_loss=False,
    )
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)

    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(model_dir),
        dataset_epoch=epoch
    )

    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"{Path(model_dir).name}_{dataset_name}_{date_str}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_eval_batch_size=per_device_eval_batch_size,
        disable_tqdm=True,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=remove_unused_columns,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    # Only keep 4 digits for floats
    test_results_fmt = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in test_results.items()}
    return test_results_fmt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--datasets', type=str, required=True, help='Comma-separated list of dataset names (e.g. CSAW,DDSM,DMID)')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated list of model directories (e.g. yolos_CSAW,yolos_DDSM)')
    parser.add_argument('--epoch', type=int, default=None, help='Dataset epoch value to pass to dataset')
    parser.add_argument('--output_csv', type=str, default='test_all_results.csv', help='Output CSV file for merged results')
    args = parser.parse_args()

    dataset_list = [d.strip() for d in args.datasets.split(',')]
    model_list = [m.strip() for m in args.models.split(',')]

    all_results = []
    for dataset_name in dataset_list:
        for model_dir in model_list:
            print(f"\n=== Testing model '{model_dir}' on dataset '{dataset_name}' ===")
            try:
                result = run_test(args.config, dataset_name, model_dir, args.epoch)
                result_row = {'dataset': dataset_name, 'model': model_dir}
                result_row.update(result)
                all_results.append(result_row)
            except Exception as e:
                print(f"Error testing model '{model_dir}' on dataset '{dataset_name}': {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        print("\n=== Merged Results Table ===")
        print(df.to_string(index=False))
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to {args.output_csv}")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
