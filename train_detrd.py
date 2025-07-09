import argparse
import os

# Suppress warnings specifically for Deformable DETR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*Some weights of.*were not initialized.*")

import torch
import datetime
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer

# Import reusable functions from train.py
from train import load_config
from dataset import BreastCancerDataset, collate_fn
from utils import get_image_processor, get_model_type
from evaluation import get_eval_compute_metrics_fn, calculate_custom_map_metrics


def load_deformable_detr_model(model_name, config):
    """Load Deformable DETR model with proper configuration."""
    print(f"🔧 Loading Deformable DETR: {model_name}")
    
    # Basic model kwargs
    model_kwargs = {
        'id2label': {0: 'cancer'},
        'label2id': {'cancer': 0},
        'ignore_mismatched_sizes': True,
    }
    
    # Add Deformable DETR specific settings if available in config
    if 'deformable_detr' in config:
        detr_config = config['deformable_detr']
        model_kwargs.update({
            'num_queries': detr_config.get('num_queries', 300),
            'auxiliary_loss': False,  # Keep simple for now
        })
        print(f"   Num queries: {model_kwargs['num_queries']}")
    
    try:
        model = AutoModelForObjectDetection.from_pretrained(model_name, **model_kwargs)
        print("✅ Deformable DETR loaded successfully")
        return model
    except Exception as e:
        print(f"⚠️ Failed with custom config: {e}")
        print("🔄 Trying basic configuration...")
        
        # Fallback to basic config
        model = AutoModelForObjectDetection.from_pretrained(
            model_name,
            id2label={0: 'cancer'},
            label2id={'cancer': 0},
            ignore_mismatched_sizes=True,
        )
        print("✅ Deformable DETR loaded with basic config")
        return model

def create_deformable_training_args(config, dataset_name, epoch_override=None):
    """Create optimized training arguments for Deformable DETR."""
    training_cfg = config.get('training', {})
    
    # Optimized settings for Deformable DETR
    batch_size = min(training_cfg.get('batch_size', 2), 2)  # Force small batch
    grad_accum = max(training_cfg.get('gradient_accumulation_steps', 16), 16)  # High accumulation
    learning_rate = training_cfg.get('learning_rate', 0.0002)
    # Use epoch_override if provided, otherwise use config
    epochs = epoch_override if epoch_override is not None else training_cfg.get('epochs', 50)
    
    print(f"🔧 Deformable DETR Training Config:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {batch_size * grad_accum}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {epochs}")
    
    # Create run name
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"DeformableDETR_{dataset_name}_{date_str}"
    
    # Use a metric that always exists for best model selection (e.g. 'eval_loss')
    metric_for_best_model = training_cfg.get('metric_for_best_model', 'eval_loss')
    if metric_for_best_model not in ['eval_loss', 'eval_runtime', 'eval_samples_per_second', 'eval_steps_per_second', 'epoch']:
        metric_for_best_model = 'eval_loss'
    
    # Training arguments optimized for Deformable DETR
    training_args = TrainingArguments(
        output_dir=training_cfg.get('output_dir', '../tmp'),
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=training_cfg.get('weight_decay', 0.0001),
        warmup_ratio=training_cfg.get('warmup_ratio', 0.1),
        lr_scheduler_type=training_cfg.get('lr_scheduler_type', 'linear'),
        
        # Evaluation and saving
        eval_strategy=training_cfg.get('evaluation_strategy', 'epoch'),
        eval_steps=training_cfg.get('eval_steps', 50) if 'eval_steps' in training_cfg else None,
        save_strategy=training_cfg.get('save_strategy', 'epoch'),
        save_steps=training_cfg.get('save_steps', 50) if 'save_steps' in training_cfg else None,
        save_total_limit=training_cfg.get('save_total_limit', 2),
        
        # Logging
        logging_strategy=training_cfg.get('logging_strategy', 'steps'),
        logging_steps=training_cfg.get('logging_steps', 25),
        
        # Model selection
        load_best_model_at_end=training_cfg.get('load_best_model_at_end', True),
        metric_for_best_model=metric_for_best_model,
        greater_is_better=training_cfg.get('greater_is_better', False),  # For loss, lower is better
        
        # Optimization for stability
        fp16=False,  # Disable FP16 for Deformable DETR stability
        gradient_checkpointing=False,
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        dataloader_num_workers=0,  # Disable multiprocessing
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        
        # Reporting
        report_to="wandb" if training_cfg.get('use_wandb', True) else [],
        disable_tqdm=False,
    )
    
    return training_args

def main(config_path, epoch=None, dataset=None):
    """Main training function for Deformable DETR."""
    print("🚀 Deformable DETR Training Pipeline")
    print("="*50)
    
    # Load config
    config = load_config(config_path)
    DATASET_NAME = dataset if dataset else config.get('dataset', {}).get('name', 'CSAW')
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'SenseTime/deformable-detr')
    MAX_SIZE = config.get('dataset', {}).get('max_size', 800)
    
    print(f"Dataset: {DATASET_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max size: {MAX_SIZE}")
    if epoch is not None:
        print(f"Epochs (override): {epoch}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load image processor
    image_processor = get_image_processor(MODEL_NAME, MAX_SIZE)
    
    # Create datasets - don't pass epoch to dataset, it's for training epochs
    print("\n📊 Loading datasets...")
    train_dataset = BreastCancerDataset(
        split='train',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        # Remove dataset_epoch=epoch since epoch is for training epochs, not dataset epochs
    )
    
    val_dataset = BreastCancerDataset(
        split='val',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        # Remove dataset_epoch=epoch
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    # Load model
    print("\n🤖 Loading model...")
    model = load_deformable_detr_model(MODEL_NAME, config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # Create training arguments - pass epoch override
    print("\n⚙️ Setting up training...")
    training_args = create_deformable_training_args(config, DATASET_NAME, epoch)
    
    # Create compute metrics function (not used in Trainer for custom mAP)
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)

    # Create trainer WITHOUT compute_metrics for custom evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        # compute_metrics=eval_compute_metrics_fn,  # REMOVE this line for custom eval
    )

    # Start training
    print("\n🏋️ Starting training...")
    print("="*50)
    
    try:
        trainer.train()
        
        # Save model
        date_str = datetime.datetime.now().strftime("%d%m%y")
        save_path = f'../deformable_detr_{DATASET_NAME}_{date_str}'
        trainer.save_model(save_path)
        print(f"\n✅ Model saved to: {save_path}")
        
        # Final evaluation on test set
        print("\n📊 Final evaluation on test set...")
        test_dataset = BreastCancerDataset(
            split='test',
            splits_dir=SPLITS_DIR,
            dataset_name=DATASET_NAME,
            image_processor=image_processor,
            model_type=get_model_type(MODEL_NAME),
        )
        
        # Use custom mAP calculation
        print("\n🔎 Calculating custom mAP metrics...")
        map_metrics = calculate_custom_map_metrics(
            model, test_dataset, image_processor, device
        )
        print(f"\n🎯 Custom Test mAP Results:")
        print("-" * 30)
        for key, value in map_metrics.items():
            print(f"{key}: {value}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Best mAP@50: {map_metrics.get('map_50', 'N/A')}")
        print(f"Model saved: {save_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deformable DETR for breast cancer detection")
    parser.add_argument('--config', type=str, default='configs/config_deformable_detr.yaml', 
                       help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, 
                       help='Number of epochs (overrides config)')
    parser.add_argument('--dataset', type=str, default=None, 
                       help='Dataset name (CSAW/DMID/DDSM, overrides config)')
    
    args = parser.parse_args()
    
    print("🔬 Deformable DETR for Breast Cancer Detection")
    print("=" * 50)
    
    main(args.config, args.epoch, args.dataset)
