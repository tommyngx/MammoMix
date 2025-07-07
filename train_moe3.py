import argparse
import os

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'FALSE'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import pickle
import numpy as np
import yaml
from PIL import Image
from functools import partial
from pathlib import Path
import datetime
from tqdm.auto import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
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

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class DataRouter(nn.Module):
    """
    Standalone router network that learns to classify which expert is best for each image.
    Phase 1: Train this separately to learn data routing patterns.
    """
    def __init__(self, num_experts=3, device='cuda'):
        super().__init__()
        self.num_experts = num_experts
        self.device = device
        self.expert_names = ['CSAW', 'DMID', 'DDSM']
        
        # Enhanced router architecture for better feature learning
        self.feature_extractor = nn.Sequential(
            # Initial feature extraction
            nn.Conv2d(3, 64, 7, stride=4, padding=3),  # 640->160
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 160->80
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 80->40
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),  # 40->8x8 = 64 features per channel
            nn.Flatten(),  # 256*64 = 16384 features
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_experts),
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Track routing performance
        self.expert_performance = {}
        self.routing_accuracy = 0.0
        
        print(f"DataRouter initialized with {num_experts} experts")
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Router parameters: {total_params / 1e6:.2f}M')
    
    def _initialize_weights(self):
        """Proper weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, pixel_values, expert_labels=None):
        """
        Forward pass for router training.
        expert_labels: ground truth expert indices for supervised learning
        """
        features = self.feature_extractor(pixel_values)
        logits = self.classifier(features)
        
        loss = None
        if expert_labels is not None:
            # Cross-entropy loss for expert classification
            loss = F.cross_entropy(logits, expert_labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'predictions': torch.argmax(logits, dim=1)
        }

class ImageRouterMoE(nn.Module):
    """
    MoE with pre-trained router from Phase 1.
    Phase 2: Use the trained router and fine-tune the entire system.
    """
    def __init__(self, expert_models, pretrained_router, device):
        super().__init__()
        self.experts = nn.ModuleList(expert_models)
        self.device = device
        self.expert_names = ['CSAW', 'DMID', 'DDSM']
        
        # Use pretrained router
        self.router = pretrained_router
        
        # Freeze expert models
        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        # Fine-tune router (optional)
        self.freeze_router = False  # Set to True to freeze router completely
        if self.freeze_router:
            for param in self.router.parameters():
                param.requires_grad = False
        
        # Statistics tracking
        self.routing_counts = torch.zeros(3, device=device)
        self.total_routed = 0
        
        print(f"MoE initialized with pretrained router")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params / 1e6:.2f}M')
        print(f'Trainable parameters: {trainable_params / 1e6:.2f}M')

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for compatibility."""
        pass
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for compatibility."""
        pass
    
    def forward(self, pixel_values, labels=None, return_routing=False):
        """Forward pass with pretrained routing."""
        batch_size = pixel_values.shape[0]
        
        # Get routing decisions from pretrained router
        router_output = self.router(pixel_values)
        routing_logits = router_output['logits']
        routing_probs = F.softmax(routing_logits, dim=1)
        expert_choices = torch.argmax(routing_probs, dim=1)
        
        # Update statistics
        if not self.training:
            for choice in expert_choices:
                self.routing_counts[choice] += 1
            self.total_routed += batch_size
        
        # Group samples by expert choice
        expert_groups = {}
        for i in range(batch_size):
            expert_idx = expert_choices[i].item()
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(i)
        
        # Print routing distribution periodically
        if not self.training and self.total_routed % 500 == 0:
            usage_pct = (self.routing_counts / self.total_routed * 100).cpu().numpy()
            print(f"Routing: CSAW={usage_pct[0]:.1f}%, DMID={usage_pct[1]:.1f}%, DDSM={usage_pct[2]:.1f}%")
        
        # Get expert outputs
        batch_outputs = {}
        total_detection_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for expert_idx, sample_indices in expert_groups.items():
            expert_pixel_values = pixel_values[sample_indices]
            expert_labels = None
            if labels is not None:
                expert_labels = [labels[i] for i in sample_indices]
            
            # Get expert output
            with torch.no_grad():
                expert_output = self.experts[expert_idx](expert_pixel_values, labels=expert_labels)
                expert_output = self._fix_expert_output_dimensions(expert_output)
            
            batch_outputs[expert_idx] = {
                'output': expert_output,
                'indices': sample_indices
            }
            
            # Accumulate detection loss
            if hasattr(expert_output, 'loss') and expert_output.loss is not None:
                weight = len(sample_indices) / batch_size
                total_detection_loss = total_detection_loss + expert_output.loss * weight
        
        # Reconstruct batch outputs
        if not batch_outputs:
            raise RuntimeError("No expert outputs generated")
        
        first_expert_idx = list(batch_outputs.keys())[0]
        reference_output = batch_outputs[first_expert_idx]['output']
        
        # Initialize batch tensors
        num_queries = reference_output.logits.shape[1]
        num_classes = reference_output.logits.shape[2]
        
        batch_logits = torch.zeros(batch_size, num_queries, num_classes, 
                                 device=pixel_values.device, dtype=reference_output.logits.dtype)
        batch_pred_boxes = torch.zeros(batch_size, num_queries, 4, 
                                     device=pixel_values.device, dtype=reference_output.pred_boxes.dtype)
        batch_last_hidden_state = None
        
        if hasattr(reference_output, 'last_hidden_state') and reference_output.last_hidden_state is not None:
            batch_last_hidden_state = torch.zeros(batch_size, reference_output.last_hidden_state.shape[1], 
                                                 reference_output.last_hidden_state.shape[2], 
                                                 device=pixel_values.device, dtype=reference_output.last_hidden_state.dtype)
        
        # Fill batch tensors
        for expert_idx, group_data in batch_outputs.items():
            expert_output = group_data['output']
            sample_indices = group_data['indices']
            
            batch_logits[sample_indices] = expert_output.logits
            batch_pred_boxes[sample_indices] = expert_output.pred_boxes
            
            if batch_last_hidden_state is not None and hasattr(expert_output, 'last_hidden_state'):
                batch_last_hidden_state[sample_indices] = expert_output.last_hidden_state
        
        # Optional router fine-tuning loss
        router_loss = torch.tensor(0.0, device=self.device)
        if not self.freeze_router and labels is not None and self.training:
            # Small router fine-tuning loss
            router_loss = 0.1 * router_output.get('loss', torch.tensor(0.0, device=self.device))
        
        total_loss = total_detection_loss + router_loss
        
        # Safety check
        batch_pred_boxes = batch_pred_boxes[..., :4].contiguous()
        
        # Create output
        from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput
        
        combined_output = YolosObjectDetectionOutput(
            loss=total_loss,
            logits=batch_logits,
            pred_boxes=batch_pred_boxes,
            last_hidden_state=batch_last_hidden_state
        )
        
        if return_routing:
            routing_info = {
                'probs': routing_probs,
                'choices': expert_choices,
                'groups': expert_groups,
                'stats': self.get_routing_stats()
            }
            return combined_output, routing_info
        else:
            return combined_output
    
    def _fix_expert_output_dimensions(self, expert_output):
        """Fix expert output dimensions."""
        if expert_output.pred_boxes.shape[-1] != 4:
            fixed_pred_boxes = expert_output.pred_boxes[..., :4].contiguous()
            fixed_output = type(expert_output)(
                loss=expert_output.loss,
                logits=expert_output.logits,
                pred_boxes=fixed_pred_boxes,
                last_hidden_state=expert_output.last_hidden_state if hasattr(expert_output, 'last_hidden_state') else None
            )
            return fixed_output
        return expert_output
    
    def get_routing_stats(self):
        """Get routing statistics."""
        if self.total_routed == 0:
            return {name: 0.0 for name in self.expert_names}
        
        stats = {}
        usage = (self.routing_counts / self.total_routed).cpu().numpy()
        for i, name in enumerate(self.expert_names):
            stats[name] = float(usage[i])
        return stats

def load_expert_models(weight_dir, device):
    """Load the 3 expert models."""
    expert_names = ['yolos_CSAW', 'yolos_DMID', 'yolos_DDSM']
    expert_paths = [os.path.join(weight_dir, name) for name in expert_names]
    
    models = []
    processors = []
    
    for path in expert_paths:
        print(f"Loading expert from: {path}")
        processor = AutoImageProcessor.from_pretrained(path)
        model = AutoModelForObjectDetection.from_pretrained(
            path,
            id2label={0: 'cancer'},
            label2id={'cancer': 0},
            auxiliary_loss=False,
        )
        model = model.to(device)
        model.eval()
        models.append(model)
        processors.append(processor)
    
    return models, processors

def evaluate_experts_on_data(expert_models, dataset, image_processor, device, sample_size=100):
    """
    Evaluate all experts on a sample of data to create routing labels.
    Returns expert performance for each sample.
    """
    print(f"Evaluating experts on {min(sample_size, len(dataset))} samples...")
    
    # Sample subset for evaluation
    if len(dataset) > sample_size:
        indices = torch.randperm(len(dataset))[:sample_size]
        eval_dataset = Subset(dataset, indices)
    else:
        eval_dataset = dataset
    
    dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    expert_performances = []  # List of (sample_idx, expert_losses)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating experts")):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            batch_losses = []
            for expert_idx, expert in enumerate(expert_models):
                try:
                    output = expert(pixel_values, labels=labels)
                    if hasattr(output, 'loss') and output.loss is not None:
                        batch_losses.append(output.loss.item())
                    else:
                        batch_losses.append(float('inf'))
                except Exception as e:
                    print(f"Expert {expert_idx} failed on batch {batch_idx}: {e}")
                    batch_losses.append(float('inf'))
            
            # Store best expert for each sample in batch
            for i in range(len(labels)):
                sample_losses = [loss for loss in batch_losses]
                best_expert = np.argmin(sample_losses)
                expert_performances.append({
                    'sample_idx': batch_idx * 8 + i,
                    'expert_losses': sample_losses,
                    'best_expert': best_expert
                })
    
    return expert_performances

def create_routing_dataset_with_labels(config, image_processor, split, dataset_name, expert_performances):
    """Create dataset with expert routing labels."""
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    base_dataset = BreastCancerDataset(
        split=split,
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
    )
    
    # Create routing labels
    routing_labels = []
    for i in range(len(base_dataset)):
        if i < len(expert_performances):
            routing_labels.append(expert_performances[i]['best_expert'])
        else:
            # Default to expert 0 for unseen samples
            routing_labels.append(0)
    
    return base_dataset, routing_labels

def train_phase1_router(config, expert_models, device, dataset_name, epoch=None, weight_dir=None):
    """
    Phase 1: Train router separately to learn data routing patterns.
    """
    print("\n" + "="*50)
    print("PHASE 1: Training Data Router")
    print("="*50)
    
    # Load datasets
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    # Get image processor
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
    # Create base datasets
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
    
    # Evaluate experts to create routing labels
    print("Creating routing labels for training data...")
    train_performances = evaluate_experts_on_data(expert_models, train_dataset, image_processor, device, sample_size=500)
    
    print("Creating routing labels for validation data...")
    val_performances = evaluate_experts_on_data(expert_models, val_dataset, image_processor, device, sample_size=200)
    
    # Create router datasets
    train_routing_dataset, train_routing_labels = create_routing_dataset_with_labels(
        config, image_processor, 'train', dataset_name, train_performances
    )
    val_routing_dataset, val_routing_labels = create_routing_dataset_with_labels(
        config, image_processor, 'val', dataset_name, val_performances
    )
    
    print(f"Router training data: {len(train_routing_dataset)} samples")
    print(f"Router validation data: {len(val_routing_dataset)} samples")
    
    # Analyze expert distribution
    train_expert_counts = np.bincount(train_routing_labels[:len(train_performances)], minlength=3)
    print(f"Training expert distribution: CSAW={train_expert_counts[0]}, DMID={train_expert_counts[1]}, DDSM={train_expert_counts[2]}")
    
    # Create router model
    router = DataRouter(num_experts=3, device=device).to(device)
    
    # Custom training loop for router
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Training
    router.train()
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0
    
    for epoch_idx in range(10):  # Train for 10 epochs
        print(f"\nRouter Epoch {epoch_idx + 1}/10")
        
        # Training
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loader = DataLoader(train_routing_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training router")):
            pixel_values = batch['pixel_values'].to(device)
            
            # Get routing labels for this batch
            batch_size = pixel_values.shape[0]
            expert_labels = []
            for i in range(batch_size):
                sample_idx = batch_idx * 16 + i
                if sample_idx < len(train_routing_labels):
                    expert_labels.append(train_routing_labels[sample_idx])
                else:
                    expert_labels.append(0)  # Default
            
            expert_labels = torch.tensor(expert_labels, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            router_output = router(pixel_values, expert_labels)
            loss = router_output['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predictions = router_output['predictions']
            train_correct += (predictions == expert_labels).sum().item()
            train_total += expert_labels.size(0)
        
        scheduler.step()
        
        # Validation
        router.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_loader = DataLoader(val_routing_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating router")):
                pixel_values = batch['pixel_values'].to(device)
                
                # Get routing labels for this batch
                batch_size = pixel_values.shape[0]
                expert_labels = []
                for i in range(batch_size):
                    sample_idx = batch_idx * 16 + i
                    if sample_idx < len(val_routing_labels):
                        expert_labels.append(val_routing_labels[sample_idx])
                    else:
                        expert_labels.append(0)
                
                expert_labels = torch.tensor(expert_labels, device=device)
                
                router_output = router(pixel_values, expert_labels)
                loss = router_output['loss']
                
                val_loss += loss.item()
                predictions = router_output['predictions']
                val_correct += (predictions == expert_labels).sum().item()
                val_total += expert_labels.size(0)
        
        # Calculate accuracies
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save router
            router_save_path = os.path.join(weight_dir, f'router_{dataset_name}')
            os.makedirs(router_save_path, exist_ok=True)
            torch.save(router.state_dict(), os.path.join(router_save_path, 'router.pth'))
            print(f"Best router saved to {router_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        router.train()
    
    # Load best router
    router_load_path = os.path.join(weight_dir, f'router_{dataset_name}', 'router.pth')
    router.load_state_dict(torch.load(router_load_path, map_location=device))
    router.eval()
    
    print(f"\nPhase 1 Complete! Best router accuracy: {best_val_acc:.4f}")
    print(f"Router saved to: {router_load_path}")
    
    return router

def train_phase2_moe(config, expert_models, pretrained_router, device, dataset_name, epoch=None, weight_dir=None):
    """
    Phase 2: Train MoE with pretrained router.
    """
    print("\n" + "="*50)
    print("PHASE 2: Training MoE with Pretrained Router")
    print("="*50)
    
    # Create MoE model
    model = ImageRouterMoE(expert_models, pretrained_router, device).to(device)
    
    # Load datasets
    SPLITS_DIR = Path(config.get('dataset', {}).get('splits_dir', '/content/dataset'))
    MODEL_NAME = config.get('model', {}).get('model_name', 'hustvl/yolos-base')
    
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    
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
    
    print(f'Train dataset: {len(train_dataset)} samples')
    print(f'Val dataset: {len(val_dataset)} samples')
    
    # Training configuration
    output_dir = os.path.join(weight_dir, f'moe3_{dataset_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    date_str = datetime.datetime.now().strftime("%d%m%y")
    run_name = f"MoE3_{dataset_name}_{date_str}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=epoch if epoch else 5,  # Shorter training since router is pretrained
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
        weight_decay=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model='eval_map_50',
        greater_is_better=True,
        fp16=False,
        dataloader_num_workers=0,
        gradient_accumulation_steps=2,
        remove_unused_columns=False,
        save_safetensors=True,
    )
    
    eval_compute_metrics_fn = get_eval_compute_metrics_fn(image_processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    print("Training Phase 2 MoE...")
    trainer.train()
    
    # Test evaluation
    print("\n=== Evaluating on test dataset ===")
    test_dataset = BreastCancerDataset(
        split='test',
        splits_dir=SPLITS_DIR,
        dataset_name=dataset_name,
        image_processor=image_processor,
        model_type=get_model_type(MODEL_NAME),
        dataset_epoch=epoch
    )
    
    test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
    
    print("\n=== Test Results ===")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Print final routing statistics
    print("\n=== Final Routing Statistics ===")
    final_stats = model.get_routing_stats()
    for expert_name, usage in final_stats.items():
        print(f"{expert_name}: {usage*100:.1f}%")
    
    return model, test_results

def main(config_path, epoch=None, dataset=None, weight_dir=None, phase=None):
    """
    Main function with 2-phase training:
    Phase 1: Train router to learn data routing patterns
    Phase 2: Use pretrained router in MoE training
    """
    config = load_config(config_path)
    
    # Configuration
    DATASET_NAME = dataset if dataset is not None else config.get('dataset', {}).get('name', 'CSAW')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if weight_dir is not None:
        expert_weights_dir = weight_dir
    else:
        expert_weights_dir = config.get('moe', {}).get('expert_weights_dir', '/content/Weights')
    
    if not os.path.exists(expert_weights_dir):
        raise ValueError(f"Expert weights directory not found: {expert_weights_dir}")
    
    print(f"Using device: {device}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Expert weights: {expert_weights_dir}")
    
    # Load expert models
    expert_models, expert_processors = load_expert_models(expert_weights_dir, device)
    
    if phase == "1" or phase is None:
        # Phase 1: Train router
        router = train_phase1_router(config, expert_models, device, DATASET_NAME, epoch, expert_weights_dir)
        
        if phase == "1":
            print("Phase 1 completed. Run with --phase 2 to continue to MoE training.")
            return
    
    if phase == "2" or phase is None:
        # Phase 2: Load router and train MoE
        router_path = os.path.join(expert_weights_dir, f'router_{DATASET_NAME}', 'router.pth')
        
        if not os.path.exists(router_path):
            raise FileNotFoundError(f"Router not found at {router_path}. Run Phase 1 first.")
        
        # Load pretrained router
        router = DataRouter(num_experts=3, device=device).to(device)
        router.load_state_dict(torch.load(router_path, map_location=device))
        router.eval()
        print(f"Loaded pretrained router from: {router_path}")
        
        # Train MoE
        model, test_results = train_phase2_moe(config, expert_models, router, device, DATASET_NAME, epoch, expert_weights_dir)
    
    print("\n" + "="*50)
    print("2-PHASE TRAINING COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config yaml')
    parser.add_argument('--epoch', type=int, default=None, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--weight_dir', type=str, default=None, help='Expert weights directory')
    parser.add_argument('--phase', type=str, choices=['1', '2'], default=None, help='Training phase (1: router only, 2: MoE only, None: both)')
    args = parser.parse_args()

    main(args.config, args.epoch, args.dataset, args.weight_dir, args.phase)
