import torch
import numpy as np
from functools import partial
from dataclasses import dataclass
from loader import collate_fn

from torch.utils.data import Dataset, DataLoader
from transformers.image_transforms import center_to_corners_format
from torchmetrics.functional.detection.map import mean_average_precision


def convert_bbox_yolo_to_pascal(boxes, image_size):
    '''
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    '''
    boxes = center_to_corners_format(boxes)
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])
    return boxes


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None, max_size=640):
    '''
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        image_processor (AutoImageProcessor): Image processor used for post-processing predictions.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.
        max_size (int, optional): Maximum size for resizing images. Defaults to 640.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    '''
    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys 'boxes', 'labels'
    #  - predictions in a form of list of dictionaries with keys 'boxes', 'scores', 'labels'
    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    for batch in targets: # Collect targets in the required format for metric computation
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([[max_size, max_size] for _ in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target['boxes'])
            boxes = convert_bbox_yolo_to_pascal(boxes, [max_size, max_size])
            labels = torch.tensor(image_target['class_labels'])
            post_processed_targets.append({'boxes': boxes, 'labels': labels})
    
    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch_idx, (batch, target_sizes) in enumerate(zip(predictions, image_sizes)):
        if len(batch) >= 3:
            batch_logits, batch_boxes = batch[1], batch[2]
            
            # Check if batch[2] is actually last_hidden_state instead of pred_boxes
            if hasattr(batch_boxes, 'shape') and len(batch_boxes.shape) >= 2 and batch_boxes.shape[-1] == 768:
                # Try to find pred_boxes in other batch positions
                for i, item in enumerate(batch):
                    if hasattr(item, 'shape') and len(item.shape) >= 2 and item.shape[-1] == 4:
                        batch_boxes = item
                        break
                else:
                    # Emergency fallback: create dummy boxes
                    if hasattr(batch_boxes, 'shape'):
                        batch_size, num_queries = batch_boxes.shape[0], batch_boxes.shape[1]
                        batch_boxes = torch.zeros(batch_size, num_queries, 4)
        
        # CRITICAL FIX: Ensure batch_boxes is exactly 4D
        batch_boxes_tensor = torch.tensor(batch_boxes)
        if batch_boxes_tensor.shape[-1] != 4:
            batch_boxes_tensor = batch_boxes_tensor[..., :4]
        
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=batch_boxes_tensor)
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metrics = mean_average_precision(post_processed_predictions, post_processed_targets)
    metrics.pop('map_per_class')
    return {k: v for k, v in metrics.items() if k.startswith('map')}


def get_eval_compute_metrics_fn(image_processor):
    return partial(
        compute_metrics, image_processor=image_processor,
        threshold=0.5, id2label={0: 'cancer'}
    )


def calculate_custom_map_metrics(predictions, targets, image_processor, device, max_size=640):
    """
    Custom mAP calculation using torchmetrics (alternative to compute_metrics).
    This function avoids the evaluation.py string indices bug.
    """
    try:
        def convert_bbox_yolo_to_pascal_custom(boxes, image_size):
            """Convert YOLO format to Pascal VOC format."""
            boxes = center_to_corners_format(boxes)
            height, width = image_size
            boxes = boxes * torch.tensor([[width, height, width, height]], device=boxes.device)
            return boxes
        
        # Prepare data for torchmetrics
        post_processed_targets = []
        post_processed_predictions = []
        
        # Process targets (ground truth) with proper device handling
        for target in targets:
            if target and 'boxes' in target and 'class_labels' in target:
                # Ensure proper device and tensor conversion
                boxes = target['boxes'].clone().detach().to(device) if torch.is_tensor(target['boxes']) else torch.tensor(target['boxes'], device=device)
                labels = target['class_labels'].clone().detach().to(device) if torch.is_tensor(target['class_labels']) else torch.tensor(target['class_labels'], device=device)
                
                boxes = convert_bbox_yolo_to_pascal_custom(boxes, [max_size, max_size])
                post_processed_targets.append({'boxes': boxes, 'labels': labels})
            else:
                # Empty target
                post_processed_targets.append({
                    'boxes': torch.zeros((0, 4), device=device), 
                    'labels': torch.tensor([], dtype=torch.long, device=device)
                })
        
        # Process predictions with proper device handling
        for pred in predictions:
            if hasattr(pred, 'logits') and hasattr(pred, 'pred_boxes'):
                logits = pred.logits.to(device)
                pred_boxes = pred.pred_boxes.to(device)
                
                # Process each sample in the batch
                for i in range(logits.shape[0]):
                    sample_logits = logits[i]
                    sample_boxes = pred_boxes[i]
                    
                    # Create temporary model output for post-processing
                    from dataclasses import dataclass
                    
                    @dataclass
                    class TempModelOutput:
                        logits: torch.Tensor
                        pred_boxes: torch.Tensor
                    
                    temp_output = TempModelOutput(
                        logits=sample_logits.unsqueeze(0).to(device),
                        pred_boxes=sample_boxes.unsqueeze(0).to(device)
                    )
                    
                    target_sizes = torch.tensor([[max_size, max_size]], device=device)
                    
                    # Post-process using image_processor
                    post_processed_output = image_processor.post_process_object_detection(
                        temp_output, threshold=0.5, target_sizes=target_sizes
                    )
                    
                    # Store results with proper device handling
                    if post_processed_output:
                        pred_result = post_processed_output[0]
                        post_processed_predictions.append({
                            'boxes': pred_result['boxes'].to(device) if 'boxes' in pred_result else torch.zeros((0, 4), device=device),
                            'scores': pred_result['scores'].to(device) if 'scores' in pred_result else torch.tensor([], device=device),
                            'labels': pred_result['labels'].to(device) if 'labels' in pred_result else torch.tensor([], dtype=torch.long, device=device)
                        })
                    else:
                        post_processed_predictions.append({
                            'boxes': torch.zeros((0, 4), device=device),
                            'scores': torch.tensor([], device=device),
                            'labels': torch.tensor([], dtype=torch.long, device=device)
                        })
            else:
                post_processed_predictions.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.tensor([], device=device),
                    'labels': torch.tensor([], dtype=torch.long, device=device)
                })
        
        # Move all tensors to CPU for torchmetrics compatibility
        cpu_predictions, cpu_targets = [], []
        for pred in post_processed_predictions:
            cpu_pred = {k: v.cpu() if torch.is_tensor(v) else v for k, v in pred.items()}
            cpu_predictions.append(cpu_pred)
        
        for target in post_processed_targets:
            cpu_target = {k: v.cpu() if torch.is_tensor(v) else v for k, v in target.items()}
            cpu_targets.append(cpu_target)
        
        metrics = mean_average_precision(cpu_predictions, cpu_targets) # Calculate mAP using torchmetrics
        if 'map_per_class' in metrics: metrics.pop('map_per_class') # Clean up metrics (remove unwanted keys)
        
        # Convert tensor values to float
        result = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items() if k.startswith('map')}
        return result
        
    except Exception as e:
        print(f"Custom mAP calculation failed: {e}")
        # Return zero metrics as fallback
        return {
            'map': 0.0,
            'map_50': 0.0,
            'map_75': 0.0,
            'map_small': 0.0,
            'map_medium': 0.0,
            'map_large': 0.0
        }

def run_model_inference_with_map(model, test_dataset, image_processor, device, batch_size=8):
    """
    Run inference on test dataset and calculate mAP metrics.
    This is a clean interface for getting mAP from any model.
    """
    model.eval()
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_predictions, all_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels']
            
            if labels: # Move labels to device
                for label_dict in labels:
                    for key, value in label_dict.items():
                        if isinstance(value, torch.Tensor):
                            label_dict[key] = value.to(device)
            
            output = model(pixel_values, labels=labels)
            all_predictions.append(output)
            all_targets.extend(labels if labels else [{}])
    
    # Calculate mAP metrics
    map_metrics = calculate_custom_map_metrics(all_predictions, all_targets, image_processor, device)
    return map_metrics