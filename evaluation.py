import os
import torch
import pickle
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET

from PIL import Image
from functools import partial
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from transformers.image_transforms import center_to_corners_format
from torchmetrics.functional.detection.map import mean_average_precision

# You need to provide or import these utility functions:
# from your_metrics_lib import mean_average_precision, center_to_corners_format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    '''
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.
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
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None, MAX_SIZE = 640):
    '''
    Compute mean average mAP, mAR and their variants for the object detection task.
    '''
    print(f"[DEBUG EVAL] Starting compute_metrics with threshold={threshold}")
    
    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    print(f"[DEBUG EVAL] Number of prediction batches: {len(predictions)}")
    print(f"[DEBUG EVAL] Number of target batches: {len(targets)}")

    for batch in targets:
        batch_image_sizes = torch.tensor(np.array([[MAX_SIZE, MAX_SIZE] for _ in batch]))
        image_sizes.append(batch_image_sizes)
        for image_target in batch:
            boxes = torch.tensor(image_target['boxes'])
            boxes = convert_bbox_yolo_to_pascal(boxes, [MAX_SIZE, MAX_SIZE])
            labels = torch.tensor(image_target['class_labels'])
            post_processed_targets.append({'boxes': boxes, 'labels': labels})

    print(f"[DEBUG EVAL] Processing {len(predictions)} prediction batches")
    
    for batch_idx, (batch, target_sizes) in enumerate(zip(predictions, image_sizes)):
        print(f"[DEBUG EVAL] Processing batch {batch_idx}")
        batch_logits, batch_boxes = batch[1], batch[2]
        
        print(f"[DEBUG EVAL] Batch {batch_idx} - batch_boxes type: {type(batch_boxes)}")
        print(f"[DEBUG EVAL] Batch {batch_idx} - batch_boxes shape before tensor conversion: {np.array(batch_boxes).shape if isinstance(batch_boxes, (list, np.ndarray)) else 'not array-like'}")
        
        batch_boxes_tensor = torch.tensor(batch_boxes)
        print(f"[DEBUG EVAL] Batch {batch_idx} - batch_boxes_tensor shape: {batch_boxes_tensor.shape}")
        
        # CRITICAL DEBUG: Check the actual last dimension
        if len(batch_boxes_tensor.shape) >= 1:
            print(f"[DEBUG EVAL] Batch {batch_idx} - Last dimension size: {batch_boxes_tensor.shape[-1]}")
            if batch_boxes_tensor.shape[-1] != 4:
                print(f"[DEBUG EVAL] ERROR DETECTED: batch_boxes has {batch_boxes_tensor.shape[-1]} dimensions instead of 4!")
                print(f"[DEBUG EVAL] Batch {batch_idx} - Full tensor shape: {batch_boxes_tensor.shape}")
                print(f"[DEBUG EVAL] Batch {batch_idx} - Tensor content sample: {batch_boxes_tensor.flatten()[:20] if batch_boxes_tensor.numel() > 0 else 'empty'}")
        
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=batch_boxes_tensor)
        
        print(f"[DEBUG EVAL] About to call post_process_object_detection with pred_boxes shape: {output.pred_boxes.shape}")
        
        try:
            post_processed_output = image_processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
            print(f"[DEBUG EVAL] Batch {batch_idx} processed successfully")
        except Exception as e:
            print(f"[DEBUG EVAL] ERROR in batch {batch_idx}: {e}")
            print(f"[DEBUG EVAL] pred_boxes shape that caused error: {output.pred_boxes.shape}")
            print(f"[DEBUG EVAL] pred_boxes content: {output.pred_boxes}")
            raise e

    print(f"[DEBUG EVAL] All batches processed, computing mAP...")
    metrics = mean_average_precision(post_processed_predictions, post_processed_targets)
    metrics.pop('map_per_class')
    return {k: v for k, v in metrics.items() if k.startswith('map')}

def get_eval_compute_metrics_fn(image_processor):
    return partial(
        compute_metrics, image_processor=image_processor,
        threshold=0.5, id2label={0: 'cancer'}
    )
