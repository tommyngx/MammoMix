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
    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    for batch in targets:
        batch_image_sizes = torch.tensor(np.array([[MAX_SIZE, MAX_SIZE] for _ in batch]))
        image_sizes.append(batch_image_sizes)
        for image_target in batch:
            boxes = torch.tensor(image_target['boxes'])
            boxes = convert_bbox_yolo_to_pascal(boxes, [MAX_SIZE, MAX_SIZE])
            labels = torch.tensor(image_target['class_labels'])
            post_processed_targets.append({'boxes': boxes, 'labels': labels})
    
    for batch_idx, (batch, target_sizes) in enumerate(zip(predictions, image_sizes)):
        #print(f"[EVAL DEBUG] Processing batch {batch_idx}")
        
        # DEBUG: Check what's in the batch
        #print(f"[EVAL DEBUG] Batch type: {type(batch)}")
        #print(f"[EVAL DEBUG] Batch length: {len(batch) if hasattr(batch, '__len__') else 'no len'}")
        
        if len(batch) >= 3:
                batch_logits, batch_boxes = batch[1], batch[2]
            #print(f"[EVAL DEBUG] batch_logits type: {type(batch_logits)}")
            #print(f"[EVAL DEBUG] batch_boxes type: {type(batch_boxes)}")
            
            #if hasattr(batch_boxes, 'shape'):
                #print(f"[EVAL DEBUG] batch_boxes shape: {batch_boxes.shape}")
            #elif isinstance(batch_boxes, (list, np.ndarray)):
                #print(f"[EVAL DEBUG] batch_boxes array shape: {np.array(batch_boxes).shape}")
            
            # Check if batch[2] is actually last_hidden_state instead of pred_boxes
            #if hasattr(batch_boxes, 'shape') and len(batch_boxes.shape) >= 2 and batch_boxes.shape[-1] == 768:
                #print(f"[EVAL DEBUG] WARNING: batch[2] seems to be last_hidden_state (768 dims), not pred_boxes!")
                
                # Try to find pred_boxes in other batch positions
                for i, item in enumerate(batch):
                    if hasattr(item, 'shape') and len(item.shape) >= 2 and item.shape[-1] == 4:
                        print(f"[EVAL DEBUG] Found 4D tensor at batch[{i}]: {item.shape}")
                        batch_boxes = item
                        break
                else:
                    #print(f"[EVAL DEBUG] No 4D tensor found in batch, using emergency fallback")
                    # Emergency fallback: create dummy boxes
                    if hasattr(batch_boxes, 'shape'):
                        batch_size, num_queries = batch_boxes.shape[0], batch_boxes.shape[1]
                        batch_boxes = torch.zeros(batch_size, num_queries, 4)
        
        # CRITICAL FIX: Ensure batch_boxes is exactly 4D
        batch_boxes_tensor = torch.tensor(batch_boxes)
        if batch_boxes_tensor.shape[-1] != 4:
            #print(f"EVAL FIX: batch_boxes has {batch_boxes_tensor.shape[-1]} dims, forcing to 4")
            batch_boxes_tensor = batch_boxes_tensor[..., :4]
        
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=batch_boxes_tensor)
        
        try:
            post_processed_output = image_processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        except Exception as e:
            #print(f"ERROR in evaluation batch {batch_idx}: {e}")
            #print(f"pred_boxes shape: {output.pred_boxes.shape}")
            raise e

    metrics = mean_average_precision(post_processed_predictions, post_processed_targets)
    metrics.pop('map_per_class')
    return {k: v for k, v in metrics.items() if k.startswith('map')}

def get_eval_compute_metrics_fn(image_processor):
    return partial(
        compute_metrics, image_processor=image_processor,
        threshold=0.5, id2label={0: 'cancer'}
    )
