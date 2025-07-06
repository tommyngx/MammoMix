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

    print(f"DEBUG: targets type: {type(targets)}")
    print(f"DEBUG: targets length: {len(targets)}")
    if len(targets) > 0:
        print(f"DEBUG: first target type: {type(targets[0])}")
        print(f"DEBUG: first target content: {targets[0]}")

    for batch in targets:
        print(f"DEBUG: processing batch type: {type(batch)}")
        print(f"DEBUG: batch content: {batch}")
        #batch_image_sizes = torch.tensor(np.array([x['size'] for x in batch]))
        batch_image_sizes = torch.tensor(np.array([[MAX_SIZE, MAX_SIZE] for _ in batch]))
        image_sizes.append(batch_image_sizes)
        for image_target in batch:
            print(f"DEBUG: image_target type: {type(image_target)}")
            print(f"DEBUG: image_target content: {image_target}")
            print(f"DEBUG: Attempting to access image_target['boxes']...")
            boxes = torch.tensor(image_target['boxes'])
            print(f"DEBUG: boxes: {boxes}")
            #boxes = convert_bbox_yolo_to_pascal(boxes, image_target['size'])
            boxes = convert_bbox_yolo_to_pascal(boxes, [MAX_SIZE, MAX_SIZE])
            labels = torch.tensor(image_target['class_labels'])
            post_processed_targets.append({'boxes': boxes, 'labels': labels})

    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    metrics = mean_average_precision(post_processed_predictions, post_processed_targets)
    metrics.pop('map_per_class')
    return {k: v for k, v in metrics.items() if k.startswith('map')}

def get_eval_compute_metrics_fn(image_processor):
    return partial(
        compute_metrics, image_processor=image_processor,
        threshold=0.5, id2label={0: 'cancer'}
    )
