import torch
import pickle
import numpy as np

from collections import defaultdict
from tqdm.notebook import tqdm
from loader import BreastCancerDataset, collate_fn
from metrics import convert_bbox_yolo_to_pascal

from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoImageProcessor, AutoModelForObjectDetection


SPLITS_DIR = '/tmp/splits'
CONFIGS = [
    {
        'dataset_name': 'CSAW',
        'saved_dir': './Weights/yolos_CSAW',
        'model': None,
        'image_processor': None,
        'calibrator': None,
        'calibrator_dataset': None
    },
    {
        'dataset_name': 'DDSM',
        'saved_dir': './Weights/yolos_DDSM',
        'model': None,
        'image_processor': None,
        'calibrator': None,
        'calibrator_dataset': None
    },
    {
        'dataset_name': 'DMID',
        'saved_dir': './Weights/yolos_DMID',
        'model': None,
        'image_processor': None,
        'calibrator': None,
        'calibrator_dataset': None
    }
]
DATASET_NAMES = [config['dataset_name'] for config in CONFIGS]
MAX_SIZE = 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_extractor.fc = torch.nn.Identity() # Remove classification head
feature_extractor.eval()
feature_extractor.to(device)


def build_calibrate_dataset(config, image_embeddings, confidences, ious, dataset_name, batch_size=8, split='train'):
    calibrate_dataset = BreastCancerDataset(split=split, splits_dir=SPLITS_DIR, dataset_name=dataset_name, image_processor=config['image_processor'])
    calibrate_loader = DataLoader(calibrate_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    print(f'{dataset_name} {split} loader: {len(calibrate_dataset)} samples')

    for batch in calibrate_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        with torch.no_grad():
            outputs = config['model'](pixel_values=pixel_values)
            embeddings = feature_extractor(pixel_values)
            # embeddings = outputs.last_hidden_state[:, 0, :]

        target_sizes = torch.stack([label['size'] for label in labels]).to(device)
        results = config['image_processor'].post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.5
        )
        for i, (result, label) in enumerate(zip(results, labels)):
            pred_boxes = result['boxes'].cpu()
            pred_scores = result['scores'].cpu()
            pred_labels = result['labels'].cpu()

            if len(pred_boxes) <= 0: continue
            cancer_mask = pred_labels == 0 # Filter predictions for class 'cancer' (class_id=0)
            num_cancers = len(pred_scores[cancer_mask])

            if dataset_name == config['dataset_name']: # Not out-of-domain dataset
                gt_boxes = convert_bbox_yolo_to_pascal(label['boxes'], label['size'])
                if len(gt_boxes) <= 0: continue
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                max_ious, _ = iou_matrix.max(dim=1)
                ious.extend(max_ious[cancer_mask].numpy())
            else: ious.extend([0] * num_cancers)

            confidences.extend(pred_scores[cancer_mask])
            for _ in range(num_cancers): image_embeddings.append(embeddings[i].cpu().numpy())


def combine_datasets(config, dataset_names, split='train'):
    image_embeddings, confidences, ious = [], [], []
    for dataset_name in tqdm(dataset_names):
        build_calibrate_dataset(config, image_embeddings, confidences, ious, dataset_name, split=split)

    confidences = np.array(confidences).reshape(-1, 1)
    inputs = np.concatenate([image_embeddings, confidences], axis=-1)
    print(f'Combined inputs shape: {inputs.shape}\n') # [num_samples, 513]
    return inputs, ious


def soft_nms(boxes, scores, sigma_nms=0.1, iou_nms=0.65, score_thresh=0, method='gaussian'):
    '''
    Soft-NMS: suppresses overlapping boxes softly by reducing their scores.

    Args:
        boxes (Tensor): [N, 4] in (x1, y1, x2, y2) format
        scores (Tensor): [N] confidence scores
        sigma_nms (float): Sigma for Gaussian method
        iou_nms (float): IoU threshold for linear method
        score_thresh (float): Remove boxes with score < score_thresh
        method (str): 'linear' or 'gaussian' (default: 'gaussian')

    Returns:
        keep_boxes (Tensor): [M, 4] boxes after suppression
        keep_scores (Tensor): [M] refined scores
    '''
    keep_boxes, keep_scores = [], []
    boxes, scores = boxes.clone(), scores.clone()

    while boxes.numel() > 0:
        max_idx = torch.argmax(scores)
        max_box = boxes[max_idx].clone()
        max_score = scores[max_idx].clone()

        keep_boxes.append(max_box)
        keep_scores.append(max_score)

        boxes = torch.cat([boxes[:max_idx], boxes[max_idx+1:]], dim=0)
        other_scores = torch.cat([scores[:max_idx], scores[max_idx+1:]], dim=0)
        ious = box_iou(max_box.unsqueeze(0), boxes).squeeze(0)

        # Score decay
        if method == 'linear': decay = torch.where(ious > iou_nms, 1 - ious, torch.ones_like(ious))
        else: decay = torch.exp(-(ious ** 2) / sigma_nms) # 'gaussian'

        scores = other_scores * decay
        keep_mask = scores > score_thresh
        boxes, scores = boxes[keep_mask], scores[keep_mask]
    return torch.stack(keep_boxes), torch.tensor(keep_scores)


def score_voting(boxes, scores, sigma_sv=0.1):
    '''
    Refines boxes using Score Voting—each box is updated by averaging nearby boxes,
    weighted by IoU similarity and confidence score (excluding self).

    Args:
        boxes (Tensor): [N, 4] tensor of boxes in [x1, y1, x2, y2] format.
        scores (Tensor): [N] tensor of calibrated confidence scores.
        sigma_sv (float): Score Voting sigma parameter.

    Returns:
        Tensor: Refined boxes.
        Tensor: Refined scores.
    '''
    if len(boxes) == 0: return boxes, scores
    iou_matrix = box_iou(boxes, boxes) # Compute pairwise IoU between all boxes [N, N]
    iou_weights = torch.exp(-((1 - iou_matrix)**2) / sigma_sv) # Compute similarity weights: high IoU -> high influence
    iou_weights.fill_diagonal_(0) # Remove self-influence by zeroing diagonal
    weights = scores.unsqueeze(1) * iou_weights # Compute weights (calibrated score * IoU weight)

    # Weighted average of boxes
    numerator = (weights.unsqueeze(2) * boxes.unsqueeze(1)).sum(dim=1) # [N, 4]
    denominator = weights.sum(dim=1, keepdim=True)
    refined_boxes = numerator / (denominator + 1e-8) # Add epsilon to avoid division by zero

    # Refined scores based on neighborhood agreement (weighted average)
    refined_scores = (scores.unsqueeze(1) * iou_weights).sum(dim=1) / (iou_weights.sum(dim=1) + 1e-8)
    return refined_boxes, refined_scores


def combine_predictions(
        image_processors, models, calibrators, dataset_name, splits_dir,
        batch_size=8, sigma_nms=0.08, iou_nms=0.65, score_thresh=0, method='gaussian'
):  # Combine predictions from multiple models using MoCaE's Refining NMS
    map_metric = MeanAveragePrecision()
    test_dataset = BreastCancerDataset(split='test', splits_dir=splits_dir, dataset_name=dataset_name, image_processor=image_processors[dataset_name])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    print(f"{dataset_name}'s test loader: {len(test_dataset)} samples")

    for batch in test_loader:
        pixel_values = batch['pixel_values'].to(device)
        with torch.no_grad():
            embeddings = feature_extractor(pixel_values)

        batch_predictions = defaultdict(list)
        for image_processor, model, calibrator in zip(image_processors.values(), models, calibrators):
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                # embeddings = outputs.last_hidden_state[:, 0, :]

            target_sizes = torch.stack([label['size'] for label in batch['labels']]).to(device)
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=0.5
            )
            for image_idx, (result, embedding) in enumerate(zip(results, embeddings)):
                pred_boxes = result['boxes'].cpu()
                pred_scores = result['scores'].cpu()
                pred_labels = result['labels'].cpu()
                cancer_mask = pred_labels == 0 # Filter predictions for class 'cancer' (class_id=0)
                pred_boxes, pred_scores, pred_labels = pred_boxes[cancer_mask], pred_scores[cancer_mask], pred_labels[cancer_mask]

                if len(pred_boxes) > 0:
                    calibrator_input = np.concatenate([
                        [embedding.cpu().numpy() for _ in range(len(pred_boxes))],
                        pred_scores.numpy().reshape(-1, 1)
                    ], axis=-1)
                    pred_scores = torch.tensor(calibrator.predict(calibrator_input))
                batch_predictions[image_idx].append({'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels})

        mocae_predictions = []
        for image_idx, expert_preds in batch_predictions.items(): # Combine predictions for each image in the batch
            combined_boxes, combined_scores, combined_labels = [], [], []
            for pred in expert_preds:
                if len(pred_boxes) > 0:
                    combined_boxes.append(pred['boxes'])
                    combined_scores.append(pred['scores'])
                    # combined_labels.append(pred['labels'])

            if len(combined_boxes) > 0:
                combined_boxes, combined_scores = soft_nms( # Apply Soft NMS
                    torch.cat(combined_boxes, dim=0), torch.cat(combined_scores, dim=0),
                    sigma_nms=sigma_nms, iou_nms=iou_nms, score_thresh=score_thresh, method=method
                )
                combined_labels = torch.zeros_like(combined_scores, dtype=torch.int64) # Set all labels to 0 due to dataset's nature

                # Apply Score Voting
                combined_boxes, combined_scores = score_voting(combined_boxes, combined_scores, sigma_sv=sigma_nms)
                mocae_predictions.append({'boxes': combined_boxes, 'scores': combined_scores, 'labels': combined_labels})
            else: mocae_predictions.append({'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])})

        targets = [{
            'boxes': convert_bbox_yolo_to_pascal(label['boxes'], label['size']),
            'labels': label['class_labels']
        } for label in batch['labels']]
        map_metric.update(mocae_predictions, targets)
    return map_metric.compute()


for config in CONFIGS:
    print('Loading artifacts from', config['saved_dir'])
    config['image_processor'] = AutoImageProcessor.from_pretrained(
        config['saved_dir'],
        do_resize=True, do_pad=True, use_fast=True,
        size={'max_height': MAX_SIZE, 'max_width': MAX_SIZE},
        pad_size={'height': MAX_SIZE, 'width': MAX_SIZE},
    )
    model = AutoModelForObjectDetection.from_pretrained(
        config['saved_dir'],
        id2label={0: 'cancer'},
        label2id={'cancer': 0},
        ignore_mismatched_sizes=True,
    )
    model.eval()
    model.to(device)
    config['model'] = model

    inputs_val, ious_val = combine_datasets(config, DATASET_NAMES, split='val')
    config['calibrator_dataset'] = (inputs_val, ious_val)


for config in tqdm(CONFIGS):
    dataset_name, (inputs_val, ious_val) = config['dataset_name'], config['calibrator_dataset']
    calibrator = RandomForestRegressor(n_estimators=300, n_jobs=-1)
    calibrator.fit(inputs_val, ious_val)

    calibrator_path = os.path.join(config['saved_dir'], 'calibrator.pkl')
    with open(calibrator_path, 'wb') as f:
        pickle.dump(calibrator, f)
    config['calibrator'] = calibrator

    y_pred = calibrator.predict(inputs_val)
    mse = mean_squared_error(ious_val, y_pred)
    mae = mean_absolute_error(ious_val, y_pred)
    r2 = r2_score(ious_val, y_pred)
    print(f'[Calibrated {dataset_name}] MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')


models = [config['model'] for config in CONFIGS]
calibrators = [config['calibrator'] for config in CONFIGS]
image_processors = {config['dataset_name']: config['image_processor'] for config in CONFIGS}

for dataset_name in tqdm(DATASET_NAMES): # Combine predictions for each dataset
    print(combine_predictions(image_processors, models, calibrators, dataset_name, SPLITS_DIR))