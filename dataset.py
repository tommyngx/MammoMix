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

from utils import parse_voc_xml, xml2dicts

class BreastCancerDataset(Dataset):
    def __init__(self, split, splits_dir, dataset_name, image_processor, model_type='detr', dataset_epoch=None):
        """
        Args:
            split (str): Split type ('train', 'val', 'test').
            splits_dir (str): Path to the directory containing split .txt files.
            dataset_name (str): Name of the dataset.
            image_processor: DETR image processor.
            model_type (str): 'detr' or 'yolos'. Controls pixel_mask output.
            dataset_epoch (int, optional): Epoch value, if provided.
        """
        self.split = split
        self.splits_dir = splits_dir
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.dataset_epoch = dataset_epoch
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be one of ['train', 'val', 'test'].")

        split_file = os.path.join(splits_dir, dataset_name, f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f'{split_file} not found. Please check the path.')
        with open(split_file, 'r') as f:
            self.image_paths = [os.path.join(splits_dir, line.strip()) for line in f.readlines()]

        self.image_processor = image_processor
        self.transforms = self.get_transforms()

    def get_transforms(self):
        if self.split == 'train':
            return A.Compose([
                #A.ElasticTransform(alpha=50, sigma=5, approximate=False, p=0.5),
                #A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.Affine(
                    scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-10, 10), shear=(-5, 5),
                    interpolation=1, p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(std_range=(0.05, 0.05), mean_range=(0.0, 0.0), per_channel=True, p=0.5),
                #A.GaussianBlur(p=0.5),
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_area=25,
                min_visibility=0.1,
                clip=True
            ))
        return A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True)
        )

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base, ext = os.path.splitext(img_path)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(f"Unsupported image format: {ext}. Supported formats are .jpg, .jpeg, .png")

        image = np.array(Image.open(img_path).convert('RGB'))
        # Store original image dimensions for evaluation
        original_height, original_width = image.shape[:2]
        
        label_path = base.replace('images', 'labels') + '.xml'
        bboxes, labels = [], []

        if os.path.exists(label_path):
            xml_data = parse_voc_xml(label_path)
            annotations = xml2dicts(xml_data['bboxes'], xml_data['width'], xml_data['height'])
            for ann in annotations:
                bboxes.append([ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']])
                labels.append(ann['class_id'])
        else:
            print(f'No annotation found for {img_path}, skipping.')

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
        labels = np.array(transformed['labels'], dtype=np.int64)
        if len(transformed['labels']) <= 0:
            return self.__getitem__(idx)  # Retry if no valid boxes after augmentation

        image = transformed['image']
        bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        annotations = {
            'image_id': idx,
            'annotations': [{
                'image_id': idx,
                'category_id': label,
                'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                'iscrowd': 0
            } for bbox, label in zip(bboxes, labels)]
        }

        encoding = self.image_processor(images=image, annotations=annotations, return_tensors='pt')
        result = {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': encoding['labels'][0],
            # Include size as a tuple of (height, width)
            'size': (original_height, original_width)
        }
        if self.model_type == 'detr' and 'pixel_mask' in encoding:
            result['pixel_mask'] = encoding['pixel_mask'].squeeze(0)
        # If YOLOS, do not include pixel_mask (commented out)
        # if self.model_type == 'yolos':
        #     #'pixel_mask': encoding['pixel_mask'].squeeze(0),
        return result

    def __len__(self):
        return len(self.image_paths)

def collate_fn(batch):
    import torch
    data = {}
    data['pixel_values'] = torch.stack([x['pixel_values'] for x in batch])
    data['labels'] = [x['labels'] for x in batch]
    # Include size information in collated batch
    data['size'] = [x['size'] for x in batch]
    if 'pixel_mask' in batch[0]:
        data['pixel_mask'] = torch.stack([x['pixel_mask'] for x in batch])
    return data
