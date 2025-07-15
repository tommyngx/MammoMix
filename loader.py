import os
import torch
import numpy as np
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from utils import parse_voc_xml, xml2dicts


class BreastCancerDataset(Dataset):
    def __init__(self, split, splits_dir, dataset_name, image_processor):
        '''
        Args:
            split (str): Split type ('train', 'val', 'test').
            splits_dir (str): Path to the directory containing split .txt files.
            dataset_name (str): Name of the dataset.
            image_processor (AutoImageProcessor): DETR image processor.
        '''
        self.split = split
        self.splits_dir = splits_dir
        self.dataset_name = dataset_name
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be one of ['train', 'val', 'test'].")

        split_file = os.path.join(splits_dir, dataset_name, f'{split}.txt')
        if not os.path.exists(split_file): raise FileNotFoundError(f'{split_file} not found. Please check the path.')
        with open(split_file, 'r') as f:
            self.image_paths = [os.path.join(splits_dir, line.strip()) for line in f.readlines()]

        self.image_processor = image_processor
        self.transforms = self.get_transforms()  # Get augmentation transforms


    def get_transforms(self):
        if self.split == 'train': # Apply augmentation if training
            return A.Compose([
                # Geometric transformations
                A.ElasticTransform(alpha=50, sigma=5, approximate=False, p=0.5), # Elastic deformation to simulate tissue variability
                A.Perspective(scale=(0.05, 0.1), p=0.5), # Perspective distortion to simulate different angles
                A.HorizontalFlip(p=0.5), # Mirror image
                A.Rotate(limit=10, p=0.5), # Small angles to avoid disrupting anatomical structure
                A.RandomScale(scale_limit=0.2, p=0.5), # Random scaling to simulate different distances
                A.Affine(
                    scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-10, 10), shear=(-5, 5),
                    interpolation=1, p=0.5 # Affine transformation to simulate different angles and scales
                ),

                # Color and intensity transformations
                # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(std_range=(0.05, 0.05), mean_range=(0.0, 0.0), per_channel=True, p=0.5),
                A.GaussianBlur(p=0.5),
            ], bbox_params=A.BboxParams(
                format='pascal_voc', # [x_min, y_min, x_max, y_max]
                label_fields=['labels'], # Labels for bounding boxes
                min_area=25, # Drop boxes smaller than 25 pixels after augmentation
                min_visibility=0.1, # Discard boxes with less than 10% visibility after augmentation
                clip=True # Clip bounding boxes to image boundaries
            ))
        return A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True))


    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base, ext = os.path.splitext(img_path)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError(f"Unsupported image format: {ext}. Supported formats are .jpg, .jpeg, .png")

        image = np.array(Image.open(img_path).convert('RGB'))
        label_path = base.replace('images', 'labels') + '.xml'
        bboxes, labels = [], []

        if os.path.exists(label_path):
            xml_data = parse_voc_xml(label_path)
            annotations = xml2dicts(xml_data['bboxes'], xml_data['width'], xml_data['height'])
            for ann in annotations:
                bboxes.append([ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']])
                labels.append(ann['class_id'])
        else: print(f'No annotation found for {img_path}, skipping.')

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
        labels = np.array(transformed['labels'], dtype=np.int64)
        if len(transformed['labels']) <= 0: return self.__getitem__(idx) # Retry if no valid boxes after augmentation

        image = transformed['image']
        bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        annotations = { 'image_id': idx, 'annotations': [{ # Prepare annotations in COCO-like format
            'image_id': idx,
            'category_id': label,
            'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], # [x_min, y_min, width, height]
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            'iscrowd': 0
        } for bbox, label in zip(bboxes, labels)]}

        # Apply the image processor transformations: resizing, rescaling, normalization
        encoding = self.image_processor(images=image, annotations=annotations, return_tensors='pt')
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),  # Remove batch dimension
            # 'pixel_mask': encoding['pixel_mask'].squeeze(0),
            'labels': encoding['labels'][0]  # DETR expects a dict with 'boxes', 'class_labels'
        }

    def __len__(self):
        return len(self.image_paths)


def collate_fn(batch):
    data = {}
    data['pixel_values'] = torch.stack([x['pixel_values'] for x in batch])
    data['labels'] = [x['labels'] for x in batch]
    if 'pixel_mask' in batch[0]:
        data['pixel_mask'] = torch.stack([x['pixel_mask'] for x in batch])
    return data


if __name__ == '__main__':
    DATASET_NAME = 'CSAW'
    SPLITS_DIR = 'AJCAI25/splits'
    MODEL_NAME = 'hustvl/yolos-base'
    MAX_SIZE = 640

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        do_resize=True, do_pad=True, use_fast=True,
        size={'max_height': MAX_SIZE, 'max_width': MAX_SIZE},
        pad_size={'height': MAX_SIZE, 'width': MAX_SIZE},
    )
    train_dataset = BreastCancerDataset(
        split='train',
        splits_dir=SPLITS_DIR,
        dataset_name=DATASET_NAME,
        image_processor=image_processor
    )
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    print(f'Train loader: {len(train_dataset)} samples')

    for sample in train_loader:
        pixel_values, pixel_mask, labels = sample['pixel_values'], sample.get('pixel_mask'), sample['labels']
        print(f'Batch images shape: {pixel_values.shape}')
        if pixel_mask is not None: print('Pixel mask shape:', pixel_mask.shape)
        for label in labels: print(label)
        break