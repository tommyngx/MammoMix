import os
import shutil
import logging
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import parse_voc_xml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_with_csv(xml_data, csv_path): # Cross-validate XML annotations with bbox_annotations.csv
    csv_data = pd.read_csv(csv_path)
    csv_row = csv_data[csv_data['name'] == xml_data['image_name']]
    if csv_row.empty:
        logger.warning(f"No CSV entry found for {xml_data['image_name']}")
        return True  # Proceed, but log warning
    
    csv_width, csv_height = csv_row['width'].iloc[0], csv_row['height'].iloc[0]
    if csv_width != xml_data['width'] or csv_height != xml_data['height']:
        logger.error(f"Size mismatch for {xml_data['image_name']}: XML ({xml_data['width']}, {xml_data['height']}) vs CSV ({csv_width}, {csv_height})")
        return False
    return True


def split_dataset(dataset_name, raw_data_dir, splits_dir, val_split=0.2):
    logger.info(f"Splitting dataset: {dataset_name}")
    raw_dataset_dir = os.path.join(raw_data_dir, dataset_name)
    processed_dataset_dir = os.path.join(splits_dir, dataset_name)
    csv_path = os.path.join(raw_dataset_dir, 'bbox_annotations.csv')
    
    for split in ['train', 'val', 'test']: # Create directories
        os.makedirs(os.path.join(processed_dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(processed_dataset_dir, split, 'labels'), exist_ok=True)
    
    for split in ['train', 'test']: # Process train and test data
        image_dir = os.path.join(raw_dataset_dir, split, 'images')
        label_dir = os.path.join(raw_dataset_dir, split, 'labels')
        processed_image_dir = os.path.join(processed_dataset_dir, split, 'images')
        processed_label_dir = os.path.join(processed_dataset_dir, split, 'labels')
        
        image_paths = []
        for img_file in tqdm(os.listdir(image_dir)):
            if not img_file.endswith(('.jpg', '.png')): continue
            img_path = os.path.join(image_dir, img_file)
            xml_path = os.path.join(label_dir, Path(img_file).stem + '.xml')
            
            if not os.path.exists(xml_path):
                logger.warning(f"No XML annotation for {img_file}")
                continue
            
            xml_data = parse_voc_xml(xml_path) # Parse XML
            if not validate_with_csv(xml_data, csv_path):
                continue

            # Copy image and XML files
            shutil.copy(img_path, os.path.join(processed_image_dir, img_file))
            shutil.copy(xml_path, os.path.join(processed_label_dir, Path(img_file).stem + '.xml'))
            image_paths.append(os.path.join(dataset_name, split, 'images', img_file).replace('\\', '/'))
            
        # Save split file
        split_file = os.path.join(processed_dataset_dir, f"{split}.txt")
        with open(split_file, 'w') as f:
            f.write('\n'.join(image_paths))
    
    # Create validation split from training data
    train_images = [p for p in open(os.path.join(processed_dataset_dir, 'train.txt')).read().splitlines()]
    train_images, val_images = train_test_split(train_images, test_size=val_split, random_state=42)
    val_images = [p.replace('train', 'val') for p in val_images] # Update paths for validation split
    
    with open(os.path.join(processed_dataset_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_images)) # Update train split
    
    with open(os.path.join(processed_dataset_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_images)) # Save validation split
    
    for img_path in tqdm(val_images): # Move validation images and labels
        img_file = os.path.basename(img_path)
        shutil.move(
            os.path.join(processed_dataset_dir, 'train', 'images', img_file),
            os.path.join(processed_dataset_dir, 'val', 'images', img_file)
        )
        shutil.move(
            os.path.join(processed_dataset_dir, 'train', 'labels', Path(img_file).stem + '.xml'),
            os.path.join(processed_dataset_dir, 'val', 'labels', Path(img_file).stem + '.xml')
        )
    logger.info(f"Completed splitting for {dataset_name}")


# Configuration
DATASETS = ['CSAW', 'DDSM', 'DMID']
RAW_DATA_DIR = 'AJCAI25/raw'
SPLITS_DIR = 'AJCAI25/splits'
VAL_SPLIT = 0.2

os.makedirs(SPLITS_DIR, exist_ok=True)
for dataset in DATASETS:
    split_dataset(dataset, RAW_DATA_DIR, SPLITS_DIR, VAL_SPLIT)