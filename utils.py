import yaml
from transformers import AutoImageProcessor
import xml.etree.ElementTree as ET

def load_config(config_path='configs/train_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_image_processor(model_name, max_size):
    return AutoImageProcessor.from_pretrained(
        model_name,
        do_resize=True, do_pad=True, use_fast=True,
        size={'max_height': max_size, 'max_width': max_size},
        pad_size={'height': max_size, 'width': max_size},
    )

def get_model_type(model_name):
    return 'yolos' if 'yolos' in model_name else 'detr'

def parse_voc_xml(xml_path):
    """
    Parse VOC XML annotation file to extract bounding box information.
    Returns a dict with image name, width, height, and list of bboxes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_name = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        if xmin < xmax and ymin < ymax:
            bboxes.append({
                'class': name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
        else:
            print(f'Invalid bbox in {image_name}: ({xmin}, {ymin}, {xmax}, {ymax})')
    return {
        'image_name': image_name,
        'width': width,
        'height': height,
        'bboxes': bboxes
    }

def xml2dicts(bboxes, width, height):
    """
    Convert VOC boxes to a list of dictionaries for DETR.
    """
    detr_bboxes = []
    for bbox in bboxes:
        class_id = 0  # Single class 'cancer'
        xmin, ymin = bbox['xmin'], bbox['ymin']
        xmax, ymax = bbox['xmax'], bbox['ymax']
        detr_bboxes.append({
            'class_id': class_id,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })
    return detr_bboxes
