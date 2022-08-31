import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from collections import deque
from tqdm import tqdm
import json
from bdd_utils import to_mask, bbox_from_instance_mask, get_colored_mask
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Define color map to be used when displaying the images with bounding boxes
COLOR_MAP = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue']


class BDD(Dataset):

    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 obj_cls: list,
                 img_size: int = 600,
                 stage: str = 'train'):

        super(BDD, self).__init__()

        assert stage in ['train', 'test'], "You must choose between 'train' and 'test' for stage parameter."

        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.obj_cls = obj_cls
        self.img_size = img_size
        self.stage = stage

        if self.stage == 'train':
            self.images_dir = self.images_dir / Path('train')
            self.polygon_path = self.masks_dir / Path('polygons/ins_seg_train.json')
        elif self.stage == 'test':
            self.images_dir = self.images_dir / Path('val')
            self.polygon_path = self.masks_dir / Path('polygons/ins_seg_val.json.json')

        self.cls_to_idx, self.idx_to_cls = self.__create_idx()
        self.db = self.__create_db()

    def __create_idx(self):
        cls_to_idx = {}
        idx_to_cls = {}
        idx = 0

        for obj in self.obj_cls:

            # if obj is a traffic light, add the class with the color except the NA
            if obj == 'traffic light':
                """
                cls_to_idx['tl_NA'] = idx
                idx_to_cls[idx] = 'tl_NA'
                idx += 1
                """

                cls_to_idx['tl_G'] = idx
                idx_to_cls[idx] = 'tl_G'
                idx += 1

                cls_to_idx['tl_R'] = idx
                idx_to_cls[idx] = 'tl_R'
                idx += 1

                cls_to_idx['tl_Y'] = idx
                idx_to_cls[idx] = 'tl_Y'
                idx += 1

            else:
                cls_to_idx[self.obj_cls[idx]] = idx
                idx_to_cls[idx] = self.obj_cls[idx]
                idx += 1

        return cls_to_idx, idx_to_cls

    def __load_annotations(self):
        with open(self.polygon_path, 'r') as f:
            polygon_annotation = json.load(f)

        return polygon_annotation

    def __filter_labels(self, labels):
        filtered_labels = []
        for label in labels:
            if label['category'] in self.obj_cls:
                filtered_labels.append(label)

        return filtered_labels

    def __create_db(self):
        polygon_annotations = deque(self.__load_annotations())
        db = deque()

        for polygon in tqdm(polygon_annotations):

            filtered_labels = self.__filter_labels(polygon['labels'])

            if len(filtered_labels) > 0:
                db.append({
                    'image_path': self.images_dir / Path(polygon['name']),
                    'labels': filtered_labels
                })

        return db

    def data_augmentation(self, image, masks, bboxes, labels):
        """
        method to apply image augmentation technics to reduce overfitting
        :param image: numpy array with shape of HxWx3 (RGB image)
        :param masks: list of masks, each mask must have the same W and H with the image (2D mask)
        :param bboxes: list of bounding boxes, each box must have (xmin, ymin, xmax, ymax)
        :param labels: idx of the labels
        :return: image, masks, bboxes
        """
        class_labels = [self.idx_to_cls[label] for label in labels]
        for idx, box in enumerate(bboxes):
            box.append(class_labels[idx])

        augmentation_transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.HorizontalFlip(p=1),  # Random Flip with 0.5 probability
            A.CropAndPad(px=100, p=0.5),  # crop and add padding with 0.5 probability
            A.PixelDropout(dropout_prob=0.01, p=0.5),  # pixel dropout with 0.5 probability
        ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3))  # return bbox with xyxy format

        transformed = augmentation_transform(image=image, masks=masks, bboxes=bboxes)

        transformed_boxes = []
        transformed_labels = []
        for box in transformed['bboxes']:
            box = list(box)
            label = box.pop()
            transformed_boxes.append(box)
            transformed_labels.append(label)

        labels = [self.cls_to_idx[label] for label in transformed_labels]

        return transformed['image'], transformed['masks'], transformed_boxes, labels

    @staticmethod
    def image_transform(image):
        t_ = transforms.Compose([
            transforms.ToTensor(),  # convert the image to tensor
            transforms.Normalize(mean=[0.407, 0.457, 0.485],
                                 std=[0.229, 0.224, 0.225])  # normalize the image using mean ans std
        ])
        return t_(image)

    def get_image(self, idx):
        image_path = self.db[idx]['image_path']
        image = np.array(Image.open(image_path).convert('RGB'))
        return image

    def get_masks(self, idx):
        image_annotation = self.db[idx]
        mask_shape = self.get_image(idx).shape
        target = {}
        boxes = []
        masks = []
        labels = []

        for label in image_annotation['labels']:
            poly2d = label['poly2d'][0]['vertices']
            mask = to_mask(mask_shape, poly2d)
            box = bbox_from_instance_mask(mask)
            label = self.cls_to_idx[label['category']]

            masks.append(np.array(mask, dtype=np.uint8))
            boxes.append(box)
            labels.append(label)

        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks

        return target

    def display(self, idx):
        image = self.get_image(idx)
        target = self.get_masks(idx)
        boxes = target['boxes']
        labels = target['labels']
        masks = target['masks']

        for mask in masks:
            rgb_mask = get_colored_mask(mask)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)
        for i, mask in enumerate(labels):
            bbox = boxes[i]
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     edgecolor=COLOR_MAP[labels[i]],
                                     facecolor="none", linewidth=2)
            plt.text(bbox[0], bbox[1], self.idx_to_cls[labels[i]], verticalalignment="top",
                     color=COLOR_MAP[labels[i]])

            ax.add_patch(rect)

        plt.axis('off')
        plt.show()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        image = self.get_image(idx)
        target = self.get_masks(idx)

        if self.stage == 'train':
            image, masks, bboxes, labels = self.data_augmentation(np.array(image), target['masks'], target['boxes'],
                                                                  target['labels'])
        else:
            masks = target['masks']
            bboxes = target['boxes']
            labels = target['labels']

        image = self.image_transform(image)

        target['boxes'] = torch.tensor(bboxes)

        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        target['masks'] = torch.tensor(np.array(masks, dtype=np.uint8))

        return image, target