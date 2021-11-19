import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch.cuda
import albumentations as albu
import segmentation_models_pytorch as smp
from  segmentation_models_pytorch.utils.base import Metric
from segmentation_models_pytorch.base.modules import Activation
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
def load_data(path):
    all_images = os.listdir(path / 'images')
    all_masks = os.listdir(path / 'masks')

    data = {'images': [],
            'masks': []}
    for i in range(len(all_images)):
        data['images'].append(str(path / 'images' / all_images[i]))
        data['masks'].append(str(path / 'masks' / all_masks[i]))
    return pd.DataFrame(data)


class Dataset(BaseDataset):
    def __init__(
            self,
            images_path,
            masks_path,
            augmentation=None,
            preprocessing=None,
    ):
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.images = images_path
        self.masks = masks_path

    def __getitem__(self, i):
        #         print(self.images[i])
        
        image = cv2.imread(str(self.images[i]))
        mask = cv2.imread(self.masks[i], 0)
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images)