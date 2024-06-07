from GPUtil import showUtilization as gpu_usage

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import cv2
# import random
import glob
import PIL
from PIL import Image
from tqdm import tqdm
import imghdr
from patchify import patchify 


import time
import torch
import torchvision
import torch.optim as optim
import albumentations as A
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import Dropout
from torchsummary import summary

from torch.nn import BatchNorm2d 

from torchvision.transforms import CenterCrop
from torch.nn import functional as F
from torch.nn.functional import normalize




def readData(imgPath, labelPath, convertType):
    """Reads and creates a list of target""" 
    image = []
    label = []
    imageList = []
    labelList = []
  
    for i, image_name in enumerate(sorted(os.listdir(imgPath))):
        if ((('.').join(image_name.split('.')[-1:])== 'tif') or (('.').join(image_name.split('.')[-1:]) == 'tiff')):
            label_name = '.'.join(image_name.split('.')[:-1]) +  '_bn.tif'
            
            if label_name in list(os.listdir(labelPath)): 
                # normalise by 255.0 -> convert to array -> append to list
                img_Path = os.path.join(imgPath, image_name)
                img = Image.open(img_Path).convert(convertType)
                img = np.array(img, dtype = np.float32)/255.0
                image.append(img)
                imageList.append((img_Path))
                
                label_Path = os.path.join(labelPath, label_name)
                img = Image.open(label_Path).convert(convertType)
                label.append(np.where(np.array(img) >= 1, 1.0, 0.0))
                labelList.append((label_Path))
            else:
                print('Images with no mask-->', image_name)
        else: print('Image with new extension', image_name)
    print(f'total images --> {len(imageList)}, total masks --> {len(labelList)}')       
    return image, label


def dataTransform(image, mask):
    images_list, masks_list = [], []
    
    transform = A.Compose([ A.HorizontalFlip(p = 0.5),
                            A.VerticalFlip(p = 0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.ElasticTransform(p=0.5),
                            A.GridDistortion(p = 0.5), ])

    for i in range(NUM_AUGMENTATION):
        augmentations = transform(image = np.array(image), mask = np.array(mask))
        images_list.append(augmentations["image"])
        masks_list.append(augmentations["mask"])
        
    return images_list, masks_list



def createPatches(imgList, maskList, PATCH_SIZE):
    images = []
    masks = []
    for i, (image, mask) in enumerate(zip(imgList, maskList)):      
        patch_images = patchify(image, (PATCH_SIZE,PATCH_SIZE), step = PATCH_SIZE)
        patch_masks = patchify(mask, (PATCH_SIZE,PATCH_SIZE), step = PATCH_SIZE)
        
        for i in range(patch_images.shape[0]):
            for j in range(patch_images.shape[1]):
                single_patch_img = patch_images[i,j,:,:]
                images.append(single_patch_img)
                
                single_patch_img = patch_masks[i,j,:,:]
                masks.append(single_patch_img)
                
    images = torch.reshape(torch.tensor(np.array(images)), [-1,1,PATCH_SIZE,PATCH_SIZE])       
    masks = torch.reshape(torch.tensor(np.array(masks)), [-1,1,PATCH_SIZE,PATCH_SIZE])
        
    return images, masks    
    


class SpheroidDataset(Dataset):

    def __init__(self, imagePaths, maskPaths, transforms = None, num_augmentations=NUM_AUGMENTATION):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.num_augmentations = num_augmentations

        # Read and store image and mask data efficiently
        self.images, self.masks = readData(self.imagePaths, self.maskPaths, "L")

    def __len__(self):
        if self.transforms == True:
            length = len(self.images) * self.num_augmentations
        else:
            length = len(self.images)
        print(f'Images Augmentated-->{self.transforms}; so Dataset_length-->{length}')
        return length 
    

    def __getitem__(self, idx):
        if self.transforms == True:
            original_idx = idx // self.num_augmentations
            augmentation_idx = idx % self.num_augmentations
        else:
            original_idx = idx 

        # Retrieve image and mask from pre-loaded data
        img = self.images[original_idx]
        mask = self.masks[original_idx]

        # Transform(y/n) -> patches 
        if self.transforms == True:
            images_list, masks_list = dataTransform(img, mask)
            augmented_img = images_list[augmentation_idx]
            augmented_mask = masks_list[augmentation_idx]
            image_patch, mask_patch = createPatches([augmented_img], [augmented_mask], PATCH_SIZE)
        else:
            image_patch, mask_patch = createPatches([img], [mask], PATCH_SIZE)

        return image_patch, mask_patch 