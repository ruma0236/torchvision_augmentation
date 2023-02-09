import torch
import numpy as np
import matplotlib.pylab as plt
import os, time, cv2
from tqdm import tqdm
# from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from utils import SegmentationDataset
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from albumentations import HorizontalFlip, Compose, Resize, Normalize

# Scenario J
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=1024, scale=(0.4, 1.0)),
        transforms.RandomAffine(degrees=90, translate=None, scale=(0.8, 1.2), shear=20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
    ]
)

dataset = SegmentationDataset(
    dir_images = "./ISIC-2017_Training_Data/",
    dir_masks = "./ISIC-2017_Training_Part1_GroundTruth/",
    transform=transform
)

for i in tqdm(range(len(dataset))):
    img, mask = dataset[i]
    img = img.permute(1, 2, 0)
    mask = mask.permute(1, 2, 0)
    plt.figure(figsize=(25,10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(mask)

    plt.savefig('results_add_to_colorjitter/'+str(i)+'.png',format='png')
