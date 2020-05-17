import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
from utils import *

class ImageDataSet(torch.utils.data.Dataset):
    def __init__(self, template_dir_path, image_name, thresh_csv=None, transform=None):
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.template_path = list(template_dir_path.iterdir())
        self.image_name = image_name

        self.image_raw = cv2.imread(self.image_name)

        self.thresh_df = None
        if thresh_csv:
            self.thresh_df = pd.read_csv(thresh_csv)
        
        if self.transform:
            self.image = self.transform(self.image_raw).unsqueeze(0)

def __len__(self):
    return len(self.template_names)

def __getitem__(self, idx):
    template_path = str(self.template_path[idx])
    template = cv2.imread(template_path)
    if self.transform:
        template_path = self.transform(template)
    thresh = 0.7
    if self.thresh_df is not None:
        if self.thresh_df.path.isin([template_path]).sum() > 0:
            thresh = float(self.thresh_df[self.thresh_df.path == template_path].thresh)
        
    return {'image_raw': self.image_raw,
            'image_name': self.image_name,
            'template': template.unsqueeze(0),
            'template_name': template_path,
            'template_h': template.size()[-2],
            'template_w': template.size()[-1],
            'thresh': thresh}

template_dir = 'template/'
image_path = 'sample/sample.jpg'
dataset = ImageDataSet(Path(template_dir), image_path, thresh_csv='thresh_template.csv')