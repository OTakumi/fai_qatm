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

class Featex():
    '''
    Extract Feature
    '''
    def __init__(self, model, use_cuda):
        self.use_cuda = use_cuda
        self.feature1 = None
        self.feature2 = None
        self.model = copy.deepcopy(model.eval())
        self.model = self.model[:17]
        for param in self.model.parameters():
            param.requires_grad = False
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model[2].register_forward_hook(self.save_feature1)
        self.model[16].register_forward_hook(self.save_feature2)
    
    def save_feature1(self, module, input, output):
        self.feature1 = output.detach()
    
    def save_feature2(self, module, input, output):
        self.feature2 = output.detach()

    def __call__(self, input, mode='big'):
        if self.use_cuda:
            input = input.cuda()
            _ = self.model(input)
        if mode=='big':
            # resize feature1 to the same size of feature2
            self.feature1 = F.interpolate(self.feature1, size=(self.feature2.size()[2], self.feature2.size()[3]), 
            model='bilinear', align_corners=True)
        else:
            # resize feature2 to the same size of feature1
            self.feature2 = F.interpolate(self.feature2, size=(self.feature1.size()[2], self.feature1.size()[3]), 
            model='bilinear', align_corners=True)
        
        return torch.cat((self.feature1, self.feature2), dim=1)

class MyNormLayer():
    def __call__(self, x1, x2):
        bs, _, H, W = x1.size()
        _, _, h, w = x2.size()
        x1 = x1.view(bs, -1, H*W)
        x2 = x2.view(bs, -1, h*w)
        concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True)
        x1 = (x1 - x_mean) / x_std
        x2 = (x2 - x_mean) / x_std
        x1 = x1.view(bs, -1, H, W)
        x2 = x2.view(bs, -1, h, w)

        return [x1, x2]

class CreateModel():
    def __init__(self, alpha, model, use_cuda):
        self.alpha = alpha
        self.featex = Featex(model, use_cuda)
        self.I_feat = None
        self.I_feat_name = None

    def __call__(self, tempplate, image, image_name):
        T_feat = self.featex(template)
        if self.I_feat_name is not image_name:
            self.I_feat = self.featex(image)
            self.I_feat_name = image_name
        conf_maps = None
        batchsize_T = T_feat.size()[0]
        for i in range(batchsize_T):
            T_feat_i = T_feat[i].unsqueeze(0)
            I_feat_norm, T_feat_i = MyNormLayer()(self.I_feat, T_feat_i)
            dist = torch.einsum("xcab, xcde -> xabde", I_feat_norm / torch.norm(I_feat_norm, dim=1, keepdim=True),
                                T_feat_i / torch.norm(T_feat_i, dim=1, keepdim=True))
            conf_map = QATM(self.alpha)(dist)
            if conf_maps is None:
                conf_maps = conf_map
            else:
                conf_maps = torch.cat([conf_maps, conf_map], dim=0)
            
        return conf_maps

