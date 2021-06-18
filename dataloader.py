# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:02:03 2021

@author: Tamal
"""
import itertools, imageio, torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
#from scipy.misc import imresize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import PIL
import os
import cv2

    
    
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, path, resize_to):
        'Initialization'
        self.list_rgb = []
        self.list_ir = []
        self.path = path
        self.resize_to = resize_to

        list_rgb_ = os.listdir(os.path.join(self.path, 'RGB'))
        list_ir_ = os.listdir(os.path.join(self.path, 'thermal_8_bit'))

        for i in list_ir_:
            name = i.split('.')[0]
            if name+'.jpg' in list_rgb_:
                self.list_rgb.append(name+'.jpg')
                self.list_ir.append(name+'.jpeg')

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_ir)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #ID = self.list_IDs[index]

        # Load data and get label


        rgb = cv2.resize(cv2.imread(os.path.join(self.path, 'RGB',
                                                   self.list_rgb[index])),
                                                   (self.resize_to,self.resize_to)) / 255.0
        ir = cv2.resize(cv2.imread(os.path.join(self.path, 'thermal_8_bit',
                                                  self.list_ir[index]), 0),
                                                  (self.resize_to,self.resize_to)) / 255.0
        rgb = torch.tensor(rgb, dtype = torch.float32).permute(2,0,1)
        ir = torch.tensor(ir, dtype = torch.float32).unsqueeze(0)

        return rgb, ir