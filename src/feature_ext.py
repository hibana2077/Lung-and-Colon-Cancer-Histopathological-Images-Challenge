'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-06-07 12:08:43
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-06-08 00:24:31
FilePath: \Lung-and-Colon-Cancer-Histopathological-Images-Challenge\src\feature_ext.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import timm.optim
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import timm
import timm.optim as tioptim
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import json

from rich import print as rprint
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import confusion_matrix

# time start
start_time = time.time()

# load data 
data_dir = './data/lung_colon_image_set/lung_image_sets'

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

datasets = ImageFolder(data_dir, transform=data_transform)
train_size = int(0.8 * len(datasets))
test_size = len(datasets) - train_size
train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# load model
model = torch.load('./model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

# print model
rprint(model)

model.eval()
dummy_input = torch.randn(1, 3, 256, 256)
dummy_input = dummy_input.to(device)
test_out = model.forward_features(dummy_input)
print(test_out.shape)