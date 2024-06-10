'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-06-07 12:08:43
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-06-10 10:08:34
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

dummy_input = torch.randn(1, 1024, 8, 8)
test_model = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
test_out = test_model(dummy_input)
print(test_out.shape)

flatten_model = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

# Feature extraction

train_features = []
train_labels = []
test_features = []
test_labels = []

time_feature_extraction_start = time.time()
for inputs, labels in tqdm(train_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    features = flatten_model(model.forward_features(inputs))
    train_features.append(features.cpu().detach().numpy())
    train_labels.append(labels.cpu().detach().numpy())

for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    features = flatten_model(model.forward_features(inputs))
    test_features.append(features.cpu().detach().numpy())
    test_labels.append(labels.cpu().detach().numpy())

print('Feature extraction time:', time.time() - time_feature_extraction_start)

train_features = np.concatenate(train_features, axis=0)
train_labels = np.concatenate(train_labels, axis=0)
test_features = np.concatenate(test_features, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

print(train_features.shape, train_labels.shape) 
print(test_features.shape, test_labels.shape)

# concat features and labels
train_data = np.concatenate([train_features, train_labels[:, np.newaxis]], axis=1)
test_data = np.concatenate([test_features, test_labels[:, np.newaxis]], axis=1)

# convert to pandas dataframe
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# save to csv
train_df.to_csv('./train_features.csv', index=False)
test_df.to_csv('./test_features.csv', index=False)