import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.onnx
import timm
import timm.optim as tioptim
import matplotlib.pyplot as plt
import os
import time
import json

from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import ImageFolder

from sklearn.metrics import confusion_matrix

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# time start
start_time = time.time()

# load data 
data_dir = '/mnt/sda/lung_colon_image_set/lung_image_sets'

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

datasets = ImageFolder(data_dir, transform=data_transform)
train_size = int(0.8 * len(datasets))
test_size = len(datasets) - train_size
train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

teacher_model = torch.load('teacher_model.pth')
student_model = timm.create_model('resnet18', num_classes=3)