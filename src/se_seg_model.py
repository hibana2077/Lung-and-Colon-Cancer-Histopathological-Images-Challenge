'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-30 20:08:08
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-06-04 21:49:10
FilePath: \Lung-and-Colon-Cancer-Histopathological-Images-Challenge\src\se_seg_model.py
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
import json

from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

# time start
start_time = time.time()

# load data 
data_dir = './data/lung_colon_image_set/lung_image_sets'