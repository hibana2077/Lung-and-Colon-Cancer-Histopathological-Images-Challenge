'''
Author: hibana2077 hibana2077@gmail.com
Date: 2024-05-29 14:45:23
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2024-06-03 15:14:03
FilePath: \Lung-and-Colon-Cancer-Histopathological-Images-Challenge\src\main.py
Description: 
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

from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import confusion_matrix

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

model = timm.create_model('convnext_base', num_classes=3)

# define loss function and optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = tioptim.Lookahead(timm.optim.AdamW(model.parameters(), lr=1e-3))

# train model
num_epochs = 10
loss_history = []
acc_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = []
    running_corrects = 0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate accuracy
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        # calculate loss
        running_loss.append(loss.item())

    acc_history.append(running_corrects/len(train_dataset))
    loss_history.append(np.mean(running_loss))
    print(f'Epoch {epoch+1}/{num_epochs} Loss: {np.mean(running_loss):.4f} Acc: {running_corrects/len(train_dataset):.4f}')

# test model

model.eval()
running_loss = []
running_corrects = 0
cf = np.zeros((3, 3))
for images, labels in tqdm(test_loader):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    # calculate accuracy
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)

    # calculate loss
    running_loss.append(loss.item())

    # calculate confusion matrix
    cf += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=[0, 1, 2])

print(f'Test Loss: {np.mean(running_loss):.4f} Acc: {running_corrects/len(test_dataset):.4f}')
print(f'Confusion Matrix: {cf}')

# make plot (loss)

plt.plot(list(map(lambda x: x.cpu().numpy(), loss_history)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss.png')

# make plot (accuracy)

plt.plot(list(map(lambda x: x.cpu().numpy(), acc_history)))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.savefig('accuracy.png')