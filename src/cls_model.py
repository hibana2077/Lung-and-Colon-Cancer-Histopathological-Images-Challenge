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

from sklearn.metrics import confusion_matrix

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

# model = timm.create_model('convnext_base', num_classes=3)
model = timm.create_model('resnet50', num_classes=3)

# define loss function and optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = tioptim.Lookahead(timm.optim.AdamW(model.parameters(), lr=1e-3))

# train model
num_epochs = 12
loss_history = []
acc_history = []
test_loss_history = []
test_acc_history = []

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
    print(f'Epoch {epoch+1}/{num_epochs} Loss: {np.mean(running_loss):.4f} Acc: {running_corrects/len(train_dataset):.4f}', end=' ')

    running_test_loss = []
    running_test_corrects = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # calculate accuracy
        _, preds = torch.max(outputs, 1)
        running_test_corrects += torch.sum(preds == labels.data)

        # calculate loss
        running_test_loss.append(loss.item())

    test_acc_history.append(running_test_corrects/len(test_dataset))
    test_loss_history.append(np.mean(running_test_loss))
    print(f'Test Loss: {np.mean(running_test_loss):.4f} Acc: {running_test_corrects/len(test_dataset):.4f}')

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
print(f'Elapsed Time: {time.time()-start_time:.2f}s')

# save model (onnx)
dummy_input = torch.randn(1, 3, 256, 256).to(device)
torch.onnx.export(model, dummy_input, 'model.onnx')
print('ONNX model saved')

# save model (pth)
torch.save(model, 'model.pth')
print('Torch model saved')

# make plot of loss and accuracy

plt.figure()
plt.plot(loss_history, label='train loss', color='red')
plt.plot(test_loss_history, label='test loss', color='blue')
plt.legend()
plt.savefig('loss.png')

plt.figure()
plt.plot(acc_history, label='train acc', color='red')
plt.plot(test_acc_history, label='test acc', color='blue')
plt.legend()
plt.savefig('acc.png')