---
layout: post
title: Pytorch官方教程(四)—Transfer Learning Tutorial
category: Framework
tags: pytorch
keywords: pytorch tutorial
description:
---


迁移学习主要的两种应用场景：

- 微调卷积网络：使用预训练模型初始化网络，而不是随机初始化，比如在imagenet 1000数据集上训练的网络，剩余的正常训练。
- 卷积网络作为固定的特征提取器：冻结除最后一个全连接层外所有网络的权值。最后一个全连接层被替换为一个具有随机权重的新层，并且只训练这个层。


```python
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
```

# Load Data

使用 torchvision 和 torch.utils.data 加载数据。

训练一个蚂蚁和蜜蜂分类的模型。训练图片每个类有 120 张图片。验证集每个类有 75 张图片。通常，如果从零开始训练，这是一个非常小的数据集。使用转移学习，模型会具有更好的泛化行。

这个数据集是imagenet的一个非常小的子集。


```python
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
```

## Visualize a few images

可视化部分训练集的图片以便理解数据增强。


```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
```

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/12.png">

</center>

# Training the model

定义模型训练的通用函数。需要声明：

- 学习率的调度
- 保存最好的模型

下面的代码中，参数 scheduler 是来自 torch.optim.lr_scheduler 的学习率调度的对象。


```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
```

## Visualizing the model predictions

显示预测的泛化函数


```python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```

# Finetuning the convnet

加载预训练的模型并重置最后的全连接层。


```python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

## Train and evaluate

在 CPU 上需 15-25 分钟. 在 GPU 不需要 1 分钟。


```python
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

    Epoch 0/24
    ----------
    train Loss: 0.6469 Acc: 0.6393
    val Loss: 0.2514 Acc: 0.9020

    ...

    Epoch 24/24
    ----------
    train Loss: 0.3563 Acc: 0.8320
    val Loss: 0.2286 Acc: 0.9020

    Training complete in 1m 5s
    Best val Acc: 0.954248


# ConvNet as fixed feature extractor

冻结除最后一层外的全部网络。设置参数 requires_grad == False 冻结参数，在反向传播时不会计算梯度。


```python
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

## Train and evaluate

网络的大部分梯度不需要计算，只需要计算前向传播。


```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

    Epoch 0/24
    ----------
    train Loss: 0.5829 Acc: 0.6967
    val Loss: 0.3998 Acc: 0.8039

    ...

    Epoch 24/24
    ----------
    train Loss: 0.3436 Acc: 0.8320
    val Loss: 0.2410 Acc: 0.9020

    Training complete in 0m 53s
    Best val Acc: 0.954248



```python
visualize_model(model_conv)

plt.ioff()
plt.show()
```

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/13.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/14.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/15.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/16.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/17.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/18.png">

</center>
