---
layout: post
title: Pytorch官方教程(五)—Saving and Loading Models
category: Framework
tags: pytorch
keywords: pytorch tutorial
description:
---


# Core Function

## torch.save

使用 pickle 实现对象序列化并保存到磁盘。模型、tensor和所有类型的字典对象都可以使用这个函数保存。

```python
torch.save(obj, f, pickle_module=<module '...'>, pickle_protocol=2)
```
- obj：保存对象
- f：类文件对象 (必须实现写和刷新)或一个保存文件名的字符串
- pickle_module：用于 pickling 元数据和对象的模块
- pickle_protocol：指定 pickle protocal 可以覆盖默认参数

**e.g.**

```python
# Save to file
x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, 'tensor.pt')

# Save to io.BytesIO buffer
buffer = io.BytesIO()
torch.save(x, buffer)
```

## torch.load

torch.load() 使用 Python 的 解压工具（unpickling）来反序列化 pickled object 到对应存储设备上。首先在 CPU 上对压缩对象进行反序列化并且移动到它们保存的存储设备上，如果失败了（如：由于系统中没有相应的存储设备），就会抛出一个异常。用户可以通过 register_package 进行扩展，使用自己定义的标记和反序列化方法。

```python
torch.load(f, map_location=None, pickle_module=<module 'pickle' from '...'>)
```
- f：类文件对象 (返回文件描述符)或一个保存文件名的字符串
- map_location：一个函数或字典规定如何映射存储设备
- pickle_module：用于 unpickling 元数据和对象的模块 (必须匹配序列化文件时的 pickle_module )

**e.g.**

```python
torch.load('tensors.pt')

# Load all tensors onto the CPU
torch.load('tensors.pt', map_location=torch.device('cpu'))

# Load all tensors onto the CPU, using a function
torch.load('tensors.pt', map_location=lambda storage, loc: storage)

# Load all tensors onto GPU 1
torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))

# Map tensors from GPU 1 to GPU 0
torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})

# Load tensor from io.BytesIO object
with open('tensor.pt') as f:
    buffer = io.BytesIO(f.read())
torch.load(buffer)
```

## torch.nn.Module.load_state_dict

使用 state_dict 反序列化模型参数字典。用来加载模型参数。将 state_dict 中的 parameters 和 buffers 复制到此 module 及其子节点中。

```python
torch.nn.Module.load_state_dict(state_dict, strict=True)
```

- state_dict(dict)：保存 parameters 和 persistent buffers 的字典
- strict(bool, optional)：state_dict 中的 key 是否和 model.state_dict() 返回的 key 一致。


# What is a state_dict?

## state_dict

字典对象，将每个层映射到参数 tensor。key，具有可学习参数（卷积层，线性层等）和已注册缓冲区（batchnorm running_mean）的层。

## torch.nn.Module.state_dict

```python
torch.nn.Module.state_dict(destination=None, prefix='', keep_vars=False)
```

返回一个包含模型状态信息的字典。包含参数（weighs and biases）和持续的缓冲值（如：观测值的平均值）。只有具有可更新参数的层才会被保存在模型的 state_dict 数据结构中。


## torch.optim.Optimizer.state_dict

```python
torch.optim.Optimizer.state_dict()
```
返回一个包含优化器状态信息的字典。包含两个 key：

- state：字典，保存当前优化器的状态信息。不同优化器内容不同。
- param_groups：字典，包含所有参数组（eg：超参数）。

**Example:**


```python
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
```


```python
# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print('{0:<50}{1:<40}{2:<10}{3:<10}'.format(param_tensor,
                  str(model.state_dict()[param_tensor].size()),
                  str(model.state_dict()[param_tensor].device),
                  str(model.state_dict()[param_tensor].requires_grad)))

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
```

    Model's state_dict:
    conv1.weight                                      torch.Size([6, 3, 5, 5])                cpu       False
    conv1.bias                                        torch.Size([6])                         cpu       False
    conv2.weight                                      torch.Size([16, 6, 5, 5])               cpu       False
    conv2.bias                                        torch.Size([16])                        cpu       False
    fc1.weight                                        torch.Size([120, 400])                  cpu       False
    fc1.bias                                          torch.Size([120])                       cpu       False
    fc2.weight                                        torch.Size([84, 120])                   cpu       False
    fc2.bias                                          torch.Size([84])                        cpu       False
    fc3.weight                                        torch.Size([10, 84])                    cpu       False
    fc3.bias                                          torch.Size([10])                        cpu       False
    Optimizer's state_dict:
    state 	 {}
    param_groups 	 [{'lr': 0.001, 'params': [140049757854384, 140049757854960, 140049757855032, 140049757855104, 140049757401160, 140049757401232, 140049757401304, 140049757401376, 140049757401448, 140049757401520], 'weight_decay': 0, 'momentum': 0.9, 'nesterov': False, 'dampening': 0}]


# Saving & Loading Model for Inference

## Save/Load state_dict (Recommended)

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```

保存推理模型时，只需要保存训练好的模型的学习参数。使用 torch.save() 保存模型的 state_dict，将为你后续恢复模型提供更大的灵活性，推荐使用该方法保存模型。

文件扩展名：.pt / .pth

在运行推理之前，必须调用 model.eval() 将 dropout 和 BN 设置为评估模式。否则，推理的结果会是错误的。

load_state_dict() 接收的是一个字典对象，而不是保存的对象路径。这意味着在将保存的state_dict传递给 load_state_dict() 函数之前必须反序列化它。例如，不能使用 model.load_state_dict(PATH) 加载。

## Save/Load Entire Model

**Save:**

```python
torch.save(model, PATH)
```

**Load:**

```python
# Model class must be defined somewhere
model = torch.load(PATH)
model.eval()
```

这种 save/load 过程使用最直观的语法并涉及最少量的代码。以这种方式保存模型将使用 Python 的 pickle 模块保存整个模块。缺点是，保存模型时序列化的数据绑定到特定类以及使用确切的目录结构，这是因为pickle不保存模型类本身。相反，它会在加载时保存包含类的文件路径。因此，在其他项目中或在重构之后使用时，代码可能会以各种方式中断。

文件扩展名：.pt / .pth

在运行推理之前，必须调用 model.eval() 将 dropout 和 BN 设置为评估模式。否则，推理的结果会是错误的。

# Saving & Loading a General Checkpoint for Inference and/or Resuming Training

**Save:**

```python
# 先将需要保存的数据组织成字典的形式，然后用torch.save()保存
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        ...
        }, PATH)
```

**Load:**

```python
# 先初始化模型, 在利用 torch.load() 函数加载需要的数据
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()
```

在保存用于推理或恢复训练的 checkpoint 时，不能仅保存模型的 state_dict。保存优化器的 state_dict 也很重要，因为它包含模型训练时更新的缓冲区和参数。其他可能需要保存的内容如epoch、loss、torch.nn.Embedding layers等。

要保存多个部件，把它们整理到字典中，然后使用 torch.save() 序列化字典。

文件扩展名：.tar

加载时，首先要初始化模型和优化器，然后使用 torch.load() 在本地加载字典。可通过简单地访问字典来轻松查询已保存的内容。

在运行推理之前，调用 model.eval() 将 dropout 和 BN 设置为评估模式。否则，推理的结果是错误的。在运行恢复之前，调用 model.train() 将模型设置为训练模式。

# Saving Multiple Models in One File

**Save:**

```python
torch.save({
        'modelA_state_dict': modelA.state_dict(),
        'modelB_state_dict': modelB.state_dict(),
        'optimizerA_state_dict': optimizerA.state_dict(),
        'optimizerB_state_dict': optimizerB.state_dict(),
        ...
        }, PATH)
```

**Load:**

```python
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()
```

保存由多个torch.nn.Modules组成的模型时，例如 GAN，seq2seq 模型或模型集合，可以采用与保存常规检查点相同的方法。也就是说，保存每个模型的 state_dict 和相应的优化器的 state_dict。如前所述，也可以保存任何其他内容，只要将它们附加到字典中，就帮助模型恢复训练。

文件扩展名：.tar

加载模型，首先要初始化模型和优化器，然后使用 torch.load() 本地加载字典。可通过简单地访问字典来轻松查询已保存的内容。

在运行推理之前，调用 model.eval() 将 dropout 和 BN 设置为评估模式。否则，推理的结果是错误的。在运行恢复之前，调用 model.train() 将模型设置为训练模式。

# Warmstarting Model Using Parameters from a Different Model

**Save:**

```python
torch.save(modelA.state_dict(), PATH)
```

**Load:**

```python
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)
```

在转移学习或训练新的复杂模型时，部分加载模型或加载部分模型是常见的情况。利用训练好的参数，即使只有少数可用，也有助于加速训练过程，并且比从头开始训练收敛的更快。

无论是从缺省 key 的 state_dict 部分加载，还是从冗余 key 的 state_dict 加载，都可以设置 load_state_dict() 的参数 strict = False 来忽略不匹配的 key。

如果要将参数从一个层加载到另一个层，但某些键不匹配，只需更改加载的 state_dict 的 key 的名称，以匹配要加载模型中的 key。

# Saving & Loading Model Across Devices

## Save on GPU, Load on CPU

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
# 通过 torch.load() 中的 map_location 参数指定模型加载 state_dict 的设备
device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
```

在 CPU 上加载 GPU 训练的模型时，将 torch.load() 中的 map_location 参数设置为 torch.device('cpu')。在这种情况下，使用 map_location 参数将 tensor 的存储位置动态映射到 CPU 上。

## Save on GPU, Load on GPU

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

在 GPU 上加载 GPU 训练保存的模型时，只需用 model.to(torch.device('cuda')) 将初始化模型转换为 cuda 优化模型。此外，确保模型输入的数据使用 .to(torch.device('cuda'))。

注意：调用 my_tensor.to(device) 会在 GPU 上返回 my_tensor 的新副本，不会覆盖 my_tensor。因此，使用 my_tensor = my_tensor.to(torch.device('cuda')) 手动覆盖。

## Save on CPU, Load on GPU

**Save:**

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
device = torch.device("cuda")
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
```

在 GPU 上加载 CPU 训练保存的模型时，将 torch.load() 函数的 map_location 参数 设置为 cuda:device_id。这种方式将模型加载到指定设备。下一步，确保调用 model.to(torch.device('cuda')) 将模型参数 tensor 转换为 cuda tensor。最后，确保模型输入使用 .to(torch.device('cuda')) 为 cuda 优化模型准备数据。

注意：调用 my_tensor.to(device) 会在 GPU 上返回 my_tensor 的新副本，不会覆盖 my_tensor。因此，使用 my_tensor = my_tensor.to(torch.device('cuda')) 手动覆盖。

## Saving torch.nn.DataParallel Models

```python
torch.save(model.state_dict(), PATH)
```

**Load:**

```python
# Load to whatever device you want
```

torch.nn.DataParallel 是支持模型使用 GPU 并行的封装器。要保存一个一般的 DataParallel 模型， 请保存 model.module.state_dict()。这种方式，可以灵活地以任何方式加载模型到任何设备上。
