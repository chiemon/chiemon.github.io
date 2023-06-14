---
layout: post
title: Pytorch官方教程(二)—Data Loading and Processing Tutorial
category: Framework
tags: pytorch
keywords: pytorch tutorial
description:
---


#  Introduction

在解决机器学习的问题时，大部分精力都花费在准备数据上。Pytorch 提供了很多工具来简化数据加载，使代码具有很好的可读性。

下面的教程我们将学习如何加载数据、预处理/增强数据。


```python
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
```

数据集：[人脸姿态](https://download.pytorch.org/tutorial/faces.zip)

存放路径：data/faces/

这个数据集实际上是使用 'dlib姿态估计' 对标记为 'face' 的 imagenet 图像生成的。

数据集附带一个 csv 文件，带有如下标注：

    image_name,part_0_x,part_0_y,part_1_x,part_1_y,part_2_x, ... ,part_67_x,part_67_y

    0805personali01.jpg,27,83,27,98, ... 84,134

    1084239450_e76e00b7e7.jpg,70,236,71,257, ... ,128,312

读取 CSV 并将标注放入(N, 2)的数组中，其中 N 是关键点的数量。


```python
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))
```

    Image name: person-7.jpg
    Landmarks shape: (68, 2)
    First 4 Landmarks: [[32. 65.]
     [33. 76.]
     [34. 86.]
     [34. 97.]]


显示图像和标注的关键点函数


```python
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
               landmarks)
plt.show()
```

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/3.png">

</center>

# Dataset class

torch.utils.data.Dataset 是表示抽象数据集的类。自定义数据集需继承 Dataset 并重写以下方法：

- __len__  len(dataset) 返回数据及大小
- __getitem__ 支持索引，例如 dataset[i] 可以用于得到第 i 个样本

为人脸关键点数据集创建数据类。__init__：读取 csv 文件，__getitem__：读取图片。这样会节省内存，因为并不是所有的图像都存储在内存中，而是按需读取。

数据集示例：{'image': image, 'landmarks': landmarks}。数据集接受一个可选的参数 transform，以便任何所需的处理都可以应用于示例。


```python
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

实例化类并遍历数据样本。显示前 4 个样本的大小并显示关键点。


```python
face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break
```

    0 (324, 215, 3) (68, 2)



<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/4.png">

</center>


    1 (500, 333, 3) (68, 2)



<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/5.png">

</center>


    2 (250, 258, 3) (68, 2)



<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/6.png">

</center>


    3 (434, 290, 3) (68, 2)



<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/7.png">

</center>


# Transforms

预处理代码，三个变换：

- Rescal：缩放图像
- RandomCrop：随机裁剪图像，数据增强
- ToTensor：将图片由 numpy 转化为 Torch（需要交换坐标轴）

将其写成可调用的类，而不是简单的函数，这样就不必每次调用 Transforms 时都传递参数。为此，我们仅需要实现 __call__ 方法并且如果需要也可以实现 __init__ 方法。可以像这样使用 transform：

```python
tsfm = Transform(params)
transformed_sample = tsfm(sample)
```

观察下面这些转换如何同时应用到图像和 landmark 的。


```python
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
```

# Compose transforms

将 transforms 应用于实例。假设我们想把图像的短边重设为256，然后随机裁剪一个224的正方形。例如，我们想组合 Rescale 和 RandomCrop 变换。torchvision.transforms.Compose 是一个允许我们这样做的简单可调用类。


```python
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()
```


<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/8.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/9.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/10.png">

</center>



# Iterating through the dataset

把这些都放在一起创建一个具有组合转换的数据集。总之，每次使用数据集样本时：

- 从文件中动态读取图像
- 转换读取的图像
- 由于其中一种转换是随机的，因此数据在抽样时得到了扩充

使用 for i in range 循环遍历创建的数据集


```python
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break
```

    0 torch.Size([3, 224, 224]) torch.Size([68, 2])
    1 torch.Size([3, 224, 224]) torch.Size([68, 2])
    2 torch.Size([3, 224, 224]) torch.Size([68, 2])
    3 torch.Size([3, 224, 224]) torch.Size([68, 2])


使用简单的 for 循环遍历数据会丢失很多的可用参数。特别的，如：

- Batching the data
- Shuffling the data
- Load the data in parallel using multiprocessing workers

torch.utils.data.DataLoader 是一个提供了所有这些可用参数的迭代器。下面这些用到的参数应该是明确的，第一个参数 collate_fn，指定需要如何对样本进行批处理。然而，默认的 collate 可以解决我们用到的大部分工作。


```python
dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
```

    0 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
    1 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
    2 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])
    3 torch.Size([4, 3, 224, 224]) torch.Size([4, 68, 2])


<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Pytorch/11.png">

</center>



# Afterword: torchvision

这个教程中，我们已经了解了如何编写、使用、转换和加载数据集。torchvision 包提供了一些共有数据集和转换。你甚至可能不需要编写自定义类。ImageFolder 是 torchvision 中更通用的一个数据集。它假设图片的目录结构如下：

    root/ants/xxx.png
    root/ants/xxy.jpeg
    root/ants/xxz.png
    .
    .
    .
    root/bees/123.jpg
    root/bees/nsdf3.png
    root/bees/asd932_.png

 'ants', 'bees'等是类标签。类似的通用变换也可以在 PIL.Image 上运行，如 RandomHorizontalFlip, Scale。可以使用这些来编写像这样的数据阅读器：


```python
import torch
from torchvision import transforms, datasets

data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
                                           transform=data_transform)
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
```
