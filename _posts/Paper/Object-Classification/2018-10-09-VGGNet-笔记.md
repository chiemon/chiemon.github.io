---
layout: post
title: VGGNet 笔记
category: Paper
tags: 图像分类
keywords: vggnet
description:
---

## 1. 网络介绍

* ImageNet-2104竞赛第二，是网络改造的首选基础网络（图片描述，图片问答）；
* 一个大的卷积核分解为连续多个小卷积核；
* 应用了核分解的思想：将7X7核->3个3X3核（由ReLU连接）；
* 对应的参数数量由49通道数变为27通道数；
* 优点是减少参数，显存可用的容量对应就多了，降低了计算，增加了深度；
* 继承了AlexNet结构的特点：简单和有效；

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/VGG/1.png">

</center>

## 2. VGG16 - weight layers

- 输入图像大小：224X224X3
- 2 个3X3相当于 1 个5X5；
- 3 个3X3相当于 1 个7X7；
- 共 5 个池化层，3 个FC层；
- 拆解时通道数不便，依次是：64, 128, 256, 512, 512；

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/VGG/2.png">

</center>

VGGNet探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3\*3的小型卷积核和2\*2的最大池化层，VGGNet成功地构筑了16~19层深的卷积神经网络。
VGG-16和VGG-19结构如下：

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/VGG/3.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/VGG/4.png">

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/VGG/5.png">

</center>

## 总结：
- VGG-16网络中的16代表的含义为：含有参数的有16个层，共包含参数约为1.38亿。

- VGG-16网络结构很规整，没有那么多的超参数，专注于构建简单的网络，都是几个卷积层后面跟一个可以压缩 图像大小的池化层。即：

    全部使用3\*3的小型卷积核和2\*2的最大池化层。
    卷积层：CONV=3*3 filters, s = 1, padding = same convolution。
    池化层：MAX_POOL = 2\*2 , s = 2。

- 优点：简化了卷积神经网络的结构；

- 缺点：训练的特征数量非常大；

- 随着网络加深，图像的宽度和高度都在以一定的规律不断减小，每次池化后刚好缩小一半，信道数目不断增加一倍。