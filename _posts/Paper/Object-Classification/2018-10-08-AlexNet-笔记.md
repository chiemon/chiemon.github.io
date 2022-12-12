---
layout: post
title: AlexNet 笔记
category: Paper
tags: 图像分类
keywords: alexnet
description:
---

## 1. 网络介绍：

* ImageNet2012竞赛第一名；他标志着DNN深度学习革命的开始；
* 网络包含5个卷积层+3个全连接层；
* 60M个参数+650K个神经元；
* 2个分组——>2个GPU（3G，受限于当时硬件），训练时长一周，50x加速；
* 引入的新技术有：<br>
&emsp;
ReLU -- 非线性激活；<br>
&emsp;
Max pooling -- 池化；<br>
&emsp;
Dropout regularization -- 用于防止过拟合，在判断决策的FC层使用；

## 2. 模型网络框图：

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Alexnet/1.png">

</center>

输入图片大小理论上应为227X227X3（大小为227*227的RGB图）

每一层的结构：

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Alexnet/2.png">

</center>

其中LRN为局部响应归一化，具体解释可参考文章：
<http://blog.csdn.net/hduxiejun/article/details/70570086>

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Alexnet/3.png">

</center>