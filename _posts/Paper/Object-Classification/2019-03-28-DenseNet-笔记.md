---
layout: post
title: DenseNet 笔记
category: Paper
tags: 图像分类
keywords: densenet
description:
---

## 1. Densenet vs Resnet

ResNet：每个层与前面的某层（一般是2~3层）短路连接（shortcuts、skip connection）在一起，连接方式是通过元素级相加（element-wise addition）。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/1.png">

</center>

DenseNet：每个层都会与前面所有层在 channel 维度上连接（channel-wise concatenation/特征重用）在一起（各个层的特征图大小是相同的），并作为下一层的输入。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/2.png">

</center>

传统的网络, l 层的输出为：
$$x_l = H_l(x_{l-1})$$

ResNet，增加了来自上一层输入的identity函数：
$$x_l = H_l(x_{l-1}) + x_{l-1}$$

DenseNet，连接前面所有层作为输入：
$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

其中，上面的 $H_l(\cdot)$ 代表是非线性转化函数（non-liear transformation），它是一个组合操作，其可能包括一系列的BN(Batch Normalization)，ReLU，Pooling及Conv操作。注意这里 l 层与 l-1 层之间可能实际上包含多个卷积层。

DenseNet的前向过程如下图所示，可以更直观地理解其密集连接方式，比如 $h_3$ 的输入不仅包括来自 $h_2$ 的 $x_2$ ，还包括前面两层的 $x_1$ 和 $x_2$ ，它们是在channel维度上连接在一起的。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/3.png">

</center>


## 2. 网络结构

DenseNet 的网络结构主要由 DenseBlock 和 Transition 组成。

### 2.1 DenseBlock

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/4.png">

</center>

DenseBlock 是包含很多层的模块，每个层的特征图大小相同，可以在channel维度上连接，层与层之间采用密集连接方式。DenseBlock 中的非线性组合函数 $H(\cdot)$ 采用的是 BN + ReLU + 3x3 Conv 的结构。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/5.png">

</center>

#### Growth Rate

所有 DenseBlock 中各个层卷积之后均输出 k 个特征图，即得到的特征图的channel 数为 k。 k 在 DenseNet 称为 growth rate，是一个超参数。一般情况下使用较小的 k（比如12），就可以得到较佳的性能。假定输入层的特征图的 channel 数为 $k_{0}$ ，那么 l 层输入的 channel 数为 $k_{0}+k\left(l - 1\right)$ 。

#### Bottleneck Layer

DenseBlock 内部可以采用 bottleneck 层来减少计算量，因为后面层的输入会非常大。bottleneck 主要是原有的结构中增加 1x1 Conv，即 BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv，称为 **DenseNet-B** 结构。其中1x1 Conv得到 4k 个特征图它起到的作用是降低特征数量。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/6.png">

</center>

### 2.2 Transition

Transition 模块是连接两个相邻的 DenseBlock，并且通过 Pooling 使特征图大小降低。Transition 层包括一个 1x1 的卷积和 2x2 的 AvgPooling，结构为BN+ReLU+1x1 Conv+2x2 AvgPooling。

#### Compression Factor

假定 Transition 的上接 DenseBlock 得到的特征图 channels 数为 m ，Transition 层可以产生 $\lfloor\theta m\rfloor$ 个特征（通过卷积层），其中 $\theta \in (0,1]$ 是压缩系数（compression rate）。当压缩系数小于1时，可以起到压缩模型的作用，这种结构称为DenseNet-C，文中使用 $\theta=0.5$ 。对于使用 bottleneck 层的 DenseBlock 结构和压缩系数小于 1 的 Transition 组合结构称为 **DenseNet-BC**。


## 3. Densenet 优势

### 3.1 强梯度流

误差信号可以更直接地传播到早期的层中。这是一种隐含的深度监督（deep supervision），因为早期的层可以从最终的分类层直接获得监督。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/7.png">

</center>

### 3.2 参数和计算效率

对于每个层，RetNet 中的参数与c×c成正比，而 DenseNet 中的参数与1×k×k成正比。由于 k<<C, 所以 DenseNet 比 ResNet 的size更小。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/8.png">

</center>

### 3.3 更加多样化的特征

由于 DenseNet 中的每一层都接收前面的所有层作为输入，因此特征更加多样化，并且倾向于有更丰富的模式。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/9.png">

</center>

### 3.4 保持低复杂度特征

在标准 ConvNet 中，分类器使用最复杂的特征。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/10.png">

</center>

在 DenseNet 中，分类器使用所有复杂级别的特征，使得具有非常好的抗过拟合性能，尤其适合于训练数据相对匮乏的应用。利用浅层复杂度低的特征，因而更容易得到一个光滑的具有更好泛化性能的决策函数。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/DenseNet/11.png">

</center>


