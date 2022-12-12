---
layout: post
title: Pytorch Tricks
category: Framework
tags: pytorch
keywords: trick
description:
---

## 1. In-place Operate

会改变 tensor 的函数操作会用一个下划线后缀来标示。比如，torch.FloatTensor.abs_() 会在原地计算绝对值，并返回改变后的tensor，而 tensor.FloatTensor.abs() 将会在一个新的 tensor 中计算结果。

## 2. Dynamic Graphs

PyTorch 采用动态计算图，而 TensorFlow 采用静态计算图

- 静态计算图：只对计算图定义一次，而后会多次执行这个计算图。

    可以预先对计算图进行优化，融合一些计算图上的操作，并且方便在分布式多GPU或多机的训练中优化模型。

- 动态计算图：每执行一次都会重新定义一张计算图。

    控制流就像Python一样，更容易被人接受，可以方便的使用for，if等语句来动态的定义计算图，并且调试起来较为方便。

## 3. net.zero_grad()

再利用 loss.backward() 计算梯度之前，需要先清空已经存在的梯度缓存（因为PyTorch是基于动态图的，每迭代一次就会留下计算缓存，到一下次循环时需要手动清楚缓存），如果不清除的话，梯度就会**累加**（注意不是覆盖）。

```python
net.zero_grad()  # 清除缓存
print(net.conv1.bias.grad) # tensor([0., 0., 0., 0., 0., 0.])

loss.backward()

print(net.conv1.bias.grad) # tensor([ 0.0181, -0.0048, -0.0229, -0.0138, -0.0088, -0.0107])
```

## 4. reshape & view

view 可以看成，先按行展开成一维 tensor 再转换成所需维度的 tensor。只能作用在连续的内存空间上. 并且不会对 tensor 进行复制。当它作用在非连续内存空间的 tensor 上时，会报错。
reshape 可以作用在任何空间上，并且会在需要的时候创建 tenosr 的副本。

## 5. tensor.detach() & tensor.data

PyTorch 0.4中，.data 仍保留，但建议使用 .detach()，区别在于 .data 返回和 x 的相同数据 tensor，但不会加入到 x 的计算历史里，且 require s_grad = False，这样有些时候是不安全的，因为 x.data 不能被 autograd 追踪求微分。.detach() 返回相同数据的 tensor，且 requires_grad=False，但在进行反向传播的时候，能通过 in-place 操作报告给 autograd。

**tensor.data**

```python
>>> a = torch.tensor([1,2,3.], requires_grad =True)
>>> out = a.sigmoid()
>>> c = out.data
>>> c.zero_()
tensor([ 0., 0., 0.])

>>> out                   #  out的数值被c.zero_()修改
tensor([ 0., 0., 0.])

>>> out.sum().backward()  #  反向传播
>>> a.grad                #  这个结果很严重的错误，因为out已经改变了
tensor([ 0., 0., 0.])
```

**tensor.detach()**

```python
>>> a = torch.tensor([1,2,3.], requires_grad =True)
>>> out = a.sigmoid()
>>> c = out.detach()
>>> c.zero_()
tensor([ 0., 0., 0.])

>>> out                   #  out的值被c.zero_()修改 !!
tensor([ 0., 0., 0.])

>>> out.sum().backward()  #  需要原来out得值，但是已经被c.zero_()覆盖了，结果报错
RuntimeError: one of the variables needed for gradient
computation has been modified by an
```

## 6. 指定 GPU

终端

```bash
CUDA_VISIBLE_DEVICES=1 python my_script.py
```

代码文件

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```