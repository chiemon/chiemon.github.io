---
layout: post
title: Pytorch Tensor Shape
category: Framework
tags: pytorch
keywords: tensor-shape
description:
---

## torch.flatten()

```python
# 展平一个连续范围的维度，输出类型为 Tensor
torch.flatten(input, start_dim=0, end_dim=-1) → Tensor
# Parameters：input (Tensor) – 输入为 Tensor
# start_dim (int) – 展平的开始维度
# end_dim (int) – 展平的最后维度
# example
# 一个 3x2x2 的三维张量
>>> t = torch.tensor([[[1, 2],
                       [3, 4]],
                      [[5, 6],
                       [7, 8]],
                      [[9, 10],
                       [11, 12]]])
# 当开始维度为0，最后维度为 -1，展开为一维
>>> torch.flatten(t)
tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
# 当开始维度为0，最后维度为 -1，展开为 3x4，也就是说第一维度不变，后面的压缩
>>> torch.flatten(t, start_dim=1)
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])
>>> torch.flatten(t, start_dim=1).size()
torch.Size([3, 4])
# 下面的和上面进行对比应该就能看出是，当锁定最后的维度的时候前面的就会合并
>>> torch.flatten(t, start_dim=0, end_dim=1)
tensor([[ 1,  2],
        [ 3,  4],
        [ 5,  6],
        [ 7,  8],
        [ 9, 10],
        [11, 12]])
>>> torch.flatten(t, start_dim=0, end_dim=1).size()
torch.Size([6, 2])
```

## torch.reshape()

```python
# reshape(*shape) → Tensor
# 参数：
# shape (tuple of python:ints or int...) – 想要转换的形状
# 返回具有与self相同的数据和元素数量但具有指定形状的张量。
# 如果 shape 与当前形状兼容，则此方法返回一个视图。
torch.reshape(input, shape) → Tensor
# 返回具有与 input 相同的数据和元素数量，但具有指定形状的张量。如果可能，返回的张量将
# 是 input 视图。否则，它将是 copy 的版本。连续输入和具有兼容步幅的输入可以在不复制的
# 情况下进行重塑，但是您不应该依赖复制与查看行为。
# 某个尺寸可能为 -1，在这种情况下，它是根据剩余尺寸和输入的元素数推断出来的。
# 参数：input (Tensor) – 原始的 shape
# shape (tuple of python:ints) – 需要的 shape
>>> a = torch.arange(4.)
>>> torch.reshape(a, (2, 2))
tensor([[ 0.,  1.],
        [ 2.,  3.]])
>>> b = torch.tensor([[0, 1], [2, 3]])
# 只有一个-1的话，就是直接展开
>>> torch.reshape(b, (-1,))
tensor([ 0,  1,  2,  3])
# 给定一个的时候，另外一个就能就算出来
>>> torch.reshape(b, (-1, 2))
tensor([[0, 1],
        [2, 3]])
>>> b = torch.tensor([[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[0, 1], [2, 3]]])
>>> torch.reshape(b, (-1, 2, -1))
RuntimeError: only one dimension can be inferred
>>> torch.reshape(b, (2, 2, -1))
tensor([[[0, 1, 2],
         [3, 0, 1]],
        [[2, 3, 0],
         [1, 2, 3]]])
```

## torch.Tensor.view()

```python
# view(*shape) → Tensor
# 返回一个新张量，其数据与 self 张量相同，但 shape 是不同。 返回的张量共享相同的数据，
# 并且必须具有相同数量的元素，但可能具有不同的大小。一个张量可以被 view，new view 的
# 尺寸必须与其原始尺寸和 stride 兼容，也就是说每个新视图尺寸必须是原始尺寸的子空间，
# 或者只能跨越原始的 d, d+1,..., d+k 这些要满足以下的连续性条件，
# stride[i]=stride[i+1]×size[i+1] 对于任意的 i=0，...,k-1
# 一个 tensor 必须是连续的,才能被 contiguous() 查看。
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 的位置是可以推断出来
>>> z.size()
torch.Size([2, 8])

>>> a = torch.randn(1, 2, 3, 4)
>>> a.size()
torch.Size([1, 2, 3, 4])
>>> b = a.transpose(1, 2)  # 交换第二维度和第三维度，改变了内存的位置
>>> b.size()
torch.Size([1, 3, 2, 4])
>>> c = a.view(1, 3, 2, 4)  # 不改变内存中的位置
>>> c.size()
torch.Size([1, 3, 2, 4])
>>> torch.equal(b, c) # 所以不是一个
False
>>> torch.equal(a, c) # 还是不是一个，不清楚官方给这个栗子想说明什么
False

# 顺便说一下 contiguous()
# 返回一个包含与 self 张量相同数据的连续张量。如果 self 张量是连续的，
# 则此函数返回 self 张量。其实就是返回一个结果。
# torch.transpose(input, dim0, dim1) → Tensor
# 返回张量，该张量是输入的转置版本。
# 给定的尺寸 dim0 和 dim1 被交换。
# 生成的张量与输入张量共享其基础存储，因此更改其中一个的内容将更改另一个的内容。
>>> x = torch.randn(2, 3)
>>> x
tensor([[ 1.0028, -0.9893,  0.5809],
        [-0.1669,  0.7299,  0.4942]])
# 第一维度和第二维度转换
>>> torch.transpose(x, 0, 1)
tensor([[ 1.0028, -0.1669],
        [-0.9893,  0.7299],
        [ 0.5809,  0.4942]])
```