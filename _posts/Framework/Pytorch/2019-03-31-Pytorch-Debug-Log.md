---
layout: post
title: Pytorch Debug Log
category: Framework
tags: pytorch
keywords: debug
description:
---

## 1. 模型与参数的类型不符

Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

TensorIterator expected type torch.cuda.FloatTensor but got torch.FloatTensor

解决方法

1. 在每一处新建立的tensor上将其手动移动到 cuda 或者 cpu 上
2. 设置默认设备和类型

    ```python
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ```

