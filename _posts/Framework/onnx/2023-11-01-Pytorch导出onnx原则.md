---
layout: post
title: pytorch 导出 onnx 原则
category: Framework
tags: onnx
keywords: onnx
description:
---

## 原则

1. 对于任何用到shape、size返回值的参数时，例如：`tensor.view(tensor.size(0), -1)`，`B,C,H,W = x.shape` 这类操作，避免直接使用 tensor.size 的返回值，而是加上int转换，`tensor.view(int(tensor.size(0)), -1)`, `B,C,H,W = map(int, x.shape)`，断开跟踪。

2. 对于`nn.Upsample`或`nn.functional.interpolate`函数，一般使用scale_factor指定倍率，而不是使用size参数指定大小。如果源码中就是插值为固定大小，则该条忽略。

3. 关于 batch 动态 shape 还是宽高动态 shape

    - 对于`reshape`、`view`操作时，-1的指定请放到batch维度。其他维度计算出来即可。batch维度禁止指定为大于-1的明确数字。如果是一维，那么直接指定为-1就好。
    - torch.onnx.export指定dynamic_axes参数，并且只指定batch维度，禁止其他动态

4. 使用opset_version=11，不要低于11

5. 避免使用inplace操作，例如`y[…, 0:2] = y[…, 0:2] * 2 - 0.5`，可以采用如下代码代替 `tmp = y[…, 0:2] * 2 - 0.5; y = torch.cat((y[..., 2:], tmp), dim = -1)`

6. 尽量少的出现5个维度，例如ShuffleNet Module，可以考虑合并wh避免出现5维

7. 尽量把让后处理部分在onnx模型中实现，降低后处理复杂度。比如在目标检测网络中最终输出设置为xywh或者xyxy，而不是一个中间结果。
