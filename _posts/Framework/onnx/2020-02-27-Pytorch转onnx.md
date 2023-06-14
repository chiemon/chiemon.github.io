---
layout: post
title: pytorch -> onnx 问题汇总
category: Framework
tags: onnx
keywords: onnx
description:
---

## 1. 版本问题

PyTorch v1.0.1 Reshape 不支持报错 [Solution]

PyTorch v1.2.0 需要升级 cuda10.0 以上

- Ubuntu 16.04
- RTX2080TI, Driver Version: 410.79
- CUDA 10.0
- cudnn 7.6.3 (经测低版本如 7.5.0 无影响)
- pycuda 2019.1.2
- pytorch 1.3.1
- torchvision 0.4.2
- tensorrt 6.0.1.5
- python 3.6.9
- - 经测 ONNX 无法使用，建议使用 python 3.7.x
- - onnx 1.6.0
- - protobuf 3.9.2 (需要降级到 3.9.x，不然 onnx 会报 libprotobuf.so.20 的错)

## 2. 多输入问题

创建多个 dummy_input，然后利用一个 tuple，传入函数中

```python
dummy_input0 = torch.LongTensor(Batch_size, seg_length).to(torch.device("cuda"))
dummy_input1 = torch.LongTensor(Batch_size, seg_length).to(torch.device("cuda"))
dummy_input2 = torch.LongTensor(Batch_size, seg_length).to(torch.device("cuda"))
torch.onnx.export(model, (dummy_input0, dummy_input1, dummy_input2), filepath)
```

## 3. 索引

像 data[index] = new_data 这样的张量就地索引分配目前在导出中不受支持。解决这类问题的一种方法是使用算子散点，显式地更新原始张量。
就是像 tensorflow 的静态图，不能随便改变 tensor 的值，可以用 torch 的 scatter_ 方法解决

**错误的方式**

```python
def forward(self, data, index, new_data):
    data[index] = new_data          # 重新赋值
    return data
```

**正确的方式**

```python
def forward(self, data, index, new_data):
    new_data = new_data.unsqueeze(0)
    index = index.expand(1, new_data.size(1))
    data.scatter_(0, index, new_data)
    return data
```

## 4. ONNX export failed on ATen operator group_norm because torch.onnx.symbolic.group_norm does not exist

解决：~/anaconda3/envs/py36/lib/python3.6/site-packages/torch/onnx/symbolic.py

```python
@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    return g.op("ATen", input, weight, bias, num_groups_i=num_groups,
                eps_f=eps, cudnn_enabled_i=cudnn_enabled, operator_s="group_norm")
```

## 5. RuntimeError: ONNX export failed: Couldn’t export operator aten::adaptive_avg_pool2d

**原因**

因为 PyTorch 的网络中用了 torch.nn.AdaptiveAvgPool2d，ONNX 没有 avg_pool2d 操作。

目前 PyTorch2ONNX 流程中，ONNX 并不支持 AdaptivePooling 操作，该操作仅存于 PyTorch 中。

```python
self.global_average = nn.AdaptiveAvgPool2d((1, 1))
```

**解决办法①**

使用 AvgPool2d 替换 AdaptiveAvgPool2d

```python
self.global_average = nn.AvgPool2d(kernel_size=(7, 7),stride=(7, 7), ceil_mode=False)

# 这两个公式转换为标准的 Max/AvgPooling

# 只需要知道输入的input_size ，就可以推导出stride 与kernel_size ，从而替换为标准的Max/AvgPooling

stride = floor((input_size / (output_size)))
kernel_size = input_size − (output_size − 1) * stride
padding = 0
```

**解决办法②**

添加额外Option

```python
import torch
torch.onnx.export(model, ..., operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```

该方法只是阻止 ONNX 替换 PyTorch 的 OP、而是使用 ATen 的 OP 替换，PyTorch2ONNX 能通，但 ONNX2TRT 却不能通，原因是 ONNX phaser 识别不到非 ONNX 的OP。

**整报错信息**

```
UserWarning: ONNX export failed on adaptive_avg_pool2d because output size that are not factor of input size not supported

RuntimeError: ONNX export failed: Couldn't export operator aten::adaptive_avg_pool2d
```

## 6. RuntimeError: ONNX export failed: Couldn’t export operator aten::upsample_bilinear2d

近似地，应该与警告信息 UserWarning: ONNX export failed on upsample_bilinear2d because align_corners == True not supported 相关联。

**原因**

转换 ONNX 使用的版本较低，PyTorch.ONNX 不支持。另外，参考源码， torch.onnx.export 默认使用 opset_version=9

**解决办法**

警告信息已经完整说明，ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11.，因此将 ONNX 的导出代码中规定其版本，具体如下：

```python
import torch
torch.onnx.export(model, ..., opset_version=11)
```

**整报错信息**

```
UserWarning: You are trying to export the model with onnx:Upsample for ONNX opset version 9. This operator might cause results to not match the expected results by PyTorch.
ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. Attributes to determine how to transform the input were added in onnx:Resize in opset 11 to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).
We recommend using opset 11 and above for models using this operator.

UserWarning: ONNX export failed on upsample_bilinear2d because align_corners == True not supported

RuntimeError: ONNX export failed: Couldn't export operator aten::upsample_bilinear2d
```

## 7. Error: In node 2 (importGather): UNSUPPORTED_NODE: Assertion failed: !(data->getType() == nvinfer1::DataType::kINT32 && nbDims == 1) && “Cannot perform gather on a shape tensor!”

**原因**

"Cannot perform gather on a shape tensor!"，网络内部使用 x_size = x.size()[1:] 等类似操作，TensorRT 在 trace 的时候，会被解析成一个 shape layer的输出，获得一个 shape tensor，用 Netron 工具可视化就可以发现，对应的 node 2 实际上是个 Constant node，与预期不符。

**解决办法①**

不使用该操作

**解决办法②**

```python
x_size = torch.tensor(x.shape)[1:]
```

## 8. Error: In node 1 (importUpsample): UNSUPPORTED_NODE: Assertion failed: (nbDims >= 1) && (nbDims <= 3)

使用 Netron 工具可视化模型，找到对应的 node 1，就可以发现对应的是 F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False) 操作。

**原因**

目前 ONNX2TRT 的转换过程中，不支持 F.interpolate 的 bilinear 模式，只支持 linear 和 nearest。

**解决办法**

将所有的 bilinear 模式替换为 nearest 模式。