---
layout: post
title: onnxruntime opset
category: Framework
tags: opset
keywords: opset
description:
---

For more details, see [this page](https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md)

### onnxruntime

| ONNX Runtime release version | ONNX release version | ONNX opset version | ONNX ML opset version | Supported ONNX IR version | Windows ML Availability |
| ---------------------------- | -------------------- | ------------------ | --------------------- | ------------------------- | ----------------------- |
| 1.10.0                       | **1.10** down to 1.2 | 15                 | 2                     | 7                         | Windows AI 1.10+        |
| 1.9.0                        | **1.10** down to 1.2 | 15                 | 2                     | 7                         | Windows AI 1.9+         |
| 1.8.2                        | **1.9** down to 1.2  | 14                 | 2                     | 7                         | Windows AI 1.8+         |
| 1.8.1                        | **1.9** down to 1.2  | 14                 | 2                     | 7                         | Windows AI 1.8+         |
| 1.8.0                        | **1.9** down to 1.2  | 14                 | 2                     | 7                         | Windows AI 1.8+         |
| 1.7.0                        | **1.8** down to 1.2  | 13                 | 2                     | 7                         | Windows AI 1.7+         |
| 1.6.0                        | **1.8** down to 1.2  | 13                 | 2                     | 7                         | Windows AI 1.6+         |
| 1.5.3                        | **1.7** down to 1.2  | 12                 | 2                     | 7                         | Windows AI 1.5+         |
| 1.5.2                        | **1.7** down to 1.2  | 12                 | 2                     | 7                         | Windows AI 1.5+         |
| 1.5.1                        | **1.7** down to 1.2  | 12                 | 2                     | 7                         | Windows AI 1.5+         |
| 1.4.0                        | **1.7** down to 1.2  | 12                 | 2                     | 7                         | Windows AI 1.4+         |
| 1.3.1                        | **1.7** down to 1.2  | 12                 | 2                     | 7                         | Windows AI 1.4+         |
| 1.3.0                        | **1.7** down to 1.2  | 12                 | 2                     | 7                         | Windows AI 1.3+         |
| 1.2.0、1.1.2、1.1.1、1.1.0   | **1.6** down to 1.2  | 11                 | 2                     | 6                         | Windows AI 1.3+         |
| 1.0.0                        | **1.6** down to 1.2  | 11                 | 2                     | 6                         | Windows AI 1.3+         |
| 0.5.0                        | **1.5** down to 1.2  | 10                 | 1                     | 5                         | Windows AI 1.3+         |
| 0.4.0                        | **1.5** down to 1.2  | 10                 | 1                     | 5                         | Windows AI 1.3+         |
| 0.3.1、0.3.0                 | **1.4** down to 1.2  | 9                  | 1                     | 3                         | Windows 10 2004+        |
| 0.2.1、0.2.0                 | **1.3** down to 1.2  | 8                  | 1                     | 3                         | Windows 10 1903+        |
| 0.1.5、0.1.4                 | **1.3** down to 1.2  | 8                  | 1                     | 3                         | Windows 10 1809+        |

### ONNXRUNTIME-GPU DEPENDENCY

For more details, see [this page](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

| ONNX Runtime | CUDA   | cuDNN                                | Notes                                                        |
| ------------ | ------ | ------------------------------------ | ------------------------------------------------------------ |
| 1.10         | 11.4   | 8.2.4 (Linux)<br/>8.2.2.26 (Windows) | libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.1.51<br/>libcublas 11.6.1.51<br/>libcudnn 8.2.4 |
| 1.9          | 11.4   | 8.2.4 (Linux)<br/>8.2.2.26 (Windows) | libcudart 11.4.43<br/>libcufft 10.5.2.100<br/>libcurand 10.2.5.120<br/>libcublasLt 11.6.1.51<br/>libcublas 11.6.1.51<br/>libcudnn 8.2.4 |
| 1.8          | 11.0.3 | 8.0.4 (Linux)<br/>8.0.2.39 (Windows) | libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4 |
| 1.7          | 11.0.3 | 8.0.4 (Linux)<br/>8.0.2.39 (Windows) | libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4 |
| 1.5-1.6      | 10.2   | 8.0.3                                | CUDA 11 can be built from source                             |
| 1.2-1.4      | 10.1   | 7.6.5                                | Requires cublas10-10.2.1.243;<br/>cublas 10.1.x will not work |
| 1.0-1.1      | 10.0   | 7.6.4                                | CUDA versions from 9.1  up to 10.1<br/>cuDNN versions from 7.1 up to 7.4  should also work with Visual Studio 2017 |
