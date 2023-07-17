---
layout: post
title: triton系列：client 安装
category: nvidia
tags: triton
keywords: triton
description:
---

**https://github.com/triton-inference-server/client**

## 1. NGC 镜像自带

```bash
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk

docker run --gpus all --rm -it --net host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk

# inside container
perf_analyzer -m <model>
```

- exec: /workspace/install/bin
- head: /workspace/install/include
- library: /workspace/install/lib
- python wheel: /workspace/install/python

**重要提示：** 当使用 Docker 容器运行 server 或 client 并使用 [CUDA 共享内存功能](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md#cuda-shared-memory) 时 ，需要在启动容器时添加 `--pid host` 参数。因为 `CUDA IPC API` 要求导出指针的源和目标的 PID 不同。否则，Docker 会启用 PID 命名空间，这可能会导致源 PID 和目标 PID 相等。当两个容器都以非交互模式启动时，将始终观察到该错误。

## 2. pip 安装

GRPC 和 HTTP client库可以使用 python pip安装。

```bash
# 安装 HTTP/REST 和 GRPC 库
pip install tritonclient[all]

# 安装 HTTP/REST 库
pip install tritonclient[http]
```

安装包的组件：

- http
- grpc [ `service_pb2`, `service_pb2_grpc`, `model_config_pb2` ]
- utils [ linux distribution will include `shared_memory` and `cuda_shared_memory`]

该软件包的 Linux 版本还包括 [perf_analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md) 二进制文件。perf_analyzer 二进制文件是在 Ubuntu 20.04 上构建的，可能无法在其他 Linux 发行版上运行。要运行 perf_analyzer，必须安装以下依赖项：

```bash
apt update
apt install libb64-dev
```

*Windows 上的安装不包括 perf_analyzer、shared_memory/cuda_shared_memory 组件。*

## 3. 下载压缩包

client 库和 perf_analyzer 可执行文件可以从与您感兴趣的版本相对应的 [Triton GitHub release page](https://github.com/triton-inference-server/server/releases) 下载。例如 v2.3.0_ubuntu2004.clients.tar.gz。

perf_analyzer 二进制文件是在 Ubuntu 20.04 上构建的，可能无法在其他 Linux 发行版上运行。要使用 C++ 库或 perf_analyzer 可执行文件，您必须安装一些依赖项。

```bash
apt-get update

apt-get install curl libcurl4-openssl-dev libb64-dev
```

## 4. 源码安装

### 非 Windows 系统

使用cmake配置构建。您应该根据您正在使用和想要构建的 Triton 客户端组件来调整标志。例如，如果您想使用 Triton C API 构建 Perf 分析器，您可以使用 `-DTRITON_ENABLE_PERF_ANALYZER=ON`、`-DTRITON_ENABLE_PERF_ANALYZER_C_API=ON`. 您还可以使用`TRITON_ENABLE_PERF_ANALYZER_TFS`和`TRITON_ENABLE_PERF_ANALYZER_TS`标志在性能分析器中分别启用/禁用对 TensorFlow Serving 和 TorchServe 后端的支持。

以下命令演示了如何构建具有所有功能的客户端：

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=ON -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON -DTRITON_ENABLE_PERF_ANALYZER_TFS=ON -DTRITON_ENABLE_PERF_ANALYZER_TS=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DTRITON_ENABLE_PYTHON_GRPC=ON -DTRITON_ENABLE_JAVA_HTTP=ON -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON ..
```

如果您在发布分支（或基于发布分支的开发分支）上进行构建，则还必须使用其他 cmake 参数来指向客户端构建所依赖的存储库的发布分支。例如，如果您正在构建 r21.10 客户端分支，那么您需要使用以下附加 cmake 标志：

```bash
-DTRITON_COMMON_REPO_TAG=r21.10
-DTRITON_THIRD_PARTY_REPO_TAG=r21.10
-DTRITON_CORE_REPO_TAG=r21.10
```

然后使用make构建客户端和示例。

```bash
make cc-clients python-clients java-clients
```

构建完成后，可以在安装目录中找到库和示例。

### windows 系统

要构建客户端，您必须安装适当的 C++ 编译器和构建所需的其他依赖项。最简单的方法是创建[Windows min Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#windows-10-min-container)映像 ，并在从该映像启动的容器中执行构建。

```bash
docker run  -it --rm win10-py3-min powershell
```

无需使用 Docker 或 win10-py3-min 容器进行构建，但如果不这样做，则必须在主机系统上安装适当的依赖项。

接下来使用 *cmake* 配置构建。如果您不是在 win10-py3-min 容器中构建，那么您可能需要在以下命令中调整 `CMAKE_TOOLCHAIN_FILE` 位置。

```bash
mkdir build
cd build
cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_TOOLCHAIN_FILE='/vcpkg/scripts/buildsystems/vcpkg.cmake' -DCMAKE_INSTALL_PREFIX=install -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PYTHON_GRPC=ON -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_EXAMPLES=ON -DTRITON_ENABLE_TESTS=ON ..
```

如果您在发布分支（或基于发布分支的开发分支）上进行构建，则还必须使用其他 cmake 参数来指向客户端构建所依赖的存储库的发布分支。例如，如果您正在构建 r21.10 客户端分支，那么您需要使用以下附加 cmake 标志：

```bash
-DTRITON_COMMON_REPO_TAG=r21.10
-DTRITON_THIRD_PARTY_REPO_TAG=r21.10
-DTRITON_CORE_REPO_TAG=r21.10
```

然后使用msbuild.exe进行构建。

```bash
$ msbuild.exe cc-clients.vcxproj -p:Configuration=Release -clp:ErrorsOnly
$ msbuild.exe python-clients.vcxproj -p:Configuration=Release -clp:ErrorsOnly
```

构建完成后，可以在安装目录中找到库和示例。

