---
layout: post
title: TensorRT - Custom Layer
category: nvidia
tags: tensorrt
keywords: tensorrt
description:
---

## TensorRT

1. 创建 Builder：构造器，后面一系列参数，后面所有的内容，一个最基本的容器

2. 创建 Network：保存了网络的数据（参数、架构、权重等）

3. 创建 Parser：解析器，解析网络（包含参数、架构、权重等），解析参数、权重到架构里，用于创建可执行的引擎。

4. 绑定输入、输出以及自定义组件：输入、输出--可执行引擎位于 gpu 上，需要把数据从 cpu 传输 gpu 上，在传输的过程中需要先定义好 cpu 内存地址和 gpu 内存地址，绑定好固定的位置进行读写。自定义组件--不支持的网络层；

5. 序列化或者反序列化：减少创建过程，跳过了1，2，3，4步。

6. 传输计算数据（host->device）

7. 执行计算

8. 传输计算结果（device->host）

## Custom Layer 的实现

Custom Layer 通过继承 IPlugin类，对 TensorRT 的五个阶段分别实现。

- 创建网络阶段
- 构建 Engine 阶段
- 序列化 Engine 阶段
- 反序列化 Engine 阶段
- 执行推理 Engine 阶段

### 创建网络阶段

网络创建需要告诉TensorRT该Custom Layer的output矩阵的维数，主要通过以下两种方法传递：

- getNbOutputs()——需要返回该层output矩阵的总数量
- getOutputDimensions()——需要提供该层指定output矩阵的维数

其中该函数有三个参数，分别是output矩阵索引、input矩阵大小以及输入矩阵数。后两者参数是由TensorRT提供的，该方法内部需要计算输出矩阵维数。

### 构建 Engine 阶段

- configure()——如果需要对该层的input矩阵进行验证或选择卷积算法等等操作可以通过配置此函数实现，如果无需配置任何算法可以留空。

    解释：构建Builder通过调用configure()函数来选择算法，如 input 矩阵验证算法、卷积算法等等。configure() 方法只在构建阶段会被调用，因此任何在该函数中确定的参数需要被存储作为 IPlugin 类的成员变量以供后面使用。

- getWorkspaceSize()——如果在运行时需要TensorRT管理的workspace空间可以在此配置，无需申请返回0即可。

    解释：TensorRT 可以提供 workspace 作为每层网络执行时的临时存储空间，该空间是共享的以减少显存占用。构建 Builder 通过调用 getWorkspaceSize() 来确定 workspace 的需求。如果在该函数中设置了 workspace 则在创建执行 Context 的时候分配该 workspace，在 runtime 是提供给 enqueue() 方法使用，并在销毁时回收。使用 TensorRT 分配的 workspace 的好处在于可以在不同 Custom Layer 之间共享使用以减少显存开销。

### 序列化 Engine 阶段

- getSerializationSize()——返回序列化所需的缓存空间。

- serialize()——实现该层参数的序列化，参考代码如下：

```cpp
virtual void serialize(void* buffer) override
{
    char* d = reinterpret_cast<char*>(buffer), *a=d;

    write(d, mNbInputChannels);
    write(d, mNbOutputChannels);
    write(d, (int)mBiasWeights.count);
    serializeFromDevice(d, mKernelWeights);
    serializeFromDevice(d, mBiasWeights);

    assert(d == a + getSerializationSize());
}
```

```cpp
void serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
{
    cudaMemcpy(hostBuffer, deviceWeights.values,
            deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}
```

### 反序列化 Engine 阶段

Custom Layer 的反序列化过程是在反序列化整个模型时候完成的，并通过实例化该 Plugin 类来实现的，参考代码如下所示：

```cpp
// deserialize the engine
IRuntime* runtime = createInferRuntime(gLogger);
ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(),
                                    gieModelStream->size(), &pluginFactory);
```

```cpp
FCPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data), *a =d;
    mNbInputChannels = read<int>(d);
    mNbOutputChannels = read<int>(d);
    int biasCount = read<int>(d);

    mKernelWeights = deserializeToDevice(d,
                            mNbInputChannels * mNbOutputChannels);
    mBiasWeights = deserializeToDevice(d, biasCount);
    assert(d == a + length);
}
```

其中的 deserilizationToDevice 函数是该类的私有函数，也需要编写。

### 执行推理 Engine 阶段

- initialize() and terminate()——分别会在构建 builder 执行网络自动优化和创建/销毁执行 Context 时被调用。

    解释：更多情况下作为其他的资源管理使用，例如 cudnn 资源的初始化和销毁使用

- enqueue()——实现的是该Custom Layer在执行时具体的计算操作。

    解释：该函数拥有五个参数，分别是 input batchsize、input tensor、output tensor、workspace、stream。其中 input batchsize 最大为 Engine 构建时设置的最大 Batchsize，workspace 即前面构建阶段 getWorkspaceSize() 申请的 workspace。需要注意的是这边只传递了 batchsize 信息，但输入以及输出矩阵的维数信息并没有进行传递，因此需要在序列化时将所需的信息进行序列化，例如输入输出矩阵的 channel 数。

## TensorRT 感知 Custom Layer

TensorRT 需要在导入网络模型至 TensorRT 和在创建 Runtime Engine（反序列化 Engine）的时候感知到新的Custom Layer。

### 模型导入时

#### C++ API

使用 C++ 定义网络结构时使用 addPlugin() 函数即可引入 Custom Layer，该函数的输入为该层的输入矩阵，输入矩阵总数以及该层类实现。

#### Python API

使用 Python 定义网络结构时使用 add_plugin() 函数即可引入Custom Layer，该函数的输入为该层的输入矩阵，输入矩阵总数以及该层类实现。

#### NvCaffeParser

NvCaffeParser 的导入方式采用的半自动的导入方式（提供 caffemodel 和 prototxt 的形式），因此无法使用 addPlugin() 函数来显性定义。Caffe的网络层定义中分别有 Name 和 Type 两种参数，在此处为了定位新增的 Custom Layer，TensorRT 中将使用 Name 参数作为标识符。此处的实现是通过继承 nvcaffeparser1::IPluginFactory 类并实现函数 isPlugin() 和 nvinfer1::IPlugin* createPlugin() 函数并将实例化的新类传递给 ICaffeParser::parse() 来完成的。

- isPlugin()——判断是否是Custom Layer。

    解释：该函数输入为 CaffeParser 传递过来的网络层Name参数，因此通过 Name 参数比对即可判断是否该层为特殊的 Custom Layer。

- nvinfer1::IPlugin* createPlugin()——定义层参数。

    解释：目前暂时无法通过模型定义传递参数信息，因此需要在此层中显性定义，其中该层的输入为网络层 Name 和模型 Weights。

需要注意的是，采用 NvCaffeParser 导入的方式时，Custom Layer 的实现将默认使用从 NvCaffeParser 传递的 Weights 并认为在 initialize() 函数执行前数据已经准备就绪。参考实现如下：

```cpp
// caffe parser plugin implementation
bool isPlugin(const char* name) override
{
    return !strcmp(name, "ip2");
}

virtual nvinfer1::IPlugin* createPlugin(const char* layerName,
                const nvinfer1::Weights* weights, int nbWeights) override
{
    // there's no way to pass parameters through from the model definition,
    // so we have to define it here exlicitly
    static const int NB_OUT_CHANNELS = 10;
    assert(isPlugin(layerName) && nbWeights == 2 &&
                weights[0].type == DataType::kFLOAT &&
                weights[1].type == DataType::kFLOAT);
    assert(mPlugin.get() == nullptr);
    mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(weights, nbWeights,
                                        NB_OUTPUT_CHANNELS));
    return mPlugin.get();
}
```

```cpp
ICaffeParser* parser = createCaffeParser();
parser->setPluginFactory(pluginFactory);
```

### 创建 Runtime Engine 时

创建 Runtime Engine 时（反序列化Engine时）和模型导入时不一样，此处无论是哪种方式导入的模型都已经转化成优化好的序列化的 TensorRT Engine。因此此时只需要继承类 nvinfer1::IPluginFactory 并实现其接口函数 nvinfer1::IPlugin，之后将继承的新类实例化并传递给 IInferRuntime::deserializeCudaEngine() 即可，参考代码如下：

```cpp
// deserialization plugin implementation
IPlugin* createPlugin(const char* layerName, const void* serialData,
                      size_t serialLength) override
{
    assert(isPlugin(layerName));
    assert(mPlugin.get() == nullptr);
    mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(serialData,
                                        serialLength));
    return mPlugin.get();
}
```

```cpp
// deserialize the engine
IRuntime* runtime = createInferRuntime(gLogger);
ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(),
                                    gieModelStream->size(), &pluginFactory);
```
需要注意的是，在创建 Runtime时（反序列化的实现如前文 "Custom Layer实现" 中反序列化部分所示），这部分的实现默认使用字节流数据并认为认为在 initialize() 函数执行前数据已经准备就绪。