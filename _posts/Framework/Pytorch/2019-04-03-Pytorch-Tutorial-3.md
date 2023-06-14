---
layout: post
title: Pytorch官方教程(三)—Learning Pytorch with Examples
category: Framework
tags: pytorch
keywords: pytorch tutorial
description:
---


Pytorch 提供了两个主要特性：

- n 维 Tensor，类似 numpy 不过可以在 GPU 上运行
- 构建和训练神经网络的自动微分

使用全连接 ReLU 网络作为运行示例。在网络中有一个隐藏层，并通过梯度下降训练来匹配随机数据，使网络输出与真实输出之间的欧氏距离最小化。

# Tensors

## Warm-up ：numpy

先使用 numpy 实现网络。

Numpy 提了一个 n 维数组对象和许多用于操作这些数组的函数。Numpy 是科学计算的通用框架，它不知道计算图、深度学习和梯度。然而，我们可以很容易地使用 numpy 来调试一个两层随机数据的网络，通过使用 numpy 手动实现网络的正向和反向传播:


```python
# -*- coding: utf-8 -*-
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(5):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

    0 27204862.949353315
    1 23180364.42934936
    2 21658625.784368478
    3 19759276.402975723
    4 16662639.494721428


## PyTorch：Tensor

Numpy 是一个很好的框架，不过它不支持使用 GPU 增加计算速度。对于现在的深度神经网络，GPU 可以提供 50 倍或更高的计算速度，因此，numpy 对于当前的深度学习来说是不够的。

这里我们介绍 Pytorch 最基本的概念：Tensor。 Pytorch Tensor 在概念上 numpy array 相同：Tensor 是一个 n 维矩阵，并且 Pytorch 为 Tensor 提供了很多函数操作。实际上，Tensor 可以追踪计算图和梯度，但也可以作为通用的科学计算工具。

与 numpy 不同 Pytorch Tensor 可以利用 GPU 加速计算。在 GPU 上运行 pytorch tensor，只需将其转化为新的数据类型。

这里我们使用 numpy 来调试一个两层随机数据的网络，与上面的 numpy 例子一样，手动实现网络的正向和反向传播。


```python
# -*- coding: utf-8 -*-

import torch


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(5):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

    0 39775964.0
    1 39473524.0
    2 40811516.0
    3 35731176.0
    4 24346566.0


# Autograd

## Pytorch：Tensors and autograd

在上面的例子中，我们必须手动实现网络的正向和反向传播。对于一个两层的简单网络，手动实现反向传播比较简单，但对于大的复杂网络实现反向传播会很复杂。

Pytorch 中的 autograd 包，可以自动微分自动实现神经网络的反向传播。当使用 autograd 时，网络的正向传播将定义一个计算图；图中的节点是 Tensor，边是由输入 tensor 产生输出 tensor 的函数。通过这个图的反向传播可以轻松的计算梯度。

这听起来有些复杂，但在实际应用时非常简单。每个 Tensor 代表计算图的节点，假设 x 是一个 x.requires_grad=True 的 Tensor，那么 x.grad是另一个 tensor，它代表 x 对某个标量的梯度。

我们使用 pytorch torch和自动微分来实现两层网络。现在，我们不需要手动实现网络的反向传播。


```python
# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(5):
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```

    0 27121504.0
    1 22392120.0
    2 20655246.0
    3 19197876.0
    4 16879916.0


## Pytorch：Defining new autograd functions

实际上，每个原始的 autograd 运算符实际上是两个作用于 tensor 的函数。forward 函数从输入 tensor 计算输出 tensor。backward 函数接收输出 tensor 相对于某个标量值的梯度，并计算输入 tensor 相对于该标量值的梯度。

在 pytorch 中，我们可以通过定义 torch.autograd.Function 的子类并实现 forward 和 backward 函数来轻松定义我们自己的 autograd 运算符。然后，我们可以通过构造一个实例并像调用函数一样来使用我们的新 autograd 运算符，并传递包含输入数据的 Tensors。

在这个例子中，我们定义了自己的自定义 autograd 函数来执行 非线性ReLU，并使用它来实现我们的两层网络:


```python
# -*- coding: utf-8 -*-
import torch


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(5):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
```

    0 29605528.0
    1 23772104.0
    2 23088974.0
    3 23347500.0
    4 22092518.0


## TensorFlow：Static Graphs

Pytorch 和 TnesorFlow 的 autograd 看起来很像：都定义了计算图，都是用自动微分计算梯度。不过这两个框架最大的不同是，TensorFlow 是静态计算图，Pytorch 是动态计算图。

在 TensorFlow 中我们定义一次计算图，然后反复执行相同的计算图，输入计算图的数据可能不同。在 Pytorch 中每次前向传播定义一个新的计算图。

静态图是很好的，因为你可以提前优化计算图，例如，为提高效率，融合一些计算图操作，或者将图形分布到多个 gpu 或 多个设备上。如果你重复使用相同的计算图，然后，这个可能代价高昂的前期优化可以分摊，因为相同的计算图会一遍又一遍地重新运行。

静态图和动态图的一个不同点是控制流。对于某些模型，我们可能希望对每个数据点执行不同的计算，例如，rnn 可能需要为每个数据点展开到多个不同的时间步长，这种展开可以作为一个循环实现。对于静态图，这种循环结构需要成为静态图的一部分。为了将循环嵌入到计算图中，tensorflow 提供了类似 tf.scan 的操作。对于动态图这种情况就很简单了：由于为每个示例构建了动态图，因此可以用常规命令流控制每个输入执行不同的计算。

为了与上面的 PyTorch autograd 示例形成对比，这里我们使用 tensorflow 来搭建一个简单的两层网络。


```python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import numpy as np

# First we set up the computational graph:

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create placeholders for the input and target data; these will be filled
# with real data when we execute the graph.
x = tf.placeholder(tf.float32, shape=(None, D_in))
y = tf.placeholder(tf.float32, shape=(None, D_out))

# Create Variables for the weights and initialize them with random data.
# A TensorFlow Variable persists its value across executions of the graph.
w1 = tf.Variable(tf.random_normal((D_in, H)))
w2 = tf.Variable(tf.random_normal((H, D_out)))

# Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
# Note that this code does not actually perform any numeric operations; it
# merely sets up the computational graph that we will later execute.
h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

# Compute loss using operations on TensorFlow Tensors
loss = tf.reduce_sum((y - y_pred) ** 2.0)

# Compute gradient of the loss with respect to w1 and w2.
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update the weights using gradient descent. To actually update the weights
# we need to evaluate new_w1 and new_w2 when executing the graph. Note that
# in TensorFlow the the act of updating the value of the weights is part of
# the computational graph; in PyTorch this happens outside the computational
# graph.
learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

# Now we have built our computational graph, so we enter a TensorFlow session to
# actually execute the graph.
with tf.Session() as sess:
    # Run the graph once to initialize the Variables w1 and w2.
    sess.run(tf.global_variables_initializer())

    # Create numpy arrays holding the actual data for the inputs x and targets
    # y
    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)
    for _ in range(5):
        # Execute the graph many times. Each time it executes we want to bind
        # x_value to x and y_value to y, specified with the feed_dict argument.
        # Each time we execute the graph we want to compute the values for loss,
        # new_w1, and new_w2; the values of these Tensors are returned as numpy
        # arrays.
        loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                    feed_dict={x: x_value, y: y_value})
        print(loss_value)
```

    28633724.0
    26991910.0
    30991036.0
    35920652.0
    36464064.0


# nn module

## Pytorch：nn

计算图和 autograd 对于定义复杂算子并自动求导是很强大的范式。但是，对于大型的神经网络原始的 autograd 就太底层了。

在构建神经网络时，我们经常考虑将计算排列在可学习参数的层中。在 tensorflow 中，Keas、Tensorflow-Slim、TFLearn不使用原始的计算图而是提供了高层的抽象。在Pytorch中， nn 包也有相同的功能。nn 中定义了相当于层的 Modules。Module 接受 Tensor 作为输入，计算得到输出 Tensor,不过也有可能保存内部状态，例如，包含可学习参数的 Tensor。nn 还定义了一些常见的损失函数。

使用 nn 包实现两层网络：


```python
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(5):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
```

    0 745.7744750976562
    1 691.6641845703125
    2 645.3390502929688
    3 604.81689453125
    4 569.1459350585938


## Pytorch：optim

到目前为止，我们通过手动改变有可学习参数的张量来更新模型的权重（使用 torch.no_grad() 或 .data 来阻止 autograd 追踪历史）。对于像随机梯度下降这样的简单优化算法来说，这不是一个很大的负载，但在实践中，我们经常使用更复杂的优化器，如 AdaGrad，RMSProp，Adam 等来训练神经网络。

PyTorch 中的 optim 包提取了优化算法的思想，并提供了常用优化算法的实现。在这个例子中，我们将像之前一样使用nn包来定义我们的模型，但是我们将使用optim包提供的Adam算法来优化模型:


```python
# -*- coding: utf-8 -*-
import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(5):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
```

    0 646.5396118164062
    1 629.6614990234375
    2 613.2215576171875
    3 597.3142700195312
    4 581.8770751953125


## Pytorch：Custom nn Modules

有时，你可能需要指定比现有模块序列更复杂的模型。对于这种情况，您可以通过继承 nn.Module 并定义一个接收输入 Tensor 并使用其他模块或 Tensor 上的其他 autograd 操作生成输出 Tensor 的 forward 来自定义你的模块。

在此示例中，我们将双层网络实现为自定义模块子类


```python
# -*- coding: utf-8 -*-
import torch


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(5):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    0 770.3958129882812
    1 712.6732177734375
    2 663.19677734375
    3 620.0232543945312
    4 581.3988037109375


## Pytorch：Control Flow + Weight Sharing

作为动态图和权重共享的一个例子，我们实现了一个非常奇怪的模型：一个全连接的 ReLU 网络，在每次正向传播时随机选择1到4隐藏层，多次重复使用相同的权重来计算最里面的隐藏层。对于这个模型，我们可以使用普通的 Python 流来控制循环，并且我们可以通过在定义正向传递时，多次重复使用同一个模块来实现最内层之间的权重共享。


```python
# -*- coding: utf-8 -*-
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
for t in range(5):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    0 672.7574462890625
    1 671.1217041015625
    2 666.5305786132812
    3 666.6317749023438
    4 663.6578979492188
