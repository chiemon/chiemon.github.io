---
layout: post
title: Caffe 添加 python layer
category: Framework
tags: Caffe
keywords: Caffe
description:
---

## 前期工作

- 设置WITH_PYTHON_LAYER := 1

- 编译caffe


## Example

实现功能: 从一个文件夹中批量读取图片并且resize为固定尺寸

### 定义caffe layer

创建文件 MyPythonLayer.py，文件内容如下，MyPythonLayer是 module 名字。

```python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,".../caffe/python")

import caffe
import yaml, glob
from random import shuffle
import cv2
import logging
logging.basicConfig(level=logging.INFO)

class myPythonLayer(caffe.Layer):
    """
    reshape images

    继承 caffe.Layer 的四个基本方法setup, reshape, forward, backward
    """
    def setup(self, bottom, top):
        """
        setup中主要添加类的初始化过程，而当类被调用时，则会调用方法forward
        """

        # 参数有多个对，但并不会解析多个对，提前 split
        params_str = self.param_str.split(',')

        # yaml.load(str)时，str必须以空格间隔
        # 比如”‘batch_size’: 50”, ‘batch_size’: 与 50 以空格间隔，反之会报错
        params = [yaml.load(item) for item in params_str]
        print params

        # 参数的顺序很重要，网络定义的 prototxt 的参数的顺序要根据这个来
        self.source = params[0]['source_dir']   # 图片的根目录
        self.target_size = params[1]['target_size'] # resize的目标尺寸大小
        self.batch_size = params[2]['batch_size']   # 批大小

        self.batch_loader = BatchLoader(source_dir=self.source, target_size=self.target_size)
        print 'Parameter batch_size:{}\n' \
              'source_dir:{}\n' \
              'target_size:{}'.format(self.batch_size, self.source, self.target_size)

        # top必须被 reshape, 否则在 blob.cpp 中115行会CHECK(data_)报错
        top[0].reshape(self.batch_size, self.target_size, self.target_size, 3)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        """
        forward是必须的
        读取batch数量的图片数目, 以实现批量读取的功能
        """
        for i in xrange(self.batch_size):
            top[0].data[i, ...] = self.batch_loader.next_batch()

    def backward(self, bottom, propagate_down, top):
        """
        backward不是必须的
        """
        pass

class BatchLoader(object):

    def __init__(self, source_dir, target_size):
        self.cur = 0
        self.target_size = target_size
        self.indexlist = glob.glob(source_dir+ '/*.jpg')

    def next_batch(self):
        if self.cur == len(self.indexlist):
            self.cur = 0
            shuffle(self.indexlist)
        item = self.indexlist[self.cur]
        img_tmp = cv2.imread(item)
        img_tmp = cv2.resize(src=img_tmp, dsize=(self.target_size, self.target_size))
        self.cur += 1
        logging.info('load {} images'.format(self.cur))
        return img_tmp
```

### 定义prototxt文件

```
layer{
name: "mylayer"
type: "Python"  # type必须是python， caffe编译的时候config里面 with python layer参数要打开
top: "images"
python_param{
    module: "MyPythonLayer" # module 就是我们 python 层文件的名字
    layer: "myPythonLayer"  # myPythonLayer 就写 python 文件里面的class的名字
    param_str: "'source_dir': '/home/sai/code/face_detection/train/lfw_5590','target_size': 224,'batch_size': 50"
    }
}
```

param_str是一个字符串，里面是字典的格式，用于设置定义层的参数。要注意param_str的顺序。

### 运行测试

```python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(0,".../caffe/python")

from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto

def My_layer(source_dir, batch_size, target_size):
    param_str = "'source_dir': source_dir, 'batch_size': batch_size, 'target_size': target_size"
    mylayer = L.Python(module='MyPythonLayer', layer='myPythonLayer', param_str=param_str)
    print(mylayer)
    to_proto(mylayer)

def make(filename):
    with open(filename, 'w') as f:
        print(My_layer(source_dir='/home/sai/code/face_detection/train/lfw_5590', batch_size=50, target_size=100), file=f)

if __name__=='__main__':
    net = caffe.Net('mylayer.prototxt', caffe.TEST)
    net.forward()
    images = net.blobs['images'].data
    print(images.shape)
```