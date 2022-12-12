---
layout: post
title: Tensorflow tfrecord 数据集制作及读取
category: Framework
tags: Tensorflow
keywords: tfrecord
description: 以猫狗大战（分类）为例
---

### 导入必要的库

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import tensorflow as tf
from scipy import misc
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
```

### 定义数据类型转换函数

```python
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```

### 生成获取图片地址 + 标签的txt文件(顺序)

```python
def mk_txt():
    data_dir = 'C:/Users/ahxie/PycharmProjects/Cats_vs_Dogs/data/train/'  #图片路径
    #data_dir = 'C:/Users/ahxie/PycharmProjects/Cats_vs_Dogs/data/test/'

    # 输出txt文件路径及文件名
    output_path = 'C:/Users/ahxie/PycharmProjects/Cats_vs_Dogs/list_train.txt'
    #output_path = 'C:/Users/ahxie/PycharmProjects/Cats_vs_Dogs/list_test.txt'
    fd = open(output_path, 'w')
    images_list = os.listdir(data_dir)
    for image_name in images_list:
        image_name = image_name.strip()
        labels = image_name.split('.')[0]
        if labels == 'cat':
            fd.write('{}/{} {}\n'.format(data_dir, image_name, '0'))
        elif labels == 'dog':
            fd.write('{}/{} {}\n'.format(data_dir, image_name, '1'))
    fd.close()
```

### 把训练集随机分为训练集和验证集(乱序)

```python
def train2val():
    list_path = 'list.txt'  # 全部训练集的txt文本路径
    train_list_path = 'list_train.txt'  # 生成的训练集txt文件名(或路径+文件名)
    val_list_path = 'list_val.txt'  # 生成的验证集txt文件名(或路径+文件名)
    val_per = 0.1  # 验证集占比
    RANDOM_SEED = 0

    fd = open(list_path)
    lines = fd.readlines()
    fd.close()
    num_lines = len(lines)
    NUM_VALIDATION = int(num_lines * val_per)
    random.seed(RANDOM_SEED)
    random.shuffle(lines)
    fd = open(train_list_path, 'w')
    for line in lines[NUM_VALIDATION:]:
        fd.write(line)
    fd.close()
    fd = open(val_list_path, 'w')
    for line in lines[:NUM_VALIDATION]:
        fd.write(line)
    fd.close()
```

### 通过读取txt文件，生成tfrecord文件(乱序)

```python
def mk_tfrecord():

    # 读取的txt文件的路径及文件名
    # list_path = '/home/xieqi/traffic_class/dir/retrain/data/list_train.txt'
    list_path = 'C:/Users/ahxie/PycharmProjects/Cats_vs_Dogs/list_train.txt'

    # 生成的tfrecord文件路径及文件名
    # record_name = '/home/xieqi/traffic_class/dir/retrain/data/dir_train.tfrecords'
    record_name = 'C:/Users/ahxie/PycharmProjects/Cats_vs_Dogs/train.tfrecords'

    fd = open(list_path)
    lines = fd.readlines()
    fd.close()
    random.shuffle(lines)
    writer = tf.python_io.TFRecordWriter(record_name)  # 创建一个tfrecord文件
    for item_path in lines:
        item_path = item_path.strip()  # 去掉首尾的换行符和空格
        img_path, img_label = item_path.split(' ') # 拆分图片绝对路径及标签

        img_label = int(img_label)  # 将标签由字符型转化为整型

        # imagel类型为Jpeg.File, 此处也可用cv2.imread()直接读取返回类型为numpy.ndarray，dtype=uint8
        img = Image.open(img_path)

        img = np.asarray(img, np.uint8) # 将Jpeg.File格式转化为ndarray
        img_height, img_width, img_channel = img.shape  # 图片大小
        img_raw = img.tobytes()  # 将图片转化为字符串格式（二进制文件）

        # example对象对label和image数据进行封装
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(img_label),  # 标签的数据形式为int64
            'img_raw': _bytes_feature(img_raw),  # 图片的数据形式为Bytes
            "img_height": _int64_feature(img_height),
            "img_width": _int64_feature(img_width)}))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()
```
### 读取解码数据, 返回的是Tensor

```python
def read_and_decode(filename_queue, image_W, image_H, batch_size,min_after_dequeue):
    """
    filename_queue：文件名字符串队列
    image_W, image_H：图片的宽和高
    batch_size：批次大小
    min_after_dequeue：队列中剩余图片数目
    """

    reader = tf.TFRecordReader()  # tfrecord文件阅读器--类

    # 从文件名队列中读数据，返回下一个记录（键，值）对
    _, serialized_example = reader.read(filename_queue)

    # 解析读取的样例。
    features = tf.parse_single_example(serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string),
                'img_height': tf.FixedLenFeature([], tf.int64),
                'img_width': tf.FixedLenFeature([], tf.int64)})

    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['img_height'], tf.int32) #此处必须为int32，int64无法显示图片
    width = tf.cast(features['img_width'], tf.int32)

    image = tf.decode_raw(features['img_raw'], tf.uint8)  # 将字符串解析成图像对应的像素数组
    # tf.decode_raw 转换成字符串之前是什么类型的数据，此处就要转换成对应的类型
    channel = 3
    image = tf.reshape(image, [height, width, channel])  # 向量---三维矩阵
    image = tf.cast(image, tf.float32)

    # 统一图片大小--缩放或裁剪，生成batch时图片必须为相同大小
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # 标准化
    #image = tf.image.per_image_standardization(image)

    # 随机选取样本组成batch
    # capacity: 队列大小
    # num_threads: 线程数目
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                    capacity=capacity, num_threads=64, min_after_dequeue=min_after_dequeue)

    #one_hot
    """
    label_batch = tf.one_hot(label_batch, depth= 2)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, 2])
    """

    return image_batch, label_batch
```

### 测试

```python
def test_run():

    # tfrecord存放路径及名称
    tfrecord_filename = 'C:/Users/ahxie/PycharmProjects/Cats_vs_Dogs/train.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=1)
    # 该处得到的为tensor，需要sess.run才能得到实际的数据
    image, label = read_and_decode(filename_queue, 208, 208, 16,min_after_dequeue=1000)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()  # 从队列中取数据需要先建立一个Coordinator()
        # 并建立线程开始从队列中读取数据
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1):
            example, lab = sess.run([image, label])  # 取出image和label
            print(type(example))
            print(type(lab))
            for j in range(10):
                print(lab[j])
                img = np.uint8(example[j])
                plt.imshow(img)
                plt.show()

        coord.request_stop()
        coord.join(threads)
        sess.close()
```

### 主函数

```python
if __name__ == '__main__':
    #mk_txt() # 生成获取图片地址 + 标签的txt文件
    #mk_tfrecord() # 读取txt文件，生成tfrecord文件
    test_run()
```