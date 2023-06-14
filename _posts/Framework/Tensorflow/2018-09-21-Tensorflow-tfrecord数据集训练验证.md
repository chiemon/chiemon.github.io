---
layout: post
title: Tensorflow tfrecord 数据集训练及验证
category: Framework
tags: Tensorflow
keywords: tfrecord
description: 以猫狗大战（分类）为例
---

### 导入必要的库

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-

from mk_tfrecord import *
#from model import *
from inception_v3 import *
import numpy as np
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

### 定义训练函数

```python
def training():
    N_CLASSES = 2              # 分类数目
    IMG_W = 299                # 统一图片大小，宽度
    IMG_H = 299                # 统一图片大小，高度
    BATCH_SIZE = 64            # 批次大小
    MAX_STEP = 50000           # 迭代次数
    LEARNING_RATE = 0.0001     # 学习率
    min_after_dequeue = 1000

    tfrecord_filename = '/home/xieqi/project/cat_dog/train.tfrecords'   # 训练数据集
    logs_dir = '/home/xieqi/project/cat_dog/log_v3'     # 检查点保存路径

    # 输入--要生成的字符串的一维字符串张量，shuffle默认为True，输出--字符串队列
    # 将字符串（例如文件名）输出到输入管道的队列，不限制num_epoch。
    filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=150)
    train_image, train_label = read_and_decode(filename_queue, image_W=IMG_W, image_H=IMG_H,
                batch_size=BATCH_SIZE,min_after_dequeue=min_after_dequeue) # 返回的为tensor

    train_labels = tf.one_hot(train_label, N_CLASSES)

    train_logits,_ = inception_v3(train_image,num_classes=N_CLASSES)
    train_loss = loss(train_logits, train_labels) # 损失函数
    train_acc = accuracy(train_logits, train_labels) # 模型精确度
    my_global_step = tf.Variable(0, name='global_step', trainable=False) # 全局步长
    train_op = optimize(train_loss, LEARNING_RATE, my_global_step) #训练模型

    summary_op = tf.summary.merge_all() # 收集模型统计信息
    # 初始化全局变量和局部变量
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # 限制GPU使用率
    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.70
    # sess = tf.Session(config=sess_config)

    sess = tf.Session()
    # FileWriter类提供了一个机制来创建指定目录的事件文件，并添加摘要和事件给它(异步更新，不影响训练速度)
    train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    # 将Save类添加OPS保存和恢复变量和检查点。对模型定期做checkpoint，通常用于模型恢复
    saver = tf.train.Saver()

    sess.run(init_op)
    # 线程协调员, 实现一种简单的机制来协调一组线程的终止
    coord = tf.train.Coordinator()
    # 启动图中收集的所有队列， 开始填充队列
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            # 获取一个批次的数据及标签
            image_batch, label_batch = sess.run([train_image, train_label])
            sess.run(train_op)

            # 每迭代100次计算一次loss和准确率
            if step % 100 == 0:
                losses, acc = sess.run([train_loss, train_acc])
                print('Step: %6d, loss: %.8f, accuracy: %.2f%%' % (step, losses, acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or step == MAX_STEP - 1:  # 保存检查点
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
```

### 定义验证函数

```python
# 测试检查点
def eval():
    N_CLASSES = 2
    IMG_W = 299
    IMG_H = 299
    BATCH_SIZE = 1
    MAX_STEP = 512
    min_after_dequeue=0

    test_dir = '/home/xieqi/project/cat_dog/val.tfrecords' #测试集数据
    logs_dir = '/home/xieqi/project/cat_dog/log_v3'     # 检查点目录
    false_pic_dir = '/home/xieqi/project/cat_dog/false_pic/' #错误分类的图片存储地址

    # 输入要生成的字符串的一维字符张量，输出字符串队列，shuffle默认为True
    filename_queue = tf.train.string_input_producer([test_dir], num_epochs=1)
    train_image, train_label = read_and_decode(filename_queue, image_W=IMG_W, image_H=IMG_H,
                batch_size=BATCH_SIZE,min_after_dequeue=min_after_dequeue) # 返回的为tensor

    train_labels = tf.one_hot(train_label, N_CLASSES)

    train_logits, _ = inception_v3(train_image, N_CLASSES)
    train_logits = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值

    #计算准确率
    correct_num = tf.placeholder('float')
    correct_pre = tf.div(correct_num, MAX_STEP)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')

    # 通过checkpoint文件找到模型文件名，有两个属性：model_checkpoint_path最新的模型文件的文件名
    # all_model_checkpoint_paths未被删除的所有模型文件的文件名
    ckpt = tf.train.get_checkpoint_state(logs_dir)

    if ckpt and ckpt.model_checkpoint_path:
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %d\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        correct = 0
        wrong = 0
        dt_list = []
        for step in range(MAX_STEP):

            if coord.should_stop():
                break

            st = time.time()
            image, prediction, labels = sess.run([train_image, train_logits, train_labels])
            dt = time.time() - st
            dt_list.append(dt)

            p_max_index = np.argmax(prediction)
            c_max_index = np.argmax(labels)

            if p_max_index == c_max_index:
                for i in range(BATCH_SIZE):
                    correct += 1
            else:
                for i in range(BATCH_SIZE):
                    wrong += 1
                    cv2.imwrite(false_pic_dir+'ture'+str(labels)+'predict'+ \
                                str(prediction)+'.jpg', image[i])

        accuray_rate = sess.run(correct_pre,feed_dict={correct_num: correct})
        velocity = np.mean(dt_list)
        print('Total: %5d, correct: %5d, wrong: %5d, accuracy: %3.2f%%, each speed: %.4fs' %
            (MAX_STEP, correct, wrong, accuray_rate * 100, velocity))
    except tf.errors.OutOfRangeError:
        print('OutOfRange')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()
```

### 主函数

```python
if __name__ == '__main__':
    training()
    #eval()
```