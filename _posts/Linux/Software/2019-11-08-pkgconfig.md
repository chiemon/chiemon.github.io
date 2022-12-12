---
layout: post
title: pkg-config
category: Software
tags: pkg-config
keywords: pkg-config
description:
---

Makefile 中使用

pkg-config --cflags opencv protobuf
pkg-config --libs opencv protobuf

检查库的版本号。如果所需要的库的版本不满足要求，它会打印出错误信息，避免链接错误版本的库文件。
获得编译预处理参数，如宏定义，头文件的位置。
获得链接参数，如库及依赖的其它库的位置，文件名及其它一些连接参数。
自动加入所依赖的其它库的设置。

这一切都自动的，库文件安装在哪里都没关系！

pkg-config 是一个常用的库信息提取工具。
pkg-config 工具从.pc文件中提取编译链接配置路径。
pkg-config 从一个叫做PKG_CONFIG_PATH的环境变量中查找pc文件

设置 环境变量命令：

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

查看 pkg-config 搜索 .pc 文件的路径

pkg-config --variable pc_path pkg-config

    /usr/local/lib/x86_64-linux-gnu/pkgconfig:
    /usr/local/lib/pkgconfig:
    /usr/local/share/pkgconfig:
    /usr/lib/x86_64-linux-gnu/pkgconfig:
    /usr/lib/pkgconfig:
    /usr/share/pkgconfig
