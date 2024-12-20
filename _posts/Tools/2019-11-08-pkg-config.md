---
layout: post
title: pkg-config
category: Tools
tags: pkg-config
keywords: pkg-config
description:
---

`pkg-config` 是一个帮助程序获取已安装库的编译和链接信息的工具，通常在 Unix/Linux 系统上用于简化编译过程。

- 从.pc文件中提取编译链接配置路径。
- 从`PKG_CONFIG_PATH`的环境变量中查找`pc`文件。

## 基本用法

```bash
# 列出系统中所有已安装的库及其版本
pkg-config --list-all

# 检查库是否已安装，成功则返回 0，否则返回非 0 值。
pkg-config --exists opencv protobuf

# 查询指定库的版本号
pkg-config --modversion opencv protobuf


# 获取指定库的编译选项（头文件路径等）
pkg-config --cflags opencv protobuf

# 获取指定库的链接选项，要链接哪些库和库文件所在的路径
pkg-config --libs opencv protobuf

# 同时获取编译和链接选项
pkg-config --cflags --libs opencv rotobuf
```

## Makefile 中使用


设置环境变量：

```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

查看 `pkg-config` 搜索 .pc 文件的路径

```bash
pkg-config --variable pc_path pkg-config

    /usr/local/lib/x86_64-linux-gnu/pkgconfig:
    /usr/local/lib/pkgconfig:
    /usr/local/share/pkgconfig:
    /usr/lib/x86_64-linux-gnu/pkgconfig:
    /usr/lib/pkgconfig:
    /usr/share/pkgconfig
```