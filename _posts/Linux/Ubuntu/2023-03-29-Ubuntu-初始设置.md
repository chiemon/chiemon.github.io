---
layout: post
title: Ubuntu 初始设置
category: Linux
tags: Linux
keywords: ubuntu
description:
---

### 删除多余的软件

```bash
sudo apt-get remove libreoffice-common thunderbird totem rhythmbox simple-scan gnome-mahjongg aisleriot gnome-mines cheese transmission-common gnome-sudoku remmina
sudo apt autoremove
sudo apt-get clean
```
### 修改系统源

```
sudo vi /etc/apt/sources.list
```

命令行模式：`:1,$d`清空原内容

清华镜像源：`https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/`

``
sudo apt-get update
``

### 软件安装

#### 必要软件

```bash
# 网络工具
sudo apt-get install net-tools
# ssh
sudo apt-get install openssh-server
sudo /etc/init.d/ssh status
# 修复软件依赖
sudo apt --fix-broken install
# 安装 tabby
sudo dpkg -i tabby-1.0.196-linux-x64.deb
```

#### 开发环境

```bash
# 编译工具
sudo apt-get install build-essential autoconf automake libtool pkg-config ca-certificates tzdata gdb vim wget curl unzip
# 设置时间
ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
echo Asia/Shanghai > /etc/timezone

# opencv 必要软件
sudo apt-get install libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev
# opencv 可选
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

sudo apt-get install libcurl4-openssl-dev zlib1g-dev libglib2.0-dev
````
