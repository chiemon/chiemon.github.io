---
layout: post
title: nvidia-docker 安装
category: Docker
tags: docker
keywords: docker
description:
---

### NVIDIA-Docker 安装

- Ubuntu 16.04/18.04：

    ![官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#getting-started)

    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
        && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt-get update
    sudo apt-get install nvidia-docker2
    #sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

### Issue


1. docker: Error response from daemon: Unknown runtime specified nvidia

    解决方法：

    ```bash
    # 重启 docker 服务
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```

2. docker: Error response from daemon: OCI runtime create failed: container_linux.go:380: starting container process caused: process_linux.go:545: container init caused: Running hook #0:: error running hook: signal: segmentation fault (core dumped), stdout: , stderr:: unknown.

    解决方法：

    ```bash
    # 降级
    sudo apt-get install nvidia-docker2=2.4.0-1
    # 重启 docker 服务
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    ```
