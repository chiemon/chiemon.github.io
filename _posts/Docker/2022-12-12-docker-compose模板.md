---
layout: post
title: docker-compose 模板
category: Docker
tags: docker
keywords: docker-compose
description:
---

```yaml
version: "3"

services:
  # 服务名
  triton-server-20.09:
    # 镜像标签
    image: nvcr.io/nvidia/tritonserver:20.09-py3
    # 重启策略
    restart: always
    # 网络
    # network_mode: "bridge"
    networks:
      front-tier:
        ipv4_address: 172.16.238.10
        ipv6_address: 2001:3984:3989::10
    # 容器名称
    container_name: triton-server-20.09
    # 主机名
    #   hostname: "leinao_devel"
    # env_file:
    #   - .env/xq_cross_border
    working_dir: /workspace
    # 容器的标准输入保持打开。相当于 docker run 的 -i
    # stdin_open: true
    # 分配一个虚拟终端并绑定到容器的标准输入上。相当于 docker run 的 -t
    # tty: true
    # 共享内存大小
    shm_size: 1gb
    # 限制容器使用的系统资源
    ulimits:
      # 堆栈的最大值
      stack: 67108864
    # 获取主机root权限
    privileged: true
    # 容器中添加的功能
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp:unconfined
      - apparmor:unconfined
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    environment:
    #   - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
    volumes:
        - /home/zkln/workspace/triton-r20.09/model_repository:/workspace/model_repository
    command:
      - tritonserver
      - --model-repository=/workspace/model_repository
      # - /bin/bash
    # docker-compose > 1.28.0
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              driver: nvidia
              device_ids: ["1"]

networks:
  front-tier:
    ipam:
      driver: default
      config:
        - subnet: "172.16.238.0/24"
        - subnet: "2001:3984:3989::/64"
```

## `restart`

容器重启策略。

- `"no"`：默认策略。任何情况下都不会重启。
- `always`：始终重启容器，直到将其删除。
- `on-failure`：如果退出代码指示错误，重启容器。
- `unless-stopped`：无论如何退出代码都重启容器。容器停止或删除时停止重启。

## `shm_size`

容器允许的共享内存（Linux上的/dev/shm分区）的大小。单位`b`(bytes)，`k` or `kb`(kilo bytes)，`m` or `mb`(mega bytes)，`g` or `gb`(giga bytes)。

- 2b
- 1024kb
- 2048k
- 300m
- 1gb

## `privileged`

出于安全考虑，docker容器中默认的root用户只是相当于主机上的一个普通用户权限，不允许访问主机上的任何设备。使用该参数让容器获取主机root权限，允许容器访问连接到主机的所有设备（位于/dev文件夹下）。

## `cap_add`

为容器添加指定功能。

- `SYS_PTRACE`: 添加`ptrace`能力。用于对进程进行调试或者进程注入。

## `security_opt`

覆盖容器的默认标签方案。

- `seccomp:unconfined`: 关闭seccomp profile功能。docker有Seccomp filtering功能，以伯克莱封包过滤器（Berkeley Packer Filter，缩写BPF）的方式允许用户对容器内的系统调用（syscall）做自定义的"allow"，"deny"，"trap"，"kill"，or "trace"操作，由于Seccomp filtering的限制，在默认配置下，会导致在使用GDB的时候run失败。

- `apparmor:unconfined`: 关闭`apparmor`（Application Armor 内核安全模块）限制。

