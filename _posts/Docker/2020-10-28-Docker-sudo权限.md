---
layout: post
title: Docker sudo 权限问题
category: Docker
tags: docker
keywords: docker
description:
---

```bash
# 创建 docker 组
sudo groupadd docker

# 将用户加入 docker 组
sudo gpasswd -a ${USER} docker

# 重启 docker 服务
sudo systemctl daemon-reload
sudo systemctl restart docker

# 更新用户组
newgrp  docker
```
