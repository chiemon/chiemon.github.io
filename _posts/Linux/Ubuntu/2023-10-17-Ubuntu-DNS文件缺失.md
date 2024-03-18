---
layout: post
title: Ubuntu 初始设置
category: Linux
tags: Linux
keywords: ubuntu
description:
---

1. 重新启动网络管理服务,让系统重新生成resolv.conf:
```bash
sudo service network-manager restart
```

2. 重启network-manager未生效,尝试:

```bash
# 重启resolvconf服务:
sudo service resolvconf restart
# 编辑/etc/resolvconf/resolv.conf.d/base文件,添加nameserver:
vim /etc/resolvconf/resolv.conf.d/base
> nameserver 8.8.8.8

# 然后重新生成resolv.conf:
sudo resolvconf -u
```

3. 检查DNS设置

```bash
# 检查/etc/network/interfaces文件中的DNS设置。
cat /etc/network/interfaces
# nameserver 8.8.8.8
```
