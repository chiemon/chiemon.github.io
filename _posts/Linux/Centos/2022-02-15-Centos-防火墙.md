---
layout: post
title: Centos 防火墙
category: Linux
tags: Linux
keywords: Linux 防火墙
description:
---

## Centos7

```bash
# 显示防火墙状态
sudo systemctl status firewalld.service

# 开启防火墙
sudo systemctl start firewalld.service
# 关闭防火墙
sudo systemctl stop firewalld.service

# 防火墙开机自己
sudo systemctl enable firewalld.service
# 关闭防火墙开机自己
sudo systemctl disable firewalld.service

# 显示防火墙规则
sudo firewall-cmd --list-all
# 修改防火墙规则配置--添加端口
sudo firewall-cmd --add-port=20006/tcp --permanent
# 重新加载防火墙规则配置
sudo firewall-cmd --reload
```
