---
layout: post
title: Docker NVIDIA NGC 设置
category: Docker
tags: docker
keywords: docker
description:
---

# AMD64 Linux

1. 获取 API Key

- 页面 [https://ngc.nvidia.com/setup/api-key](https://ngc.nvidia.com/setup/api-key)

- 登录账户，选择 setup，点击 Generate API Key， 在页面最下面生成 Key， 复制

2. 安装 NGC

```bash
wget --content-disposition https://ngc.nvidia.com/downloads/ngccli_linux.zip && unzip ngccli_linux.zip && chmod u+x ngc-cli/ngc

find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5

sudo ln -s $(pwd)/ngc-cli/ngc /usr/local/bin/ngc

ngc config set
# 填写 API Key

# 卸载 dirname `which ngc` | xargs rm -r
```

*参考链接：[https://ngc.nvidia.com/setup/installers/cli](https://ngc.nvidia.com/setup/installers/cli)*
