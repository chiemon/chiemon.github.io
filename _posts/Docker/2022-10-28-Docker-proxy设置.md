---
layout: post
title: Docker HTTP/HTTPS proxy
category: Docker
tags: docker
keywords: docker
description:
---

1. 创建目录

  ```bash
  sudo mkdir -p /etc/systemd/system/docker.service.d
  sudo touch /etc/systemd/system/docker.service.d/proxy.conf
  sudo vim /etc/systemd/system/docker.service.d/proxy.conf
  ```

2. 创建文件

  ```bash
  # /etc/systemd/system/docker.service.d/proxy.conf
  [Service]
  Environment="HTTP_PROXY=http://wwwproxy.mysite.com:80"
  Environment="HTTPS_PROXY=http://wwwproxy.mysite.com:80"
  ```
  `Note:`HTTPS_PROXY 指向形式为 http:// 而不是 https:// 的地址

3. 重启服务

  ```bash
  sudo systemctl daemon-reload
  sudo systemctl restart docker.service
  ```

4. 验证

```bash
sudo systemctl show --property=Environment docker

Environment=HTTP_PROXY=http://wwwproxy.mysite.com:80 HTTPS_PROXY=http://wwwproxy.mysite.com:80
```

*参考连接：[https://docs.docker.com/config/daemon/systemd/#httphttps-proxy](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy)*
