---
layout: post
title: Linux 账户管理
category: Linux
tags: Linux
keywords: Linux 账户
description:
---

#### 添加普通用户

```bash
useradd user
```

#### 修改密码

- 要管理员权限或者本账号

- 作为管理员可以直接修改某个账号的密码，而不需要知道原密码。

    ```bash
    passwd user
    ```

#### 赋予 sudo 权限

给普通用户添加 sudo 权限，修改 /etc/sudoers 文件 (该文件默认为只读)

```bash
在“root ALL=(ALL)ALL”这一行下面，加入一行(用户名 ALL=(ALL) ALL)，并保存
```

#### 用户添加到组

```bash
gpasswd -a user_name group_name

# 更新用户组
newgrp group_name
```

#### 修改已存在的用户名

需求：将用户名 hadoop106 修改为 hadoop

其中 hadoop106 必须登出（注销），切换为 root 用户进行修改。
如果不登出 hadoop106, 那么在 hadoop106 中就会存在没有关闭的进程就会导致修改失败的问题。

1. 切换为 root 用户

    ```bash
    su root
    ```

2. 修改用户名及根目录

    ```bash
    vim /etc/passwd
    ```

    将 hadoop106 修改为 hadoop（一般在文件底部）

    <center>

    <img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Centos/1.png">

    图1

    </center>

3. 修改用户组

    ```bash
    vim /etc/group
    ```

    将 hadoop106 修改为 hadoop（一般在文件底部）

    <center>

    <img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Centos/2.png">

    图2

    </center>

4. 修改 /etc/shadow 文件

    ```bash
    vim /etc/shadow
    ```

    将 hadoop106 修改为 hadoop

    <center>

    <img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Centos/3.png">

    图3

    </center>

5. 修改用户的根目录

    ```bash
    mv /home/hadoop106/  /home/hadoop
    ```

6. 登录测试

    ```bash
    su hadoop
    ```