---
layout: post
title: git Workflow工作流程
category: Git
tags: git
keywords: git workflow
description:
---

## 1. Gitflow分支

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Git/3.jpg"/>

</center>

|分支|名称|作用|
|:---:|:---:|:---:|
|master|主分支|存储正式发布的历史|
|hotfix|上线分支|bug情急修复分支|
|release|发布分支|发布上线的时候用|
|develop|开发分支|功能的集成分支|
|feature|功能分支|开发新功能都会有对应的feature分支|

## 2. 长期分支 & 辅助分支

- git-flow流程中最主要的五个分支分别为master，release，develop，feature，hotfix。
- 长期分支：master，develop。
- 辅助分支：release，feature，hotfix。、
- 长期分支是相对稳定的分支，所有被认可的提交最终都要合并到这两个分支上。
- 辅助分支是工作需要临时开的分支，在完成他们的工作之后通常是可以删除的。

## 3. 分支概述

- **master:** 对外发布产品使用的分支，该分支的提交必须是最接近对外上线的版本，不允许在该分支上进行开发，要始终保持该分支的稳定。
- **develop:** 内部开发产品所用的分支，该分支的最新提交必须是一个相对稳定的测试版本，同样地，不允许在该分支上面进行开发

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Git/4.png"/>

</center>

- **feature:** 新功能分支，每个新的功能都应该创建一个独立的分支，从develop分支派生出来，功能开发完成之后合并到develop分支，不允许功能未开发完成便合并到develop分支。新功能提交应该从不直接与master分支交互。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Git/5.png"/>

</center>

- **release:** 发布前的测试分支，一旦开发的功能满足发布条件或者预定发布日期将近，应该合并所有的功能分支到develop分支，并在develop分支开出一个release分支，在这个分支上，不能在添加新的功能，只能修复bug，一旦到了发布日期，该分支就要合并到master和develop分支，并且打出版本的标签。 另外，这些从新建发布分支以来的做的修改要合并回develop分支

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Git/6.png"/>

</center>

- **hotfix:** 修复分支，用于给产品发布版本快速生成补丁, 在master上创建的分支, 这是唯一可以直接从master分支fork出来的分支, 修复问题后，它应该合并回master和develop分支，然后在master分支上打一个新的标签。

<center>

<img src="https://raw.githubusercontent.com/chiemon/chiemon.github.io/master/img/Git/7.png"/>

</center>

## 4. 开发流程

### 4.1 创建远程仓库，并拉到本地

创建远程仓库的时候默认是创建master分支的，因此拉下来的项目也处于master分支。
```bash
$ git clone ...
```

### 4.2 创建develop分支

因为master分支上面是不允许进行开发的，创建长期开发分支develop

- 创建方式一:<br>远程仓库先创建分支, 在本地创建分支, 并关联远程分支
```bash
# 实现一
$ git checkout -b develop
$ git branch --set-upstream develop/origin develop
# 实现二
$ git checkout -b develop origin/develop    # 创建的同时就关联远程仓库
# 如果报错, 执行下面命令, 再输入该命令
$ git fetch
# 实现三
$ git fetch origin develop:develop
$ git branch --set-upstream-to=origin/develop develop
```

- 创建方式二:<br>本地创建分支, 在推送到远程仓库
```bash
$ git checkout -b develop
$ git push origin develop:develop
```

- 开发负责人本地创建develop分支，并推送到远程。
- 其他团队人员克隆后拉取develop分支，此时建议采用实现方式三拉取下来，本地创建分支并关联远程仓库。

### 4.3 开发新功能

- 假如开发新功能a，在develop分支创建新功能分支a
```bash
$ git checkout develop
$ git checkout -b feature/a
```

- 如果有必要，将该功能分支推送到远程
```bash
$ git push origin feature/a:feature/a
```

- 如果有必要，成员可将该分支拉下来
```bash
$ git fetch origin feature/a:feature/a
```

### 4.4 完成新功能
- 新功能完成之后需要将feature分支合并到develop分支，并push到远程仓库(在push之前，建议先pull一下，将本地的develop分支内容更新到最新版本，再push，避免线上版本与你commit时候文件内容产生冲突)
```bash
$ git checkout develop
$ git merge --no-ff feature/a   # --no-ff 参数可以保存feature/a分支上的历史记录
$ git push origin develop
```
- 合并完成之后，确定该分支不再使用，则删除本地和远程上的该分支
```bash
$ git branch -d feature/a
$ git push origin --delete feature/a
```

### 4.5 发布新功能

当新功能基本完成之后，我们要开始在release分支上测试新版本，在此分支上进行一些整合性的测试，并进行小bug的修复以及增加例如版本号的一些数据。版本号根据 master 分支当前最新的tag来确定即可，根据改动的情况选择要增加的位.

- 开发布分支
```bash
$ git checkout -b release/1.2.0 # 在develop分支中开
$ git push origin release/1.2.0:release/1.2.0   # 将分支推送到远程(如果有必要)
```

- 保证本地的release分支处于最新状态
```bash
$ git pull origin release/1.0.0 # 将本地的release分支内容更新为线上的分支
```

- 制定版本号
```bash
# commit 一个版本, commit的信息为版本升到1.2.0
# git commit -a 相当于git add . 再git commit
$ git commit -a -m "Bumped version number to 1.2.0"
```

- 将已制定好版本号等其他数据和测试并修复完成了一些小bug的分支合并到主分支
```bash
$ git checkout master   # 切换至主要分支
$ git merge --no-ff release/1.2.0   # 将release/1.2.0分支合并到主要分支
$ git tag -a "1.2.0" HEAD -m "新版本改动描述"   # 上标签
```

- 将release分支合并回开发分支
```bash
$ git checkout develop  # 切换至开发分支
$ git merge --no-ff release/1.2.0   # 合并分支
```

- 推送到远程仓库
```bash
$ git push origin develop   # 将开发分支推送到远程
$ git push origin master    # 将 master 分支推送到远程
```

- 删除分支
```bash
$ git branch -d release/1.2.0
$ git push origin --delete release/1.2.0
```

### 4.6 修补线上Bug

- 此修复bug针对的是线上运行的版本出现了bug，急需短时间修复，无法等到下一次发布才修复，区别于开发过程中develop上的bug，和测试过程中的release上的bug，这些bug，在原分支上改动便可以。

- 在master根据具体的问题创建hotifix分支，并推送到远程
```bash
$ git checkout master
$ git checkout -b hotfix/typo
$ git push origin hotfix/typo:hotfix/typo
```

- 制定版本号，一般最后位加1
```bash
# commit一个版本, commit的信息是版本条
$ git commit -a -m "Bumped version number to 1.2.1"
```

- 修正后commit并将本地的hotfix分支更新为线上最新的版本
```bash
$ git commit -m "..."
$ git pull origin hotfix/typo
```

- 将刚修复的分支合并到开发分支和主分支
```bash
$ git checkout develop  # 切换到开发分支
$ git merge --no-ff hotfix/typo # 合并
$ git checkout master   # 切换到主要分支
$ git merge --no-ff hotfix/typo # 将hotfix分支合并到主要分支
$ git tag -a "1.2.1" HEAD -m "fix typo" # 上标签
```

- 删除修补分支

## 5. 命名约定

- 主分支名称：master
- 主开发分支名称：develop
- 新功能开发分支名称：feature-…/feature/…，其中…为新功能简述
- 发布分支名称：release-…/release/…，其中…为版本号。
- bug修复分支名称：hotfix-…/hotfix/…，其中…为bug简述。

## 6. 附加Git的冲突

- 当我们需要将本地的分支push到远程的时候，举例：当我们新功能开发完成之后，我们合并到develop分支，要将develop分支push到远程的时候，此时如果远程的develop分支的内容有更新，我们就需要使用git pull命令将本地的develop分支更新到最新的版本，再推送，否则会产生冲突，无法推送。

- 第一种情况下的pull操作可能也会产生冲突，如果我们本地修改和新commit的内容修改了同一个文件同个位置，此时就应该进行开发者间协商。

- 当我们合并分支的如果两个分支同时修改了同个文件同个位置时候也会产生冲突，此时需要进行手动解决冲突。
