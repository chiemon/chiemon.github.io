---
layout: post
title: Matplotlib 总结
category: Python
tags: python
keywords: matplotlib
description:
---

## 1. 配置（无GUI）

### 不显示图片

- python文件中修改

    ```python
    # from matplotlib import pylot 前添加
    import matplotlib
    matplotlib.use('Agg')

    from matplotlib import pylot
    ```

- 配置文件中修改

    ```bash
    # 创建 ~/.config/matplotlib/matplotlibrc

    backend : Agg   # 添加
    ```

### 保存图片

在 plt.draw() 或者 plt.show() 之后添加保存图片的代码

```python
plt.show()
plt.savefig('/path/1.png')  # 不支持 jpg 格式
```
