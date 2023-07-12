---
layout: post
title: TVM  Windows 源码安装
category: Framework
tags: tvm
keywords: tvm
description: tvm
---

[官方文档](https://tvm.apache.org/docs/v0.11.0/install/from_source.html#install-from-source)

## 编译环境

- Windows 10 家庭中文版 22H2
- git 2.37.1
- Visual Studio 2017(14.16)
- cmake 3.21.5
- LLVM 8.0.0
- TVM 0.11.0

## 源码编译

1. 创建 python 环境

```bash
# 只是用 python3.7 或 python3.8. python3.9不支持
conda create --name tvm python=3.8
conda activate tvm
```

2. 修改配置文件

```bash
mkdir build
cp cmake/config.cmake build
```

修改 cmake 配置

```cmake
>72  set(USE_OPENCL "D:/Program Files/NVIDIA GPU Computing ToolKit/CUDA/v10.2")
>88  set(USE_VULKAN D:/ProgramData/VulkanSDK/1.3.250.0)
>145 set(USE_LLVM "D:/ProgramData/LLVM/8.0.0/bin/llvm-config.exe --link-static")
>146 set(HIDE_PRIVATE_SYMBOLS ON)
>221 set(USE_CUDNN "D:/Program Files/NVIDIA GPU Computing ToolKit/CUDA/v10.2")
```

3. python接口

```bash
# 安装
python setup.py install
# 清除
python setup.py clean --all
```

4. 验证

```bash
# Locate TVM Python package.
python -c "import tvm; print(tvm.__file__)"

# Confirm which TVM library is used.
python -c "import tvm; print(tvm._ffi.base._LIB)"

# Reflect TVM build option.
python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"

# Check device detection
python -c "import tvm; print(tvm.metal().exist)"

python -c "import tvm; print(tvm.cuda().exist)"

python -c "import tvm; print(tvm.vulkan().exist)"
```


## 异常汇总

1. tvm.driver.tvmc.config_options.ConfigsJsonNotFoundError

```bash
Traceback (most recent call last):
  File "D:\ProgramData\Anaconda3\envs\tvm-0.11.0\Scripts\tvmc-script.py", line 33, in <module>
    sys.exit(load_entry_point('tvm==0.11.0', 'console_scripts', 'tvmc')())
  File "D:\ProgramData\Anaconda3\envs\tvm-0.11.0\lib\site-packages\tvm-0.11.0-py3.7-win-amd64.egg\tvm\driver\tvmc\main.py", line 115, in main
    sys.exit(_main(sys.argv[1:]))
  File "D:\ProgramData\Anaconda3\envs\tvm-0.11.0\lib\site-packages\tvm-0.11.0-py3.7-win-amd64.egg\tvm\driver\tvmc\main.py", line 74, in _main
    json_param_dict = read_and_convert_json_into_dict(config_arg)
  File "D:\ProgramData\Anaconda3\envs\tvm-0.11.0\lib\site-packages\tvm-0.11.0-py3.7-win-amd64.egg\tvm\driver\tvmc\config_options.py", line 119, in read_and_convert_json_into_dict
    config_dir = get_configs_json_dir()
  File "D:\ProgramData\Anaconda3\envs\tvm-0.11.0\lib\site-packages\tvm-0.11.0-py3.7-win-amd64.egg\tvm\driver\tvmc\config_options.py", line 69, in get_configs_json_dir
    raise ConfigsJsonNotFoundError()
tvm.driver.tvmc.config_options.ConfigsJsonNotFoundError
```

解决方法

将 tvm 项目目录下 configs 文件夹复制到 python tvm包目录下（Lib\site-packages\tvm-0.11.0-py3.7-win-amd64.egg\tvm）

2. IMAGE_REL_AMD64_ADDR32NB relocation requires anordered section layout.

原因：LLVM版本过低，使用 LLVM 8.0.0
