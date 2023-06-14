---
layout: post
title: Windows平台 交叉编译 Android OpenCV
category: Android
tags: Android
keywords: android
description:
---

## 编译工具

cmake

[MinGW-W64](https://sourceforge.net/projects/mingw-w64/files/)

- version: MinGW-W64 GCC-5.4.0 x86_64-win32-sjlj

添加到系统环境变量`PATH`

[ant](https://archive.apache.org/dist/ant/binaries/)

- version: apache-ant-1.10.12-bin.zip


cmake_gui -> add entry

- ANDROID_ABI: arm64-v8a
- ANDROID_SDK: D:/ProgramData/Android/Sdk
- ANDROID_NDK: D:/ProgramData/Android/Sdk/ndk/17.2.4988734
- ANDROID_NATIVATE_API_LEVEL: 25
- ANDROID_STL: c++_shared
- ANT_EXECUTABLE: D:/Program Files/apache-ant-1.10.12/bin

```bash
CMake Error at cmake/android/OpenCVDetectAndroidSDK.cmake:184 (message):
  Android SDK: Can't build Android projects as requested by
  BUILD_ANDROID_PROJECTS=ON variable.

  Use BUILD_ANDROID_PROJECTS=OFF to prepare Android project files without
  building them
Call Stack (most recent call first):
  CMakeLists.txt:645 (include)
```

BUILD_ANDROID_PROJECTS:  ON -> OFF


添加 opencv-contrib

- OPENCV_EXTRA_MODULES_PATH: D:/ProgramData/Android/opencv/opencv_contrib-3.4.3/modules
- BUILD_ANDROID_PROJECTS=OFF
- BUILD_ANDROID_EXAMPLES=ON
- BUILD_PERF_TESTS=OFF
- BUILD_TESTS=OFF
- BUILD_opencv_world=OFF
- BUILD_SHARED_LIBS=OFF
- WITH_OPENCL=ON        移动端的并行架构支持
- WITH_OPENCL_SVM=ON    开启共享虚拟内存
