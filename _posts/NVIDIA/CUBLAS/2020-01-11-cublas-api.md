---
layout: post
title: cublas - api
category: nvidia
tags: cublas
keywords: cublas
description:
---

## 概述

矩阵乘法是高性能计算中最常用到一类计算模型。无论在HPC领域，例如做FFT、卷积、相关、滤波等，还是在 Deep Learning 领域，例如卷积层，全连接层等，其核心算法都直接或者可以转换为矩阵乘法。

cuBLAS 是标准线性代数库 (standard basic linear algebra subroutines (BLAS)) 的 GPU 加速实现，它支持 Level 1 (向量与向量运算) ，Level 2 (向量与矩阵运算) ，Level 3 (矩阵与矩阵运算) 级别的标准矩阵运算。

使用好 cuBLAS，首先要了解函数命令规则，找到自己算法对应的函数名称；其次要理解函数中输入/输出参数的定义，尤其是处理的数据类型，矩阵的存储方式 (行/列优先) ，运算类型等。

## 命名规则

关于 BLAS 库中各种函数命名规则，大家可以参考下面这个表：

|数据类型|矩阵类型|运算类型|
|:---|:---|:---|
|S: single(single real)  |GE: general          |DOT: scalar product, x^T y|
|D: double(double real)  |GB: general band     |AXPY: vector sum, alpha x + y|
|C: complex(complex real)|SY: symmetric        |MV: matrix-vector product, Ax|
|Z: double(double real)  |SB: symmetric band   |SV: matrix-vector solve, \|inv(A)x|
|                        |SP: symmetric packed |MM: matrix-matrix product, AB|
|                        |HE: hermitian        |SM: matrix-matrix solve, inv(A) B|
|                        |HB: hermitian band   |...|
|                        |HP: hermitian packed ||
|                        |TR: triangular       ||
|                        |TB: triangular band  ||
|                        |TP: triangular packed||
|                        |...||

cuBLAS 库的命令规则与 BLAS 库完全一致。有了上面的命令规则，我们查询 cuBLAS API 时就可以很容易了解各个函数的意义。

## 数据类型

cuBLAS 中主要包含2种数据类型：cublas 数据类型和 cuda 数据类型。

1. cublas 数据类型

- cublasOperation_t：对矩阵的操作方式为非转置，转置或共轭转置

    |Value|Meaning|
    |:---|:---|
    |CUBLAS_OP_N|the non-transpose operation is selected|
    |CUBLAS_OP_T|the transpose operation is selected|
    |CUBLAS_OP_C|the conjugate transpose operation is selected|

    注意，cublas中默认矩阵按列顺序存储的。

- cublasPointerMode_t：指针类型为 CPU 端指针或 GPU 端指针

    |Value|Meaning|
    |:---|:---|
    |CUBLAS_POINTER_MODE_HOST  |the scalars are passed by reference on the host|
    |CUBLAS_POINTER_MODE_DEVICE|the scalars are passed by reference on the device|

- cublasGemmAlgo_t：矩阵乘法的算法选择，是 cublasGemmEx 函数的参数 (参见后续调用例子)

    |Value|Meaning|
    |:---|:---|
    |CUBLAS_GEMM_DFALT  |Apply Heuristics to select the GEMM algorithm|
    |CUBLAS_GEMM_ALGO0 to CUBLAS_GEMM_ALGO17|Explicitly choose an Algorithm [0, 17]|
    |CUBLAS_GEMM_DFALT_TENSOR_OP|Apply Heuristics to select the GEMM algorithm, and allow the use of Tensor Core operations when possible|
    |CUBLAS_GEMM_ALGO0_TENSOR_OP to CUBLAS_GEMM_ALGO2_TENSOR_OP|Explicitly choose a GEMM Algorithm [0, 2] while allowing the use of Tensor Core operations when possible|

- cublasMath_t：指示是否调用tensor core (仅Volta GPU支持) 进行计算。通过cublasSetMathMode函数来设置

    |Value|Meaning|
    |:---|:---|
    |CUBLAS_DEFAULT_MATH  |Prevent the library from using Tensor Core operations|
    |CUBLAS_TENSOR_OP_MATH|Allows the library to use Tensor Core operations whenever possible|

2. cuda 数据类型——cudaDataType_t：指示数据类型为半精度、单精度、双精度、8bit整型，并可与实数、复数、有符号、无符号等进行组合

    |Value|Meaning|
    |:---|:---|
    |CUDA_R_16F|the data type is 16-bit floating-point|
    |CUDA_C_16F|the data type is 16-bit complex floating-point|
    |CUDA_R_32F|the data type is 32-bit floating-point|
    |CUDA_C_32F|the data type is 32-bit complex floating-point|
    |CUDA_R_64F|the data type is 64-bit floating-point|
    |CUDA_C_64F|the data type is 64-bit complex floating-point|
    |CUDA_R_8I |the data type is 8-bit signed integer|
    |CUDA_C_8I |the data type is 8-bit complex signed integer|
    |CUDA_R_8U |the data type is 8-bit unsigned integer|
    |CUDA_C_8U |the data type is 8-bit complex unsigned integer|

## 函数库

cuBLAS函数库包含2类——功能函数和数学计算函数：

1. 功能函数：

- cublasCreate / cublasDestroy：分别用于创建和释放cuBLAS context；

- cudaStreamCreate / cublasSetStream：创建和绑定流，可用于多个小型运算的overlap执行；

- cublasSetMatrix/cublasGetMatrix：从host向device做矩阵copy或者从device向host做矩阵copy；

- cublasSetMathMode /cublasGetMathMode：指示是否使用Volta GPU中的tensor core计算矩阵乘法。cuBLAS中共有6个函数可以支持tensor core的计算：cublasHgemm()， cublasHgemmBatched()，cublasHgemmStridedBatched()，cublasSgemm()，cublasSgemmEx()和cublasGemmEx()。对于不同的数据类型，tensor core支持的gemm函数情况如下：

    |Atype/Btype|Ctype|computeType|alpha/beta|Support Functions when<br>CUBLAS_TENSOR_OP_MATH<br>is set|
    |:---|:---|:---|:---|:---|
    |CUDA_R_16F|CUDA_R_32F|CUDA_R_32F|CUDA_R_32F|cublasGemmEx,<br>cublasSgemmEx()|
    |CUDA_R_16F|CUDA_R_16F|CUDA_R_32F|CUDA_R_32F|cublasGemmEx,<br>cublasSgemmEx()|
    |CUDA_R_16F|CUDA_R_16F|CUDA_R_16F|CUDA_R_16F|cublasHgemm(),<br>cublasHgemmBatched(),<br>cublasHgemmStridedBatched()|
    |CUDA_R_32F|CUDA_R_32F|CUDA_R_32F|CUDA_R_32F|cublasSgemm,<br>cublasGemmEx,<br>cublasSgemmEx()<br>NOTE: A conversion from<br>CUDA_R_32F to<br>CUDA_R_16F with<br>round to nearest<br>on the input values A/B<br>is performed when<br>Tensor are used|

2. 计算函数：

- Level 1/2/3计算函数：例如，针对 level 3 的矩阵乘法运算，有 cublas<t>gemm 实现普通的矩阵乘法，其中模板<t>表示输入/输出矩阵的类型可以是S (单精度)、D (双精度)、C (单精度复数)、Z (双精度复数)、H (半精度)；cublas<t>gemmBatched 实现基于 batch 的矩阵乘法 (要求子矩阵连续存储) ，或者 cublas<t>gemmStridedBatched 实现基于 batch 的矩阵乘法 (子矩阵有stride地址偏移) 。对于小规模的矩阵乘法，使用 batch 方式能够更好的利用 GPU 的计算资源，因此能够得到更好的性能；

- BLAS like扩展函数：对 level 3 级别矩阵与矩阵运算的扩展函数集。例如 cublasGemmEx 函数，可实现不同类型矩阵乘法的统一接口函数，也就是说，对于各种类型 (例如半精度、单精度、双精度、混合精度 (tensor core)、int8 整型，以及实数、复数等) 矩阵乘法，实际上你只需要会用这一个函数就足够了！

## 关于转置

用过cuBLAS做矩阵乘法的开发人员都知道，关于到底要不要转置，特别容易把人搞懵。这个问题产生的根本原因是cuBLAS默认矩阵在GPU中是按列顺序存储的，而我们通常在内存中存储矩阵的方式是按行存储。这意味着，当我们传入1个矩阵A时，cuBLAS会将它看做trans(A)即矩阵A的转置。为了给使用的灵活性（当然，灵活性的代价是增加了复杂性），cuBLAS的矩阵乘法函数中的参数cublasOperation_t允许对输入矩阵A和B进行CUBLAS_OP_N（不转置）、CUBLAS_OP_T（转置）或者CUBLAS_OP_C（共轭转置）的操作。

- 情况 1）：矩阵A和矩阵B的操作类型均为CUBLAS_OP_N（不转置），那么传入的矩阵A、B会被cuBLAS认作trans(A)、trans(B)，根据公式Trans(B) * Trans(A) = Trans(A*B) = Trans(C)，此时的计算结果Trans(C)按照列顺序存储在GPU中，对于按行顺序读取来说，刚好就是矩阵C。此时要注意2个输入矩阵的先后顺序是矩阵B在前，矩阵A在后；

- 情况 2）：矩阵A和矩阵B的操作类型均为CUBLAS_OP_T（转置），那么传入的矩阵A、B会被cuBLAS认作trans(A)、trans(B)，然后默认进行转置操作得到矩阵A和B，计算A*B=C，计算结果矩阵C按照列顺序存储在GPU中，对于按行顺序读取来说，得到的是矩阵trans(C)。此时，我们需要增加一步对输出矩阵的转置操作（或者直接按列读取），才能得到我们想要的矩阵C。

## 矩阵相乘

### 1. cublasSgemm 单精度实数

头文件: cublas_v2.h

|matrix|normal|cublas|
|:---:|:---:|:---:|
|A|m x k|k x m|
|B|k x n|n x k|
|C|m x n|n x m|

$$C_{m \times n} = \alpha*A_{m \times k}*B_{k \times n} + \beta*C_{m \times n}$$

cublasOperation_t
CUBLAS_OP_N 不转置
CUBLAS_OP_T 普通转置
CUBLAS_OP_C 共轭转置

cublasStatus_t cublasSgemm(
                cublasHandle_t handle,      // 调用 cuBLAS 库时的句柄
                cublasOperation_t transa,   // A 矩阵是否需要转置
                cublasOperation_t transb,   // B 矩阵是否需要转置
                int m,                      // A 的行数
                int n,                      // B 的列数
                int k,                      // A 的列数
                const float *alpha,         // 系数 α, host or device pointer
                const float *A,             // 矩阵 A 的指针，device pointer
                int lda,                    // 矩阵 A 的主维，if A 转置， lda = max(1, k), else max(1, m)
                const float *B,             // 矩阵 B 的指针, device pointer
                int ldb,                    // 矩阵 B 的主维，if B 转置， ldb = max(1, n), else max(1, k)
                const float *beta,          // 系数 β, host or device pointer
                float *C,                   // 矩阵 C 的指针，device pointer
                int ldc                     // 矩阵 C 的主维，ldc >= max(1, m)
                );
