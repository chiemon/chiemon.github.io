---
layout: post
title: OpenCV Mat操作
category: Software
tags: opencv
keywords: opencv
description:
---

# 使用 reshape 方法改变 4D Mat 形状

```cpp
#include <opencv2/opencv.hpp>

int main()
{
    // Create a 4D Mat with dimensions (3, 3, 3, 3)
    cv::Mat mat(3, 3, CV_32F, cv::Scalar(0));

    // Reshape the Mat to (3, 3, 9)
    mat = mat.reshape(3, cv::Size(3, 3, 9));

    // Check the new shape of the Mat
    std::cout << "mat.dims: " << mat.dims << std::endl;
    std::cout << "mat.size: " << mat.size << std::endl;

    return 0;
}
```

在此示例中，创建了一个尺寸为 (3, 3, 3, 3) 的 4D Mat，然后使用重塑方法将其重塑为 (3, 3, 9)。reshape 方法的第一个参数是重塑 Mat 中的通道数，在此示例中设置为 3。第二个参数是重塑后的 Mat 的大小，在本例中设置为 (3, 3, 9)。然后将 Mat 的新形状打印到控制台。

重要的是要注意，重塑 Mat 会改变内存中的底层数据布局，因此它可能会对性能产生影响，并且了解重塑的作用很重要。

# 更改 4D Mat 尺寸顺序

```cpp
#include <opencv2/opencv.hpp>

int main()
{
    // Create a 4D Mat with dimensions (3, 3, 3, 3)
    cv::Mat mat(3, 3, 3, 3, CV_32F, cv::Scalar(0));

    // Permute the dimensions of the Mat to (3, 3, 9, 3)
    cv::Mat permuted_mat;
    cv::transpose(mat, permuted_mat, cv::Vec4i(0, 2, 3, 1));

    // Check the new shape of the Mat
    std::cout << "permuted_mat.dims: " << permuted_mat.dims << std::endl;
    std::cout << "permuted_mat.size: " << permuted_mat.size << std::endl;

    return 0;
}
```

在此示例中，创建了一个尺寸为 (3, 3, 3, 3) 的 4D Mat，然后使用该方法将 Mat 的尺寸排列为 (3, 3, 9, 3) permute。该permute方法接受一个cv::Vec4i包含维度新顺序的对象。在此示例中，维度从 (0, 1, 2, 3) 置换为 (0, 2, 3, 1)。
重要的是要注意，排列 Mat 的维度不会改变内存中的底层数据布局，它只会改变数据的索引方式。当您想要更改数据的表示方式以用于特定操作或可视化目的时，这会很有用。


# 连接第二个维度 4D Mat

```c++
#include <opencv2/opencv.hpp>

int main()
{
    // Create two 4D Mats with dimensions (3, 3, 3, 3)
    cv::Mat mat1(3, 3, 3, 3, CV_32F, cv::Scalar(0));
    cv::Mat mat2(3, 3, 3, 3, CV_32F, cv::Scalar(1));

    // Concatenate the Mats along the second dimension
    cv::Mat concatenated_mat;
    cv::hconcat(mat1, mat2, concatenated_mat);

    // Check the new shape of the concatenated Mat
    std::cout << "concatenated_mat.dims: " << concatenated_mat.dims << std::endl;
    std::cout << "concatenated_mat.size: " << concatenated_mat.size << std::endl;

    return 0;
}
```

在此示例中，创建了两个维度为 (3, 3, 3, 3) 的 4D Mat，然后使用该 hconcat 函数沿第二个维度连接起来。该 hconcat 函数的前两个参数是要连接的两个 Mats，第三个参数是输出 Mat。连接的新形状将是 (3, 6, 3, 3) 并且第二个 Mat 的信息将连接在第一个 Mat 的第二个维度中。
重要的是要注意输入 Mat 的维度必须相同，除了它们被连接的维度之外。此外，如果您想沿其他维度连接 Mat，则可以使用vconcat函数而不是hconcat.


# 将 4D Mat 按第二维度拆分

```c++
#include <opencv2/opencv.hpp>

int main()
{
    // Create a 4D Mat with dimensions (3, 6, 3, 3)
    cv::Mat mat(3, 6, 3, 3, CV_32F, cv::Scalar(0));

    // Define the range of columns to slice
    int start_col = 3;
    int end_col = 6;

    // Slice the Mat along the second dimension
    cv::Mat sliced_mat = mat.colRange(start_col, end_col);

    // Check the new shape of the sliced Mat
    std::cout << "sliced_mat.dims: " << sliced_mat.dims << std::endl;
    std::cout << "sliced_mat.size: " << sliced_mat.size << std::endl;

    return 0;
}
```

在此示例中，创建了一个维度为 (3, 6, 3, 3) 的 4D Mat，然后使用该colRange方法沿第二个维度进行切片。该colRange方法的第一个参数是切片的起始列索引，第二个参数是切片的
结束列索引。在此示例中，Mat 被从第 3 列切片到第 6 列，从而产生了一个维度为 (3, 3, 3, 3) 的新 Mat。
请务必注意，该colRange方法仅沿第二个维度对 Mat 进行切片，如果您想沿其他维度对它进行切片，则可以改用rowRange、depth和channels方法。该Mat对象也是原始数据的视图，因此对切片 Mat 的任何更改也会影响原始 Mat。


# 将 4D Mat 按第三维度拆分

```c++
#include <opencv2/opencv.hpp>

int main()
{
    // Create a 4D Mat with dimensions (3, 3, 6, 3)
    cv::Mat mat(3, 3, 6, 3, CV_32F, cv::Scalar(0));

    // Define the range of rows to slice
    int start_row = 3;
    int end_row = 6;

    // Slice the Mat along the third dimension
    cv::Mat sliced_mat = mat(cv::Range::all(), cv::Range::all(), cv::Range(start_row, end_row), cv::Range::all());

    // Check the new shape of the sliced Mat
    std::cout << "sliced_mat.dims: " << sliced_mat.dims << std::endl;
    std::cout << "sliced_mat.size: " << sliced_mat.size << std::endl;

    return 0;
}
```

在此示例中，创建了一个维度为 (3, 3, 6, 3) 的 4D Mat，然后使用该operator()方法沿第三个维度进行切片。该operator()方法的第一个参数是一个cv::Range对象，它定义了要切片的行的范围，在这个例子中它cv::Range::all()意味着所有行。第二个参数是另一个cv::Range对象，它定义了要切片的列的范围，在这个例子中它也是cv::Range::all(). 第三个参数是一个cv::Range对象，它定义了要切片的深度范围，在这个例子中，它cv::Range(start_row, end_row)表示第 3 到第 6 个深度。第四个参数是另一个cv::Range定义要切片的通道范围的对象，在本例中它也是cv::Range::all(). 新 Mat 的尺寸为 (3,3,3,3)
请务必注意，该 operator() 方法允许您沿任意维度对 Mat 进行切片，而且Mat对象是原始数据的视图，因此对切片 Mat 的任何更改也会影响原始 Mat。

