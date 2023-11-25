// Copyright (C) 2023 Skye Zhang (skai-zhang@hotmail.com)
// Created: Skye Zhang 2023-11-23
// License: AGPL-3.0
#pragma once

#include <dlib/dnn.h>

using namespace dlib;

// ---- 深度学习神经网络模块 -------------------------------
/*
标准残差块
-----------------
block   模板参数
N       通道数
BN      归一化类型
SUBNET  输入类型
*/
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
// 计算前将输入添加到块输出
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;
/*
降采样残差块
-----------------
block   模板参数
N       通道数
BN      归一化类型
SUBNET  输入类型
*/
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
// 使用平均池化进行下采样
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;
/*
基础块
-----------------
N       通道数
BN      归一化类型
stride  步幅
SUBNET  输入类型
*/
template <int N, template <typename> class BN, int stride, typename SUBNET>
// 加入卷积层、ReLU激活函数和批量归一化层
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;
/*
残差层
-----------------
N       通道数
SUBNET  输入类型
*/
template <int N, typename SUBNET>
// 使用标准残差块传入基础块、通道数、仿射变换和输入类型
using ares = relu<residual<block, N, affine, SUBNET>>;
/*
下采样残差层
-----------------
N       通道数
SUBNET  输入类型
*/
template <int N, typename SUBNET>
// 使用下采样残差块传入基础块、通道数、仿射变换和输入类型
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;


// ---- 深度学习神经网络层级 -------------------------------
template <typename SUBNET>
// 第一层: 256通道下采样残差层
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
// 第二层: 256通道x3, 两次残差后下采样
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
// 第三层: 128通道x3, 两次残差后下采样
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
// 第四层: 64通道x4, 三次残差后下采样
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
// 第五层: 32通道三次残差
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

/*
五层神经网络(含最大池化)
--------------------
输出128维的人脸特征向量
*/
using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                                                  alevel0<
                                                      alevel1<
                                                          alevel2<
                                                              alevel3<
                                                                  alevel4<
                                                                      max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;
