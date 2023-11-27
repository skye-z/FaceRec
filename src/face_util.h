// Copyright (C) 2023 Skye Zhang (skai-zhang@hotmail.com)
// Created: Skye Zhang 2023-11-27
// License: AGPL-3.0
#pragma once

#include <cstdio>
#include <dlib/clustering.h>
#include <dlib/matrix.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "network_model.h"

using namespace dlib;
using namespace std;

template <typename image_type>
void preprocess_face_image(image_type &img)
{
    // 调整图像大小
    long original_width = img.nc();
    long original_height = img.nr();
    if (original_width > 360)
    {
        double scale_factor = 360.0 / original_width;
        long new_width = static_cast<long>(original_width * scale_factor);
        long new_height = static_cast<long>(original_height * scale_factor);

        array2d<rgb_pixel> resized_img(new_height, new_width);
        resize_image(img, resized_img);
        assign_image(img, resized_img);
    }
    // 转灰度图
    matrix<unsigned char> gray_img;
    assign_image(gray_img, img);
    assign_image(img, gray_img);
    dlib::save_jpeg(img, "data/preview.jpg");
}

class FaceUtil
{
public:
    FaceUtil()
    {
        // 初始化人脸检测器
        detector = get_frontal_face_detector();
        // 初始化人脸关键点预测器
        deserialize("model/predictor.dat") >> sp;
        // 初始化人脸识别模型
        deserialize("model/recognition.dat") >> net;
        cout << "[FU] Model loaded" << endl;
    }

    // 获取人脸特征
    std::vector<matrix<float, 0, 1>> getFaceDescriptors(std::string image_data)
    {
        std::string temp_name = getTempFileName();
        try
        {
            // 将图像数据写入临时文件
            std::ofstream temp_file(temp_name, std::ios::binary);
            temp_file.write(image_data.c_str(), image_data.length());
            temp_file.close();
            // 从临时文件加载图像
            array2d<rgb_pixel> img;
            load_image(img, temp_name);
            preprocess_face_image(img);
            // 定义人脸图像矩阵向量
            std::vector<matrix<rgb_pixel>> faces;
            // 使用人脸检测器返回检测到每张人脸
            for (auto face : detector(img))
            {
                // 提取人脸的关键点
                auto shape = sp(img, face);
                // 定义人脸图像矩阵
                matrix<rgb_pixel> face_chip;
                // 提取人脸图像块
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                // 加入向量
                faces.push_back(std::move(face_chip));
            }
            // 识别完成后删除临时文件
            std::remove(temp_name.c_str());
            // 判断是否检测到人脸
            if (faces.size() != 0)
            {
                // 提取人脸特征
                return net(faces);
            }
            return std::vector<matrix<float, 0, 1>>();
        }
        catch (const std::exception &e)
        {
            // 出错时删除临时文件
            std::remove(temp_name.c_str());
            return std::vector<matrix<float, 0, 1>>();
        }
    }

    // 计算欧式距离
    float euclideanDistance(const matrix<float, 0, 1> &vec1, const matrix<float, 0, 1> &vec2)
    {
        // 检查两个向量的维度是否相同
        if (vec1.size() != vec2.size())
        {
            return 1.0f;
        }
        // 计算欧氏距离
        matrix<float, 0, 1> diff = vec1 - vec2;
        float distance = length(diff);
        return distance;
    }
private:
    // 获取临时文件名称
    std::string getTempFileName()
    {
        std::time_t timestamp = std::time(nullptr);
        std::ostringstream temp_filename_stream;
        temp_filename_stream << "temp_" << std::put_time(std::localtime(&timestamp), "%Y%m%d%H%M%S") << ".jpg";
        return temp_filename_stream.str();
    }
private:
    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;
};