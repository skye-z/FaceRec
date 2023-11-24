// http_server.h
#pragma once

#include <cstdio>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"
#include "network_model.h"

using json = nlohmann::json;
using namespace httplib;
using namespace dlib;
using namespace std;

class HttpServer {
public:
    HttpServer() {
        // 初始化人脸检测器
        detector = get_frontal_face_detector();
        // 初始化人脸关键点预测器
        deserialize("predictor.dat") >> sp;
        // 初始化人脸识别模型
        deserialize("recognition.dat") >> net;

        // 设置HTTP路由
        server.Post("/detect_face", [&](const Request &req, Response &res) {
            handleDetectFace(req, res);
        });
    }

    void startServer() {
        // 启动 HTTP 服务器
        server.listen("localhost", 8080);
    }

private:
    // 获取临时文件名称
    std::string getTempFileName(){
        std::time_t timestamp = std::time(nullptr);
        std::ostringstream temp_filename_stream;
        temp_filename_stream << "temp_" << std::put_time(std::localtime(&timestamp), "%Y%m%d%H%M%S") << ".jpg";
        return temp_filename_stream.str();
    }
    // 处理检测人脸的HTTP请求
    void handleDetectFace(const Request &req, Response &res) {
        json result_json; 
        std::string temp_name = getTempFileName();
        try {
            // 检查是否有文件上传
            if (req.has_file("file")) {
                const auto& file = req.get_file_value("file");
                // 获取文件数据
                std::string image_data(file.content.data(), file.content.length());
                // 将图像数据写入临时文件
                std::ofstream temp_file(temp_name, std::ios::binary);
                temp_file.write(image_data.c_str(), image_data.length());
                temp_file.close();
                // 从临时文件加载图像
                array2d<rgb_pixel> img;
                load_image(img, temp_name);
                // 定义人脸图像矩阵向量
                std::vector<matrix<rgb_pixel>> faces;
                // 使用人脸检测器返回检测到每张人脸
                for (auto face : detector(img)) {
                    // 提取人脸的关键点
                    auto shape = sp(img, face);
                    // 定义人脸图像矩阵
                    matrix<rgb_pixel> face_chip;
                    // 提取人脸图像块
                    extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
                    // 加入向量
                    faces.push_back(std::move(face_chip));
                }
                // 判断是否检测到人脸
                if (faces.size() == 0) {
                    result_json["state"] = false;
                    result_json["result"] = "图中未检测到人脸";
                } else {
                    // 提取人脸特征
                    std::vector<matrix<float,0,1>> face_descriptors = net(faces);
                    // 返回特征
                    result_json["state"] = true;
                    result_json["result"] = face_descriptors[0];
                }
                res.set_content(result_json.dump(), "application/json");
                // 识别完成后删除临时文件
                std::remove(temp_name.c_str());
            } else {
                result_json["state"] = false;
                result_json["result"] = "未上传文件";

                res.status = 400;
                res.set_content(result_json.dump(), "application/json");
            }
        } catch (const std::exception& e) {
            result_json["state"] = false;
            result_json["result"] = "服务出错";

            res.status = 500;
            res.set_content(result_json.dump(), "application/json");
            // 出错时删除临时文件
            std::remove(temp_name.c_str());
        } 
    }

private:
    Server server;
    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;
};
