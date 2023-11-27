// Copyright (C) 2023 Skye Zhang (skai-zhang@hotmail.com)
// Created: Skye Zhang 2023-11-23
// License: AGPL-3.0
#pragma once

#include "../cpp-httplib/httplib.h"
#include "../nlohmann/json.hpp"
#include "face_util.h"
#include "face_data.h"

using json = nlohmann::json;
using namespace httplib;
using namespace std;

class HttpServer
{
public:
    HttpServer()
    {
        // 初始化数据库
        if (data.init())
        {
            cout << "[DB] Database started" << endl;
        }
        else
        {
            cout << "[DB] Database initialization failed" << endl;
        }

        // 设置HTTP路由
        server.Post("/add", [&](const Request &req, Response &res)
                    { handleAddFace(req, res); });
        server.Post("/remove", [&](const Request &req, Response &res)
                    { handleRemoveFace(req, res); });
        server.Post("/exists", [&](const Request &req, Response &res)
                    { handleExistsFace(req, res); });
        server.Post("/match", [&](const Request &req, Response &res)
                    { handleMatchFace(req, res); });
        cout << "[HS] Route mounted" << endl;
    }

    void startServer()
    {
        cout << "[HS] Http server started" << endl;
        // 启动 HTTP 服务器
        server.listen("0.0.0.0", 8080);
    }

private:
    // 添加人脸数据
    void handleAddFace(const Request &req, Response &res)
    {
        json result_json;
        try
        {
            // 检查是否有文件上传
            if (req.has_file("file") && req.has_param("uid"))
            {
                const auto &file = req.get_file_value("file");
                std::string uid = req.get_param_value("uid");
                if (data.exists(uid))
                {
                    httpReturnError(res, result_json, "UID已入库,请删除后再试", 200);
                }
                else
                {
                    cout << "[AF] Get face descriptors" << endl;
                    // 获取文件数据
                    std::string image_data(file.content.data(), file.content.length());
                    // 提取人脸特征
                    std::vector<matrix<float, 0, 1>> face_descriptors = util.getFaceDescriptors(image_data);
                    if (face_descriptors.size() == 0)
                    {
                        httpReturnError(res, result_json, "未检测到人脸", 200);
                    }
                    else if (data.save(uid, face_descriptors[0]))
                    {
                        httpReturnSuccess(res, result_json, "人脸已录入");
                    }
                    else
                    {
                        httpReturnError(res, result_json, "人脸数据保存失败", 200);
                    }
                }
            }
            else
            {
                httpReturnError(res, result_json, "未上传文件或未提供UID", 400);
            }
        }
        catch (const std::exception &e)
        {
            httpReturnError(res, result_json, "服务出错", 500);
        }
    }

    // 删除人脸数据
    void handleRemoveFace(const Request &req, Response &res)
    {
        json result_json;
        try
        {
            // 检查是否提供uid
            if (req.has_param("uid"))
            {
                std::string uid = req.get_param_value("uid");
                if (data.remove(uid))
                {
                    httpReturnSuccess(res, result_json, "人脸已删除");
                }
                else
                {
                    httpReturnError(res, result_json, "人脸删除失败", 200);
                }
            }
            else
            {
                httpReturnError(res, result_json, "未提供UID", 400);
            }
        }
        catch (const std::exception &e)
        {
            httpReturnError(res, result_json, "服务出错", 500);
        }
    }

    // 查询人脸是否存在
    void handleExistsFace(const Request &req, Response &res)
    {
        json result_json;
        try
        {
            // 检查是否提供uid
            if (req.has_param("uid"))
            {
                std::string uid = req.get_param_value("uid");
                if (data.exists(uid))
                {
                    httpReturnSuccess(res, result_json, "UID已录入人脸");
                }
                else
                {
                    httpReturnError(res, result_json, "UID未录入人脸", 200);
                }
            }
            else
            {
                httpReturnError(res, result_json, "未提供UID", 400);
            }
        }
        catch (const std::exception &e)
        {
            httpReturnError(res, result_json, "服务出错", 500);
        }
    }

    // 匹配人脸
    void handleMatchFace(const Request &req, Response &res)
    {
        json result_json;
        try
        {
            // 检查是否有文件上传
            if (req.has_file("file"))
            {
                float threshold = 0.3;
                if (req.has_param("valve"))
                {
                    try
                    {
                        std::string valve = req.get_param_value("valve");
                        threshold = std::stof(valve);
                    }
                    catch (const std::invalid_argument &eia)
                    {
                        threshold = 0.3;
                    }
                }
                const auto &file = req.get_file_value("file");
                cout << "[MF] Get face descriptors" << endl;
                // 获取文件数据
                std::string image_data(file.content.data(), file.content.length());
                // 提取人脸特征
                const std::vector<matrix<float, 0, 1>> face_descriptors = util.getFaceDescriptors(image_data);
                if (face_descriptors.size() == 0)
                {
                    httpReturnError(res, result_json, "未检测到人脸", 200);
                }
                else
                {
                    // 获取所有存储的人脸数据
                    std::vector<FaceData::FaceObject> face_list = data.all_list();
                    if (face_list.empty())
                    {
                        httpReturnError(res, result_json, "服务尚未初始化", 200);
                    }
                    else
                    {
                        // 计算欧式距离并找到最小距离对应的标识符
                        float min_distance = std::numeric_limits<float>::max();
                        std::string uid = "";

                        for (const FaceData::FaceObject &obj : face_list)
                        {
                            float distance = util.euclideanDistance(obj.face, face_descriptors[0]);
                            if (distance < min_distance && distance <= threshold)
                            {
                                min_distance = distance;
                                uid = obj.uid;
                            }
                        }
                        if (uid.empty())
                        {
                            httpReturnError(res, result_json, "无匹配", 200);
                            cout << "[MF] No match\n"
                                 << endl;
                        }
                        else
                        {
                            httpReturnSuccess(res, result_json, uid);
                            cout << "[MF] Match UID-" << uid << "\n"
                                 << endl;
                        }
                    }
                }
            }
            else
            {
                httpReturnError(res, result_json, "未上传文件", 400);
            }
        }
        catch (const std::exception &e)
        {
            httpReturnError(res, result_json, "服务出错", 500);
        }
    }

    void httpReturnError(Response &res, json result_json, std::string message, int code)
    {
        result_json["state"] = false;
        result_json["result"] = message;
        res.status = code;
        res.set_content(result_json.dump(), "application/json");
    }

    void httpReturnSuccess(Response &res, json result_json, std::string message)
    {
        result_json["state"] = true;
        result_json["result"] = message;
        res.status = 200;
        res.set_content(result_json.dump(), "application/json");
    }

private:
    Server server;
    FaceUtil util;
    FaceData data;
};
