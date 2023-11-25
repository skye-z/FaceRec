// Copyright (C) 2023 Skye Zhang (skai-zhang@hotmail.com)
// Created: Skye Zhang 2023-11-24
// License: AGPL-3.0
#pragma once

#include <cstdio>
#include <string>
#include <sqlite3.h>
#include "nlohmann/json.hpp"
#include <dlib/image_processing.h>

using json = nlohmann::json;

using namespace dlib;
using namespace std;

class FaceData
{
public:
    FaceData() : db(nullptr)
    {
        if (sqlite3_open("face.store", &db) != SQLITE_OK)
        {
            cout << "[DB] Database startup failed" << endl;
            db = nullptr;
        }
    }

    ~FaceData()
    {
        if (db)
        {
            // 关闭数据库
            sqlite3_close(db);
        }
    }

    struct FaceObject
    {
        std::string uid;
        matrix<float, 0, 1> face;
    };

    // 初始化数据库
    bool init()
    {
        if (!db)
        {
            cout << "[DB] Database not started" << endl;
            return false;
        }
        // 检查人脸表是否存在
        int column_count;
        int state = sqlite3_table_column_metadata(db, nullptr, "face", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (state == 0)
        {
            return true;
        }
        // 创建人脸表
        const char *create_table_sql = "CREATE TABLE face (\"id\" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,\"uid\" TEXT NOT NULL,\"face\" TEXT NOT NULL );";
        state = sqlite3_exec(db, create_table_sql, 0, 0, 0);
        return state == SQLITE_OK;
    }

    // 保存人脸数据
    bool save(const std::string &uid, const matrix<float, 0, 1> &face_descriptor)
    {
        if (!db)
        {
            cout << "[DB] Database not started" << endl;
            return false;
        }

        const char *insert_sql = "INSERT INTO face (\"uid\", \"face\") VALUES (?,?);";

        sqlite3_stmt *stmt;
        int state = sqlite3_prepare_v2(db, insert_sql, -1, &stmt, 0);
        if (state != SQLITE_OK)
        {
            return false;
        }
        // 绑定标识符
        state = sqlite3_bind_text(stmt, 1, uid.c_str(), -1, SQLITE_TRANSIENT);
        if (state != SQLITE_OK)
        {
            sqlite3_finalize(stmt);
            return false;
        }
        // 绑定特征数据
        state = sqlite3_bind_text(stmt, 2, matrix_to_string(face_descriptor).c_str(), -1, SQLITE_TRANSIENT);
        if (state != SQLITE_OK)
        {
            sqlite3_finalize(stmt);
            return false;
        }
        // 执行语句
        state = sqlite3_step(stmt);
        if (state != SQLITE_DONE)
        {
            sqlite3_finalize(stmt);
            return false;
        }
        // 完成语句
        sqlite3_finalize(stmt);
        cout << "[DB] Add face data UID-" << uid << endl;
        return true;
    }

    // 删除人脸数据
    bool remove(const std::string &uid){
        std::string sql = "DELETE FROM face WHERE uid = '" + uid + "';";
        int state = sqlite3_exec(db, sql.c_str(), 0, 0, 0);
        return state == SQLITE_OK;
    }

    // 获取全部列表
    std::vector<FaceObject> all_list()
    {
        std::vector<FaceObject> obj_list;

        const char *select_sql = "SELECT \"uid\", \"face\" FROM \"face\";";
        sqlite3_stmt *stmt;
        int state = sqlite3_prepare_v2(db, select_sql, -1, &stmt, 0);

        if (state != SQLITE_OK)
        {
            return obj_list;
        }

        // 逐行获取数据
        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            FaceObject obj;
            // 获取标识符
            obj.uid = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
            // 获取人脸特征
            std::string face = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
            dlib::matrix<float, 0, 1> loaded_face_descriptor;
            std::istringstream iss(face);
            float value;
            int num = 0;
            obj.face.set_size(128);
            while (iss >> value)
            {
                obj.face(num++) = value;
                if (iss.peek() == ',')
                {
                    iss.ignore();
                }
            }
            // 添加到列表
            obj_list.push_back(obj);
        }
        // 完成语句
        sqlite3_finalize(stmt);
        return obj_list;
    }

private:
    std::string matrix_to_string(const dlib::matrix<float, 0, 1> &mat)
    {
        std::ostringstream oss;
        for (long r = 0; r < mat.nr(); ++r)
        {
            for (long c = 0; c < mat.nc(); ++c)
            {
                oss << std::fixed << std::setprecision(8) << mat(r, c);
            }
            if (r < mat.nr() - 1)
            {
                oss << ",";
            }
        }
        return oss.str();
    }

private:
    sqlite3 *db;
};