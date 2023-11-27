# Face Recognition Service

[![](https://img.shields.io/badge/CPP-14+-%2300ADD8?style=flat&logo=cpp)](https://cplusplus.com/)
[![](https://img.shields.io/badge/Version-1.0.0-green)](control)

这是一个完整的人脸识别的HTTP API服务, 自带SQLite存储, 提供人脸数据录入和全库人脸自定义阈值匹配服务

## 编译构建
```shell
# 拉取 FaceRec 项目
git clone https://github.com/skye-z/FaceRec
# 进入 FaceRec 项目
cd FaceRec
# 拉取 json 库
curl -o nlohmann/json.hpp https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
# 拉取 httplib
git clone https://github.com/yhirose/cpp-httplib.git
# 构建
g++ -std=c++14 -I/usr/local/include -L/usr/local/lib -framework Accelerate -o face_rec main.cpp -ldlib -lpthread -lsqlite3
```

上述命令适合开发环境构建, 如果是生产环境可以使用cmake

```shell
mkdir build
cd build
cmake ..
make
```

## 手动运行

首先你需要进入`model`目录, 根据提示下载模型

```shell
# 直接运行
./face_rec
# Linux 后台运行(输出重定向到output.log)
nohup ./face_rec > output.log 2>&1 &
```

## 生产部署

编译构建后, 将产物上传至服务器, 然后在产物所在目录创建`model`目录, 将模型上传, 最后运行服务即可

当然也可以直接使用 Docker: `docker run -d -p 12680:8080 -v /data/face-rec:/app/data --name face-rec skyezhang/face-rec:1.0.2`

## HTTP接口

> 本服务所有接口均为POST请求

* /add 添加人脸数据
    * url参数: uid 用户编号
    * form数据: file 人脸图片
* /remove 删除人脸数据
    * url参数: uid 用户编号
* /exists 查询是否已录入人脸
    * url参数: uid 用户编号
* /match 匹配人脸
    * url参数: valve 阈值(默认0.3)
    * form数据: file 人脸图片

## 许可证

[GNU Affero General Public License v3.0](LICENSE)