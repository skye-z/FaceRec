# Face Recognition Service

## 编译构建
```shell
# 拉取 FaceRec 项目
git clone https://github.com/skye-z/FaceRec
# 拉取 httplib
git clone https://github.com/yhirose/cpp-httplib.git
# 构建
g++ -std=c++14 -I/usr/local/include -L/usr/local/lib -framework Accelerate -o face_rec main.cpp -ldlib -lpthread -lsqlite3
```

## 运行

首先你需要进入`model`目录, 根据提示下载模型

```shell
# 直接运行
./face_rec
# Linux 后台运行(输出重定向到output.log)
nohup ./face_rec > output.log 2>&1 &
```

## 部署

编译构建后, 将产物上传至服务器, 然后在产物所在目录创建`model`目录, 将模型上传, 最后运行服务即可