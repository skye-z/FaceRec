FROM alpine:latest

WORKDIR /app

# 安装依赖库和构建工具
RUN apk update && \
    apk add --no-cache build-base cmake git sqlite-dev

# 克隆dlib源代码并构建
RUN git clone -b 'v19.24.2' --single-branch https://mirror.ghproxy.com/https://github.com/davisking/dlib.git && \
    cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_AVX_INSTRUCTIONS=1 -DUSE_SSE4_INSTRUCTIONS=1 && \
    cmake --build . --config Release && \
    make install && \
    cd ../.. && \
    rm -rf dlib

# 复制代码文件到工作目录
COPY . /app/build

# 编译代码
RUN cd build && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make

# 清理文件
RUN mv /app/build/build/face_rec /app/face_rec && \
    mv /app/build/model /app/model && \
    cd /app && \
    rm -rf /app/build

# 暴露服务端口
EXPOSE 8080

# 运行服务
CMD ["/app/face_rec"]
