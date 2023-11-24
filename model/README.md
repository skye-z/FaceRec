# Face Recognition Model

请访问[dlib-models](https://github.com/davisking/dlib-models), 下载以下两个模型
* shape_predictor_68_face_landmarks: 关键点预测模型, 下载解压后请重命名为`predictor.dat`
* dlib_face_recognition_resnet_model_v1: 人脸识别模型, 下载解压后请重命名为`recognition.dat`

上述操作完成后, 项目目录结构应该如下
```
FaceRec
    cpp-httplib/
        ...略
    nlohmann/
        json.hpp
    model/
        predictor.dat
        recognition.dat
    network_model.h
    http_server.h
    main.cpp
```