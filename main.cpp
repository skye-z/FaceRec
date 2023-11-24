// #include <iostream>
// #include <fstream>
// #include <cstdio>
// #include <dlib/clustering.h>
// #include <dlib/string.h>
// #include <dlib/image_io.h>
// #include <dlib/image_processing.h>
// #include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/dnn.h>
// #include "cpp-httplib/httplib.h"
// #include "nlohmann/json.hpp"
// #include "network_model.h"
// #include "http_server.h"

// using json = nlohmann::json;

// using namespace httplib;
// using namespace dlib;
// using namespace std;

#include <iostream>
#include "http_server.h"

int main() {
    HttpServer http_server;
    http_server.startServer();
}
