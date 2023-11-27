// Copyright (C) 2023 Skye Zhang (skai-zhang@hotmail.com)
// Created: Skye Zhang 2023-11-22
// License: AGPL-3.0
#include <iostream>
#include "src/http_server.h"

int main() {
    cout << "[MA] Service is starting" << endl;
    HttpServer http_server;
    http_server.startServer();
}
