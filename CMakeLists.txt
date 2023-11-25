# Copyright (C) 2023 Skye Zhang (skai-zhang@hotmail.com)
# Created: Skye Zhang 2023-11-24
# License: AGPL-3.0

cmake_minimum_required(VERSION 3.10)

project(face_rec)

set(CMAKE_CXX_STANDARD 14)

find_package(dlib REQUIRED)
find_package(Threads REQUIRED)
find_package(SQLite3 REQUIRED)

add_executable(face_rec main.cpp)

target_include_directories(face_rec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(face_rec PRIVATE dlib Threads::Threads SQLite::SQLite3)