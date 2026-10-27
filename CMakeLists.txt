cmake_minimum_required(VERSION 4.0)
project(ai_learning)

set(CMAKE_CXX_STANDARD 20)

add_executable(ai_learning src/main.cpp
        src/util.h
        src/Network.cpp
        src/Network.h)
