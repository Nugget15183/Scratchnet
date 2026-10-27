cmake_minimum_required(VERSION 4.0)
project(ai_learning)

set(CMAKE_CXX_STANDARD 20)

#build glm
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/packages/glm)

#build glad
add_library(glad STATIC packages/glad/src/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/packages/glad/include)

#build glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(packages/glfw-3.5.1)


add_executable(ai_learning src/main.cpp
        src/util.h
        src/Network.cpp
        src/Network.h
        src/Window.cpp
        src/Window.h)


target_include_directories(ai_learning PRIVATE src)
target_link_libraries(ai_learning PRIVATE glfw glad glm)