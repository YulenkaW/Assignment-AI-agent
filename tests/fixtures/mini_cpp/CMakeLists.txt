cmake_minimum_required(VERSION 3.16)
project(mini_cpp LANGUAGES CXX)

add_library(mini_json INTERFACE)
target_include_directories(mini_json INTERFACE include)

add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE mini_json)
