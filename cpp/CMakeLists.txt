cmake_minimum_required(VERSION 3.10)
project(MNIST_CPP_Clean)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(mnist_cpp main.cpp)
target_link_libraries(mnist_cpp "${TORCH_LIBRARIES}" ${OpenCV_LIBS})
