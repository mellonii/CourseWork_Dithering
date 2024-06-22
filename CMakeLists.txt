cmake_minimum_required(VERSION 3.10)
project(Dithering_Ishkuvatova)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "C:/opencv/build/x64/vc16")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Dithering_Ishkuvatova dithering.cpp)
target_link_libraries(Dithering_Ishkuvatova ${OpenCV_LIBS})

file(COPY ${CMAKE_SOURCE_DIR}/main_test.jpg DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/test.jpg DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/test2.jpg DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/test3.jpg DESTINATION ${CMAKE_BINARY_DIR})