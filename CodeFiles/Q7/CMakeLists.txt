cmake_minimum_required(VERSION 3.0)
project(Question7)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Boost
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# Add executable
add_executable(Question7 Question7.cpp)

# Link OpenCV libraries
target_link_libraries(Question7 ${OpenCV_LIBS} ${Boost_LIBRARIES})

