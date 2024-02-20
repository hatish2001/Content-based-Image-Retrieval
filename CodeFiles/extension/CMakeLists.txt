cmake_minimum_required(VERSION 3.0)
project(extension)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Boost
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# Add executable
add_executable(extension extension.cpp)
target_link_libraries(extension ${OpenCV_LIBS} ${Boost_LIBRARIES})
