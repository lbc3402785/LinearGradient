cmake_minimum_required(VERSION 3.5)
project(MyProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
# 添加 /bigobj 编译选项
if(MSVC)
  add_compile_options(/bigobj)
endif()
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
# Find Eigen3 in your system
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
# Add the absolute include directory to the include paths
include_directories(D:/soft/eigen-3.4.0)

# Collect all the .cpp files in the current directory
file(GLOB SOURCES "./*.cpp")

# Create an executable from the .cpp files
add_executable(MyProject ${SOURCES})

# Link Eigen3 to the executable
target_link_libraries(MyProject Eigen3::Eigen Qt5::Core Qt5::Widgets)