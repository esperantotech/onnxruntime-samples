#------------------------------------------------------------------------------
# Copyright (C) 2024, Esperanto Technologies Inc.
# The copyright to the computer program(s) herein is the
# property of Esperanto Technologies, Inc. All Rights Reserved.
# The program(s) may be used and/or copied only with
# the written permission of Esperanto Technologies and
# in accordance with the terms and conditions stipulated in the
# agreement/contract under which the program(s) have been supplied.
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(et-onnxruntime-samples VERSION 1.0.0 LANGUAGES C CXX)


string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
option(ENABLE_STRICT_BUILD_TYPES "Whether project configuration should fail if 
    build_type is not one of the allowed values: Debug Release RelWithDebInfo 
    MinSizeRel" ON)
set(ET_DEFAULT_BUILD_TYPE "Release")
include(CommonProjectSettings)

include(GNUInstallDirs)

configure_file(Config.h.in Config.h)

find_package(fmt REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(gflags REQUIRED)

add_subdirectory(PngHelp)

add_executable(mnist mnist.cpp)

target_include_directories(mnist
    PRIVATE
        ${PROJECT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(mnist 
    PRIVATE 
        onnxruntime::onnxruntime 
        fmt::fmt
        gflags::gflags
        PngHelpLib::PngHelpLib
)

add_executable(resnet resnet.cpp)

target_include_directories(resnet
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${PROJECT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(resnet
    PRIVATE
        onnxruntime::onnxruntime
        fmt::fmt
        gflags::gflags
        PngHelpLib::PngHelpLib
)
