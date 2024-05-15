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
project(ImgUtilLib VERSION 1.0.1 DESCRIPTION "ImgUtilitites lib to manage Images from dataset")

include(GNUInstallDirs)

find_package(glow REQUIRED)
find_package(PNG REQUIRED)

add_library(ImgUtilLib SHARED CommandLine.cpp 
        ErrorMng.cpp
        FileSystem.cpp        
        PNGUtils.cpp
        PreprocessImageClassifier.cpp
        Types.cpp        
        Utils.cpp)

add_library(ImgUtilLib::ImgUtilLib ALIAS ImgUtilLib)

set_target_properties(ImgUtilLib PROPERTIES 
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER include/ImgUtilLib.h)

target_link_libraries(ImgUtilLib 
        PUBLIC
                glow::Api
        PRIVATE 
                PNG::PNG        
                glow::Base)

target_include_directories(ImgUtilLib
        PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR})