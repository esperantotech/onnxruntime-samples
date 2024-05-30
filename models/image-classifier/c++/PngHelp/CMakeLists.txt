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
project(PngHelpLib VERSION 1.0.1 DESCRIPTION "Png utilities lib to manage Images from dataset")

include(GNUInstallDirs)

find_package(PNG REQUIRED)

add_library(PngHelpLib SHARED FileSystem.cpp PNGUtils.cpp HelperFunctions.cpp)

add_library(PngHelpLib::PngHelpLib ALIAS PngHelpLib)

set_target_properties(PngHelpLib PROPERTIES 
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER include/PngHelpLib.h)

target_link_libraries(PngHelpLib 
        PRIVATE 
            PNG::PNG)

target_include_directories(PngHelpLib
        PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR})