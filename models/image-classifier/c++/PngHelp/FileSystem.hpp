/*-------------------------------------------------------------------------
 * Copyright (C) 2021, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#pragma once

#if __has_include(<filesystem>)
#include <filesystem>
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
// NOLINTNEXTLINE(cert-dcl58-cpp)
namespace std {
// NOLINTNEXTLINE(misc-unused-alias-decls)
namespace filesystem = std::experimental::filesystem;
} // namespace std
#else
#error "cannot include the filesystem library"
#endif

#include <vector>

auto readFile(const std::filesystem::path& filename) -> std::vector<char>;
void writeFile(const std::filesystem::path& filename, std::vector<char> data);
