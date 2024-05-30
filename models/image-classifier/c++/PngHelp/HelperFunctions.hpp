/*-------------------------------------------------------------------------
 * Copyright (C) 2024, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _MY_HELP_
#define _MY_HELP_

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "RawImage.hpp"

#define INFO(args_)                                                                                                    \
  do {                                                                                                                 \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                                                                   \
    std::cout << timestamp() << args_ << std::endl;                                                                    \
  } while (false)

#define VERBOSE(args_)                                                                                                 \
  if (g_verbose) {                                                                                                     \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                                                                   \
    std::cout << timestamp() << args_ << std::endl;                                                                    \
  } else                                                                                                               \
    ((void)0)

// Some namespace and type names simplifications
namespace fs = std::filesystem;

class BadImageFile : public std::runtime_error {
  std::string what_;

public:
  explicit BadImageFile(const std::string& filename, const std::string& what)
    : std::runtime_error(what)
    , what_("file '" + filename + "': " + std::runtime_error::what()) {
  }
  const char* what() const noexcept override {
    return what_.c_str();
  }
};

auto loadImage(const std::filesystem::path& filename) -> RawImage;
auto loadLabels(const std::string& filename) -> std::vector<std::string>;
auto timestamp() -> std::string;
auto directoryFiles(const std::filesystem::path& path, const char* pattern) -> std::vector<fs::path>;

#endif //_MY_HELP_