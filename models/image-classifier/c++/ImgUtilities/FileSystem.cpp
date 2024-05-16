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

#include "FileSystem.hpp"

#include <cassert>
#include <fstream>  // basic_ifstream<>
#include <ios>      // streamsize
#include <iterator> // advance()
#include <limits>

auto readFile(const std::filesystem::path &filename) -> std::vector<char> {
  const auto fileSize = std::filesystem::file_size(filename);
  std::vector<char> buff(fileSize);

  std::ifstream f(filename, std::ios::binary);
  f.exceptions(std::ios::failbit);

  auto numBytes = buff.size();
  auto *p = buff.data();
  for (const auto maxReadBytes = std::numeric_limits<std::streamsize>::max();
       maxReadBytes < numBytes; std::advance(p, maxReadBytes), numBytes -= maxReadBytes) {
    f.read(p, maxReadBytes);
  }
  assert(numBytes <= std::numeric_limits<std::streamsize>::max());
  f.read(p, static_cast<std::streamsize>(numBytes));

  return buff;
}

void writeFile(const std::filesystem::path &filename, std::vector<char> data) {
  const auto &dirname = filename.parent_path();
  if (not dirname.empty()) {
    std::filesystem::create_directories(dirname);
  }

  std::ofstream f(filename, std::ios::binary);
  f.exceptions(std::ios::failbit);

  auto numBytes = data.size();
  auto *p = data.data();
  for (const auto maxWriteBytes = std::numeric_limits<std::streamsize>::max();
       maxWriteBytes < numBytes; std::advance(p, maxWriteBytes), numBytes -= maxWriteBytes) {
    f.write(p, maxWriteBytes);
  }
  assert(numBytes <= std::numeric_limits<std::streamsize>::max());
  f.write(p, static_cast<std::streamsize>(numBytes));
}
