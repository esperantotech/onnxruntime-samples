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

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fnmatch.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "FileSystem.hpp" // readFile()
#include "PNGUtils.hpp"   // decodePNGImage()

#include "HelperFunctions.hpp"

auto loadImage(const std::filesystem::path& filename) -> RawImage {
  using namespace std::string_literals;
  if (!std::filesystem::exists(filename)) {
    throw BadImageFile(filename, "file does not exists");
  }
  const auto pngData = readFile(filename);
  auto res = decodePNGImage(pngData);
  if (const auto* errMsg = std::get_if<std::string>(&res)) {
    throw BadImageFile(filename, "error decoding PNG image data: " + *errMsg);
  }
  return std::get<RawImage>(res);
}

auto loadLabels(const std::string& filename) -> std::vector<std::string> {
  std::vector<std::string> output;

  std::ifstream file(filename);
  if (file.is_open()) {
    std::string s;
    while (getline(file, s)) {
      // skip entry 0
      if (s.find("background") == std::string::npos) {
        size_t pos = s.find(":");
        output.push_back(s.substr(pos + 1));
      }
    }
    file.close();
  }

  return output;
}

auto timestamp() -> std::string {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  const auto milli = (now.time_since_epoch().count() / 1000000) % 1000;
  struct tm local_time;
  std::ostringstream result;
  result.width(3);
  result.fill('0');
  result << std::put_time(localtime_r(&t, &local_time), "[%T.") << milli << "] ";
  return result.str();
}

auto directoryFiles(const std::filesystem::path& path, const char* pattern) -> std::vector<fs::path> {
  std::vector<fs::path> result;
  for (const auto& dirEntry : fs::directory_iterator(path)) {
    const auto& p = dirEntry.path();
    if (!fs::is_regular_file(p)) {
      continue;
    }
    if (0 == fnmatch(pattern, p.filename().string().c_str(), FNM_PATHNAME | FNM_PERIOD)) {
      result.emplace_back(p);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

auto checkImgExists(std::string imgName, std::vector<fs::path> imgFilesPath) -> bool {
  bool isInList = false;
  for (const auto& fname : imgFilesPath) {
    if (fname.generic_string().find(imgName) != std::string::npos) {
      isInList = true;
      break;
    }
  }
  return isInList;
}
