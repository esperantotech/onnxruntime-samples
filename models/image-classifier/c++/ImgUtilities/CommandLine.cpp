/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include "CommandLine.hpp"
#include <algorithm> // none_of()
#include <cassert>
#include <iterator> // next(), advance()
#include <string_view>

namespace {
using namespace std::string_view_literals;
constexpr auto kNumericChar = "0123456789"sv;
} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
auto parseCommandLine(int argc, char* argv[]) -> CommandLineParseResult {
  // PRECONDITION
  assert(argc >= 0);
  assert(argv != nullptr);
  assert(std::none_of(argv + 1, argv + argc, [](const char* v) { return v == nullptr; }));

  CommandLineParseResult result;
  result.name = ((argc > 0) and (*argv != nullptr)) ? *argv : "";
  bool collectOptions = true;
  for (char** it = std::next(argv); it - argv < argc; std::advance(it, 1)) {
    if (*it == nullptr) {
      continue;
    }
    const std::string_view arg(*it);
    if ((arg[0] == '-') and collectOptions) {
      if (arg == "--") {
        collectOptions = false;
      } else if (const auto endNamePos = arg.find('='); endNamePos != std::string_view::npos) {
        const std::string option(arg.substr(0, endNamePos));
        const auto optionValue = arg.substr(endNamePos + 1);
        result.options.try_emplace(option, optionValue);
      } else {
        const std::string option(arg);
        if (char** itNext = std::next(it); (itNext - argv < argc) and (*itNext != nullptr) and (**itNext != '-')) {
          // Next argument is not an option
          it = itNext;
          const std::string_view optionValue(*it);
          result.options.try_emplace(option, optionValue);
        } else {
          // Option with no value
          result.options.try_emplace(option, "");
        }
      }
    } else {
      result.args.emplace_back(arg);
    }
  }
  return result;
}

auto parsePositiveValue(std::string_view val) -> std::optional<std::size_t> {
  if (val.empty()) {
    return std::nullopt;
  }
  if (std::string_view::npos != val.find_first_not_of(kNumericChar)) {
    return std::nullopt;
  }
  const auto result = std::stoull(std::string(val));
  if (result == 0) {
    return std::nullopt;
  }
  return result;
}

auto parseUnsignedIntValue(std::string_view val) -> std::optional<uint32_t> {
  if (val.empty()) {
    return std::nullopt;
  }
  if (std::string_view::npos != val.find_first_not_of(kNumericChar)) {
    return std::nullopt;
  }
  const auto result = std::stoul(std::string(val));
  if (result == 0) {
    return std::nullopt;
  }
  return result;
}

auto parseProbabilityValue(std::string_view val) -> std::optional<float> {
  float v = std::stof(std::string(val));
  if (v < 0 || v > 1) {
    return std::nullopt;
  }
  return v;
}

auto parseSPositiveValue(std::string_view val) -> std::optional<float> {
  float v = std::stof(std::string(val));
  if (v <= 0) {
    return std::nullopt;
  }
  return v;
}

auto parseNonNegativeValue(std::string_view val) -> std::optional<float> {
  float v = std::stof(std::string(val));
  if (v < 0) {
    return std::nullopt;
  }
  return v;
}
