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

#ifndef COMMANDLINE_HPP_
#define COMMANDLINE_HPP_

#include <cstddef>    // size_t
#include <functional> // less<>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

// Parse the command line arguments received in the main() function.
// The argv[0] if exists is stored in the result's name field.
// All the arguments starting with '-' are collected in the options
// map till '--' is found.
// The rest is stored in the result's args vector sequentially.
// The options can have format 'name=value' or if there is no equal
// char, the value is taken from the next command line argument.
// In this later case, the value cannot start with a '-' character.

using CommandLineOptions = std::map<std::string, std::string_view, std::less<>>;
using CommandLineArgs = std::vector<std::string_view>;

struct CommandLineParseResult final {
  std::string_view name;
  CommandLineOptions options;
  CommandLineArgs args;
};

auto parseCommandLine(int argc, char* argv[]) -> CommandLineParseResult;

auto parsePositiveValue(std::string_view val) -> std::optional<std::size_t>;

auto parseUnsignedIntValue(std::string_view val) -> std::optional<uint32_t>;

auto parseProbabilityValue(std::string_view val) -> std::optional<float>;

auto parseSPositiveValue(std::string_view val) -> std::optional<float>;

auto parseNonNegativeValue(std::string_view val) -> std::optional<float>;

#endif //COMMANDLINE_HPP_