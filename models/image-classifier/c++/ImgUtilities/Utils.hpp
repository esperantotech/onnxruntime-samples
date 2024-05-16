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

#include <functional> // less<>
#include <map>
#include <optional>
#include <string>

// Mapping of symbols to values
// NOTE: Use specific less<void> Compare template argument type instead of default
// less<std::string> to have faster .find("...") calls as no conversion will be
// done from char * to std::string.
using SymbolsMap = std::map<std::string, std::string, std::less<>>;

// Add the specified indentation spaces at the beginning of each line in str.
auto indented(const std::string &str, unsigned indentation) -> std::string;

// Remove the given choppedChar from the ending of the string str.
auto chomp(const std::string &str, char choppedChar = '\n') -> std::string;

// Returns value as hex string, with zero left padding and in lowercase.
// That means that as post-condition the result size is always equal to 2,
// and either the first and the second string chars belongs to the subset
// [0-9a-f].
auto toHexString(unsigned char value) -> std::string;

// Do a variable expansion of '{var-name}' like expressions in the inpur string.
// The symbolsMap contains the vaues that will be replaced by the given var-name.
auto replaceSymbols(const std::string &str, const SymbolsMap &symbolsMap) -> std::string;

// Parse the string with mapping string values in a JSON like format.
// Format: '{' "key1": "value1", ... '}'
// Every token is optionally separated by zero or more blank characters.
// Return a nullopt is the input string has an invalid format.
auto parseStringMap(const std::string &s) -> std::optional<SymbolsMap>;

// Return a timestamp staing with the format: "[HH:MM:SS.mmm] "
auto timestamp() -> std::string;
