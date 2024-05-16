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

#include "Utils.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>   // size_t
#include <cstdio>    // snprintf()
#include <ctime>     // time_t, strftime()
#include <iomanip>   // put_time()
#include <regex>     // regex, regex_match(), sregex_iterator()
#include <sstream>   // ostringstream
#include <stdexcept> // runtime_error()
#include <string_view>

auto indented(const std::string &str, unsigned indentation) -> std::string {
  // Look for a newline except if it is in the end of the string, ie.
  // do not add indentation characters in the trailing of the string.
  // Use positive lookahead (?= ) to check the existence of a following
  // character after the newline but keeping it in the replaced string.
  static const std::regex reNewLine(R"(\n(?=.))");

  const std::string indentationChars(indentation, ' ');
  const std::string newLineIndented = "\n" + indentationChars;
  return str.empty() ? std::string{} :
                       indentationChars + std::regex_replace(str, reNewLine, newLineIndented);
}

auto chomp(const std::string &str, char choppedChar) -> std::string {
  if (str.empty()) {
    return {};
  }
  if (choppedChar == str.back()) {
    std::string res(str.begin(), --str.end());
    return res;
  }
  return str;
}

auto toHexString(unsigned char value) -> std::string {
  static const auto nibbleChar = [](unsigned val) {
    const auto nibble = val & 0xFU;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    return static_cast<char>((nibble < 10) ? '0' + nibble : 'a' + nibble - 10);
  };
  std::string result;
  result += nibbleChar(value >> 4U);
  result += nibbleChar(value);
  return result;
}

auto replaceSymbols(const std::string &str, const SymbolsMap &symbolsMap) -> std::string {
  // The variable name is a sequence of one or more characters excluding '{' and '}'
  // Surround regex with parens to capture its value.
#define RE_VARNAME "([^{}]+)"
  // A symbol is a varname surrounded by braces.
  // Use backslash to escape the spetial meaning of braces in regexes.
#define RE_SYMBOL "\\{" RE_VARNAME "\\}"
  const std::regex reSymbol(RE_SYMBOL);
  const auto beginSymbols = std::sregex_iterator(str.begin(), str.end(), reSymbol);
  const auto endSymbols = std::sregex_iterator();
  // Generate the result string by iterating over all the symbols and adding the previous
  // unprecessed string and the resolved symbol value.
  std::string result;
  auto itSuffix = str.cbegin();
  for (auto it = beginSymbols; it != endSymbols; ++it) {
    const auto &symbolName = (*it)[1].str();
    const auto itSymbol = symbolsMap.find(symbolName);
    if (itSymbol == symbolsMap.end()) {
      throw std::runtime_error("undefined symbol '" + symbolName + "'");
    }
    result += (*it).prefix().str();
    result += itSymbol->second;
    itSuffix = (*it).suffix().first;
    assert((*it).suffix().second == str.cend());
  }
  // Finally, append the final unprocessed string.
  assert(itSuffix <= str.cend());
  result += std::string_view(&*itSuffix, static_cast<std::size_t>(str.cend() - itSuffix));
  // Recursive call to resolve new generated symbols like:
  //  "Used batch size is {{Run}BatchSize}."
  //  -> "Used batch size is {IterationOneBatchSize}."
  //  -> "Used batch size is 8."
  if (result != str) {
    return replaceSymbols(result, symbolsMap);
  }
  return result;
}

auto parseStringMap(const std::string &s) -> std::optional<SymbolsMap> {
  // Expected format: '{' "key1": "value1", ... '}'
  // Every token is optionally separated by zero or more blank characters.
#define RE_STRVAL     R"("([^"]+)\")" // We want to capture the string values
#define RE_SPACE      "\\s*"
#define RE_KV_PAIR    RE_STRVAL RE_SPACE ":" RE_SPACE RE_STRVAL
#define RE_KV_SEP     RE_SPACE "," RE_SPACE
#define RE_KV_LIST    "(" RE_KV_PAIR RE_KV_SEP ")*" RE_KV_PAIR
#define RE_MAP_EXPR   "\\{" RE_SPACE "(" RE_KV_LIST ")?" RE_SPACE "\\}"
#define RE_PARSE_EXPR RE_SPACE "(" RE_MAP_EXPR ")?" RE_SPACE
  if (not std::regex_match(s, std::regex(RE_PARSE_EXPR))) {
    return std::nullopt;
  }
  SymbolsMap result;
  const auto reKV = std::regex(RE_KV_PAIR);
  const auto itEnd = std::sregex_iterator();
  for (auto it = std::sregex_iterator(s.begin(), s.end(), reKV); it != itEnd; ++it) {
    const auto &match = *it;
    const auto &[key, value] = std::make_pair(match[1].str(), match[2].str());
    result[key] = value;
  }
  return result;
}

auto timestamp() -> std::string {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  const auto milli = (now.time_since_epoch().count() / 1000000) % 1000;
  std::ostringstream result;
  result.width(3);
  result.fill('0');
  result << std::put_time(localtime(&t), "[%T.") << milli << "] ";
  return result.str();
}
