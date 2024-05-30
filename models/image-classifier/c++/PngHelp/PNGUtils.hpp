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

#ifndef _PNGUTILS_HPP_
#define _PNGUTILS_HPP_

#include "RawImage.hpp"

#include <stdexcept> // runtime_error
#include <string>
#include <variant>
#include <vector>

struct libpng_error final : std::runtime_error {
  using std::runtime_error::runtime_error;
};

auto isPNGImage(const std::vector<char>& data) -> bool;

auto decodePNGImage(const std::vector<char>& data) -> std::variant<std::string, RawImage>;

#endif //_PNGUTILS_HPP_