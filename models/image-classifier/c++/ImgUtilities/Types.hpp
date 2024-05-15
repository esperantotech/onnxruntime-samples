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

#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <optional>
#include <string>
#include <string_view>

enum class NormalizationMode {
  None,
  FromZeroToOne,
  FromZeroToMaxUInt8,
  FromZeroToMaxUInt16,
  FromMinusOneToOne,
  FromMinusInt8ToMaxInt8,
  FromMinusInt16ToMaxInt16,
};
auto parseNormalizationMode(std::string_view v) -> std::optional<NormalizationMode>;
auto toString(NormalizationMode value) -> std::string;

enum class Layout {
  BatchHeightWidthColor,
  BatchColorHeightWidth,
};
auto parseLayout(std::string_view v) -> std::optional<Layout>;
auto toString(Layout value) -> std::string;

enum class PixelChannelOrder {
  RGB,
  BGR,
};
auto parsePixelChannelOrder(std::string_view v) -> std::optional<PixelChannelOrder>;
auto toString(PixelChannelOrder value) -> std::string;

enum class ModelCategory {
  ImageClassifier,
  DLRM,
  BERT,
  GENERIC,
};
auto parseModelCategory(std::string_view v) -> std::optional<ModelCategory>;
auto toString(ModelCategory value) -> std::string;

#endif // _TYPES_HPP_