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

#include "Types.hpp"
#include <stdexcept> // domain_error

auto parseNormalizationMode(std::string_view v) -> std::optional<NormalizationMode> {
  if (v == "none") {
    return NormalizationMode::None;
  }
  if (v == "0..1") {
    return NormalizationMode::FromZeroToOne;
  }
  if (v == "0..255") {
    return NormalizationMode::FromZeroToMaxUInt8;
  }
  if (v == "0..65535") {
    return NormalizationMode::FromZeroToMaxUInt16;
  }
  if (v == "-1..1") {
    return NormalizationMode::FromMinusOneToOne;
  }
  if (v == "-128..127") {
    return NormalizationMode::FromMinusInt8ToMaxInt8;
  }
  if (v == "-32768..32767") {
    return NormalizationMode::FromMinusInt16ToMaxInt16;
  }
  return std::nullopt;
}

auto toString(NormalizationMode value) -> std::string {
  switch (value) {
  case NormalizationMode::None:
    return "none";
  case NormalizationMode::FromZeroToOne:
    return "0..1";
  case NormalizationMode::FromZeroToMaxUInt8:
    return "0..255";
  case NormalizationMode::FromZeroToMaxUInt16:
    return "0..65535";
  case NormalizationMode::FromMinusOneToOne:
    return "-1..1";
  case NormalizationMode::FromMinusInt8ToMaxInt8:
    return "-128..127";
  case NormalizationMode::FromMinusInt16ToMaxInt16:
    return "-32768..32767";
  default:
    throw std::domain_error(
        "unknown NormalizationMode value: " + std::to_string(static_cast<int>(value)));
  }
}

auto parseLayout(std::string_view v) -> std::optional<Layout> {
  if (v == "NCHW") {
    return Layout::BatchColorHeightWidth;
  }
  if (v == "NHWC") {
    return Layout::BatchHeightWidthColor;
  }
  return std::nullopt;
}

auto toString(Layout value) -> std::string {
  switch (value) {
  case Layout::BatchColorHeightWidth:
    return "NCHW";
  case Layout::BatchHeightWidthColor:
    return "NHWC";
  default:
    throw std::domain_error("unknown Layout value: " + std::to_string(static_cast<int>(value)));
  }
}

auto parsePixelChannelOrder(std::string_view v) -> std::optional<PixelChannelOrder> {
  if (v == "RGB") {
    return PixelChannelOrder::RGB;
  }
  if (v == "BGR") {
    return PixelChannelOrder::BGR;
  }
  return std::nullopt;
}

auto toString(PixelChannelOrder value) -> std::string {
  switch (value) {
  case PixelChannelOrder::RGB:
    return "RGB";
  case PixelChannelOrder::BGR:
    return "BGR";
  default:
    throw std::domain_error(
        "unknown PixelChannelOrder value: " + std::to_string(static_cast<int>(value)));
  }
}

auto parseModelCategory(std::string_view v) -> std::optional<ModelCategory> {
  if (v == "ImageClassifier") {
    return ModelCategory::ImageClassifier;
  }
  if (v == "DLRM") {
    return ModelCategory::DLRM;
  }
  if (v == "BERT") {
    return ModelCategory::BERT;
  }
  if (v == "Generic") {
    return ModelCategory::GENERIC;
  }
  return std::nullopt;
}

auto toString(ModelCategory value) -> std::string {
  switch (value) {
  case ModelCategory::ImageClassifier:
    return "ImageClassifier";
  case ModelCategory::DLRM:
    return "DLRM";
  case ModelCategory::BERT:
    return "BERT";
  case ModelCategory::GENERIC:
    return "Generic";
  default:
    throw std::domain_error(
        "unknown ModelCategory value: " + std::to_string(static_cast<int>(value)));
  }
}
