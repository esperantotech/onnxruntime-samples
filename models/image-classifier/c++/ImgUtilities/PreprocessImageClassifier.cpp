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

#include "PreprocessImageClassifier.hpp"
#include "FileSystem.hpp" // readFile()
#include "PNGUtils.hpp"   // decodePNGImage()

#include <stdexcept> // domain_error
#include <string>
#include <variant> // get<>(), get_if(<>)

auto ImageClassifierPreprocessor::loadImage(const std::filesystem::path &filename) -> RawImage {
  using namespace std::string_literals;
  if (not std::filesystem::exists(filename)) {
    throw BadImageFile(filename, "file does not exists");
  }
  const auto pngData = readFile(filename);
  auto res = decodePNGImage(pngData);
  if (const auto *errMsg = std::get_if<std::string>(&res)) {
    throw BadImageFile(filename, "error decoding PNG image data: " + *errMsg);
  }
  return std::get<RawImage>(res);
}

auto ImageClassifierPreprocessor::loadImagesAndPreprocess(const std::vector<std::string> &dataFiles,
    const PreprocessParams &params, const etsoc::GlowAPI::Type &resultTensorType)
    -> etsoc::GlowAPI::Tensor {
  const auto elemType = resultTensorType.getElementType();
  switch (elemType) {
  case etsoc::GlowAPI::ElemKind::FloatTy:
    return loadImagesAndPreprocess<float>(dataFiles, params, resultTensorType);
  case etsoc::GlowAPI::ElemKind::Float16Ty:
    return loadImagesAndPreprocess<glow::float16_t>(dataFiles, params, resultTensorType);
  case etsoc::GlowAPI::ElemKind::BFloat16Ty:
    return loadImagesAndPreprocess<glow::bfloat16_t>(dataFiles, params, resultTensorType);
  case etsoc::GlowAPI::ElemKind::Float64Ty:
    return loadImagesAndPreprocess<double>(dataFiles, params, resultTensorType);
  default:
    throw std::domain_error(
        "internal error: unsupported element type: " + std::to_string(static_cast<int>(elemType)));
  }
}