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

#include "RawImage.hpp"
#include "Types.hpp"

#include "glow/GlowAPI/etsoc.h"

#include <cassert>
#include <cstddef>   // size_t
#include <iterator>  // begin(), end(), next(), empty()
#include <stdexcept> // domain_error, runtime_error
#include <string>
#include <utility> // pair
#include <vector>

class ImageClassifierPreprocessor {
public:
  // Exception thrown when an error happens processing an image file.
  class BadImageFile : public std::runtime_error {
    std::string what_;

  public:
    explicit BadImageFile(const std::string &filename, const std::string &what)
    : std::runtime_error(what)
    , what_("file '" + filename + "': " + std::runtime_error::what()) {
    }
    const char *what() const noexcept override {
      return what_.c_str();
    }
  };

  enum class DatasetClass {
    Imagenet,
  };
  struct PreprocessParams {
    DatasetClass datasetClass = DatasetClass::Imagenet;
    NormalizationMode normalizationMode = NormalizationMode::None;
    PixelChannelOrder channelOrder = PixelChannelOrder::RGB;
    Layout layout = Layout::BatchHeightWidthColor;
  };

  static auto loadImagesAndPreprocess(const std::vector<std::string> &dataFiles,
      const PreprocessParams &params, const etsoc::GlowAPI::Type &resultTensorType)
      -> etsoc::GlowAPI::Tensor;

  static auto loadImage(const std::filesystem::path &filename) -> RawImage;
  
private:  

  template<typename ElemTy>
  static constexpr auto normalizationModeToRange(NormalizationMode mode)
      -> std::pair<ElemTy, ElemTy> {
    switch (mode) {
    case NormalizationMode::FromMinusOneToOne:
      return {-1.0, 1.0};
    case NormalizationMode::FromZeroToOne:
      return {0.0, 1.0};
    case NormalizationMode::FromZeroToMaxUInt8:
      return {0.0, 255.0};
    case NormalizationMode::FromMinusInt8ToMaxInt8:
      return {-128.0, 127.0};
    default:
      throw std::domain_error("unsupported normalization: " + toString(mode));
    }
  }

  template<typename ElemTy>
  static auto getTensorFromPNGImage(etsoc::GlowAPI::ElemKind elemType, const RawImage &img,
      std::pair<ElemTy, ElemTy> range, const std::vector<float> &mean,
      const std::vector<float> &stddev) -> etsoc::GlowAPI::Tensor {
    const auto bytesPerPixel = img.bytesPerPixel();
    const ElemTy scale = (range.second - range.first) / ElemTy(255.0);
    const ElemTy bias = range.first;
    const auto numChannels = img.bytesPerPixel();
    etsoc::GlowAPI::Tensor result(elemType, {img.height(), img.width(), numChannels});
    auto resultTensorHandle = result.getHandle<ElemTy>();
    assert(numChannels <= mean.size());
    assert(numChannels <= stddev.size());
    const auto *pRowImgData = img.data();
    for (glow::dim_t row_n = 0; row_n < img.height(); ++row_n, pRowImgData += img.rowBytes()) {
      const auto *pPixel = pRowImgData;
      for (glow::dim_t col_n = 0; col_n < img.width(); ++col_n, pPixel += bytesPerPixel) {
        for (glow::dim_t i = 0; i < numChannels; ++i) {
          const ElemTy val = (float(pPixel[i]) - mean[i]) / stddev[i];
          resultTensorHandle.at({row_n, col_n, i}) = val * scale + bias;
        }
      }
    }
    return result;
  }

  template<typename ElemTy>
  static auto getNormalizedTensorFromPNGImage(etsoc::GlowAPI::ElemKind elemType,
      const RawImage &pngImg, const PreprocessParams &params) -> etsoc::GlowAPI::Tensor {
    assert(params.datasetClass == DatasetClass::Imagenet);
    // Mean and stddev normalization values of input tensors.
    // These are standard normalization factors for imagenet, adjusted for
    // normalizing values in the 0to255 range instead of 0to1, as seen at:
    // https://github.com/pytorch/examples/blob/master/imagenet/main.py
    const std::vector<float> mean{0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    const auto range = normalizationModeToRange<ElemTy>(params.normalizationMode);
    etsoc::GlowAPI::Tensor imgTensor =
        getTensorFromPNGImage<ElemTy>(elemType, pngImg, range, mean, stddev);

    // PNG images are NHWC and RGB. Convert if needed.
    const auto &imgTensorDims = imgTensor.dims();
    const auto imgHeight = imgTensorDims[0];
    const auto imgWidth = imgTensorDims[1];
    const auto numChannels = imgTensorDims[2];

    // Convert to requested channel ordering.
    if (params.channelOrder == PixelChannelOrder::BGR) {
      auto imgTensorHandle = imgTensor.getHandle<ElemTy>();
      etsoc::GlowAPI::Tensor swizzled(imgTensor.getType());
      auto swizzledHandle = swizzled.getHandle<ElemTy>();
      for (std::size_t y = 0; y < imgHeight; ++y) {
        for (std::size_t x = 0; x < imgWidth; ++x) {
          for (std::size_t z = 0; z < numChannels; ++z) {
            swizzledHandle.at({y, x, numChannels - 1 - z}) = imgTensorHandle.at({y, x, z});
          }
        }
      }
      imgTensor = std::move(swizzled);
    }

    // Convert to requested layout.
    if (params.layout == Layout::BatchColorHeightWidth) {
      etsoc::GlowAPI::Tensor transposed;
      imgTensor.transpose(&transposed, {2u, 0u, 1u});
      imgTensor = std::move(transposed);
    }

    return imgTensor;
  }

  template<typename ElemTy>
  static auto loadImagesAndPreprocess(const std::vector<std::string> &dataFiles,
      const PreprocessParams &params, const etsoc::GlowAPI::Type &resultTensorType)
      -> etsoc::GlowAPI::Tensor {
    // Utility functions to dump tensor dimensions
    const auto dimsToString = [](const auto &dimensions) {
      auto it = std::begin(dimensions);
      return std::empty(dimensions) ?
          std::string() :
          std::accumulate(std::next(it), std::end(dimensions), std::to_string(*it),
              [](const auto &acc, auto v) { return acc + "x" + std::to_string(v); });
    };
    const auto dimsItToString = [](auto itBeg, auto itEnd) {
      return (itBeg == itEnd) ?
          std::string() :
          std::accumulate(std::next(itBeg), itEnd, std::to_string(*itBeg),
              [](const auto &acc, auto v) { return acc + "x" + std::to_string(v); });
    };

    etsoc::GlowAPI::Tensor resultTensor(resultTensorType);
    const auto &resultTensorDims = resultTensor.dims();
    assert(resultTensorDims.size() == 4);
    assert(resultTensorDims[0] == dataFiles.size()); // first dim is equal to batch size
    const auto &expectedDimsIt =
        std::next(resultTensorDims.begin()); // rest of dims should be equal to image dims
    auto resultTensorHandle = resultTensor.getHandle<ElemTy>();
    std::size_t n = 0;
    for (const auto &filename : dataFiles) {
      std::cout <<"processing image file: " << filename << std::endl;
      const auto img = loadImage(filename);
      auto imgTensor =
          getNormalizedTensorFromPNGImage<ElemTy>(resultTensorType.getElementType(), img, params);
      if (const auto &dims = imgTensor.dims();
          not std::equal(dims.begin(), dims.end(), expectedDimsIt)) {
        throw BadImageFile(filename,
            "invalid image dimensions " + dimsToString(dims) + ", expected "
                + dimsItToString(expectedDimsIt, resultTensorDims.end()));
      }
      resultTensorHandle.insertSlice(imgTensor, n++);
    }
    return resultTensor;
  }
};

