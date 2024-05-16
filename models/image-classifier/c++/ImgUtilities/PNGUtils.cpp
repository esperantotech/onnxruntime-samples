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

#include "PNGUtils.hpp"
#include "ExecuteOnExit.hpp"
#include "RawImage.hpp"

// libpng
#include "png.h"

#include <algorithm> // min(), copy_n()
#include <cassert>
#include <climits>  // CHAR_BIT
#include <cstddef>  // size_t
#include <cstdio>   // FILE
#include <fstream>  // ifstream
#include <iterator> // advance()
#include <string>
#include <variant>

namespace {

// Number of bytes of the PNG image files signature.
enum { PNG_SIGNATURE_BYTES = 8 };

// Class used to read a PNG image located in memory.
class MemFile final {
  const png_byte *current_;
  const png_byte *data_;
  std::size_t size_;

public:
  explicit MemFile(const png_byte *pData, std::size_t size)
  : current_(pData)
  , data_(pData)
  , size_(size) {
  }

  static void read(png_structp png, png_byte *buf, png_size_t len) {
    auto *const mf = static_cast<MemFile *>(png_get_io_ptr(png));
    assert(mf != nullptr);
    assert(mf->data_ <= mf->current_);
    const auto bytesConsumed = size_t(mf->current_ - mf->data_);
    assert(bytesConsumed <= mf->size_);
    const auto remainingBytes = size_t(mf->size_ - bytesConsumed);
    if (remainingBytes < len) {
      throw libpng_error("libpng internal error: read requested length (" + std::to_string(len)
          + " bytes) is greater than available image data (" + std::to_string(remainingBytes)
          + " of " + std::to_string(mf->size_) + " bytes)");
    }
    const auto n = std::min(len, remainingBytes);
    std::copy_n(mf->current_, n, buf);
    std::advance(mf->current_, n);
  }
};

[[noreturn]] void pngErrorHandler([[maybe_unused]] png_structp png, png_const_charp errorMsg) {
  // The error handling routine must NOT return to the calling routine.
  throw libpng_error(errorMsg);
}

void pngWarningHandler([[maybe_unused]] png_structp png, [[maybe_unused]] png_const_charp msg) {
}

auto getRowPointers(RawImage &img) -> std::vector<std::byte *> {
  std::vector<std::byte *> result;
  result.reserve(img.height());
  std::generate_n(std::back_inserter(result), img.height(),
      [ptr = img.data(), rowBytes = img.rowBytes()]() mutable {
        auto *res = ptr;
        std::advance(ptr, rowBytes);
        return res;
      });
  return result;
}

} // unnamed namespace

auto isPNGImage(const std::vector<char> &data) -> bool {
  /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */
  auto pData = reinterpret_cast<png_const_bytep>(data.data());
  return (PNG_SIGNATURE_BYTES <= data.size()) and (png_sig_cmp(pData, 0, PNG_SIGNATURE_BYTES) == 0);
}

auto decodePNGImage(const std::vector<char> &data) -> std::variant<std::string, RawImage> {
  // Validate signature.
  if (not isPNGImage(data)) {
    return std::string("invalid image file signature");
  }

  // Initialize
  png_structp png =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, pngErrorHandler, pngWarningHandler);
  if (png == nullptr) {
    return std::string("cannot initialize the PNG decoding process");
  }
  ExecuteOnExit freePNGResources([&png] { png_destroy_read_struct(&png, nullptr, nullptr); });
  png_infop pngInfo = png_create_info_struct(png);
  if (pngInfo == nullptr) {
    return std::string("cannot initialize the PNG decoding process");
  }
  freePNGResources.reset([&png, &pngInfo] { png_destroy_read_struct(&png, &pngInfo, nullptr); });

  /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */
  MemFile mf(reinterpret_cast<const png_byte *>(data.data()), data.size());
  png_set_read_fn(png, nullptr, static_cast<png_rw_ptr>(&MemFile::read));

  /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */
  png_init_io(png, reinterpret_cast<FILE *>(&mf));
  png_set_sig_bytes(png, 0);
  png_read_info(png, pngInfo);

  const std::size_t width = png_get_image_width(png, pngInfo);
  const std::size_t height = png_get_image_height(png, pngInfo);
  const png_byte colorType = png_get_color_type(png, pngInfo);
  const png_byte bitDepth = png_get_bit_depth(png, pngInfo);
  if (bitDepth != CHAR_BIT) {
    return std::string("invalid image: bit depth not supported.");
  }
  const bool isGray = colorType == PNG_COLOR_TYPE_GRAY;
  const bool isRGB = (colorType == PNG_COLOR_TYPE_RGB);
  if (not(isGray or isRGB)) {
    return std::string("invalid image: color type not supported.");
  }
  const std::size_t bytesPerPixel = isGray ? 1 : 3;

  png_read_update_info(png, pngInfo);
  const auto rowBytes = png_get_rowbytes(png, pngInfo);
  if (width * bytesPerPixel != rowBytes) {
    return std::string("invalid image: row bytes does not match width and color type");
  }
  RawImage result(width, height, bytesPerPixel, rowBytes);
  auto rowPointers = getRowPointers(result);
  /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */
  png_read_image(png, reinterpret_cast<png_bytepp>(rowPointers.data()));
  png_read_end(png, pngInfo);

  return result;
}