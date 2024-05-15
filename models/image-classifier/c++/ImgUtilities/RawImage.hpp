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

#ifndef _RAWIMAGE_HPP_
#define _RAWIMAGE_HPP_

#include <cassert>
#include <cstddef> // byte, size_t
#include <vector>

// Class that implements raw image data.
class RawImage final {
  std::size_t width_;
  std::size_t height_;
  std::size_t bytesPerPixel_;
  std::size_t rowBytes_;
  std::vector<std::byte> data_;

public:
  explicit RawImage(
      std::size_t width, std::size_t height, std::size_t bytesPerPixel, std::size_t rowBytes)
  : width_(width)
  , height_(height)
  , bytesPerPixel_(bytesPerPixel)
  , rowBytes_(rowBytes)
  , data_(height * rowBytes) {
    assert(width_ > 0);
    assert(height_ > 0);
    assert(bytesPerPixel_ > 0);
    assert(rowBytes_ > 0);
    assert(width_ * bytesPerPixel_ <= rowBytes_);
  }
  explicit RawImage(std::size_t width, std::size_t height, std::size_t bytesPerPixel)
  : RawImage(width, height, bytesPerPixel, width * bytesPerPixel) {
  }

  std::size_t width() const noexcept {
    return width_;
  }

  std::size_t height() const noexcept {
    return height_;
  }

  std::size_t bytesPerPixel() const noexcept {
    return bytesPerPixel_;
  }

  std::size_t rowBytes() const noexcept {
    return rowBytes_;
  }

  const std::byte *data() const noexcept {
    return data_.data();
  }
  std::byte *data() noexcept {
    return data_.data();
  }
};

#endif //_RAWIMAGE_HPP_