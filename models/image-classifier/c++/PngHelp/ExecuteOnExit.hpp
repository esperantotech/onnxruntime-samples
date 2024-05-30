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

#include <functional>

// Helper class used to execute some code at the end of the object scope.
class ExecuteOnExit final {
  using VoidFuncPtr = std::function<void()>;
  VoidFuncPtr cleanupFn_ = nullptr;

public:
  ExecuteOnExit(const ExecuteOnExit&) = delete;
  ExecuteOnExit(ExecuteOnExit&&) = delete;
  auto operator=(const ExecuteOnExit&) -> ExecuteOnExit& = delete;
  auto operator=(ExecuteOnExit &&) -> ExecuteOnExit& = delete;

  explicit ExecuteOnExit(VoidFuncPtr func = nullptr) noexcept
    : cleanupFn_(func) {
  }
  ~ExecuteOnExit() noexcept try {
    if (cleanupFn_) {
      cleanupFn_();
    }
  } catch (...) {
    // TBD
  }
  // Replace the cleanup function and returns the old one.
  auto reset(VoidFuncPtr func = nullptr) noexcept -> VoidFuncPtr {
    std::swap(cleanupFn_, func);
    return func;
  }
};
