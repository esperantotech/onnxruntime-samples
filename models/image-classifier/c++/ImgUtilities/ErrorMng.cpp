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

#include "ErrorMng.hpp"

#include <cstdlib>  // EXIT_FAILURE, exit()
#include <iostream> // cerr

// Outputs the provided error message and
// exits the program execution with an error code.
void exitWithErrorMsg(std::string_view msg) {
  std::cerr << "\nerror: " << msg << std::endl;
  exit(EXIT_FAILURE);
}
void exitWithErrorMsg(const std::string& msg) {
  exitWithErrorMsg(std::string_view(msg));
}
void exitWithErrorMsg(const char* msg) {
  exitWithErrorMsg((msg != nullptr) ? std::string_view(msg) : "unknown");
}
