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

#ifndef _ERRORMNG_HPP_
#define _ERRORMNG_HPP_

#include <string>
#include <string_view>
#include <type_traits> // is_copy_constructible_v<>()
#include <utility>     // move()


// Outputs the provided error message and
// exits the program execution with an error code.
[[noreturn]] void exitWithErrorMsg(std::string_view msg);
[[noreturn]] void exitWithErrorMsg(const std::string& msg);
[[noreturn]] void exitWithErrorMsg(const char* msg);

// Outputs an error message related to the error value if not null and
// exits the program execution with an error code.
//[[noreturn]] void exitWithErrorValue(const ErrorValue* err);

#endif // _ERRORMNG_HPP_