/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QAssert/IR/QAssertOps.h"

#include "mqt/Dialect/QAssert/IR/QAssertDialect.h"

#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::qassert;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mqt/Dialect/QAssert/IR/QAssertOpsDialect.cpp.inc"

void QAssertDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mqt/Dialect/QAssert/IR/QAssertOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

LogicalResult SupOp::verify() {
  if (getQubitsIn().empty()) {
    return emitOpError("at least one qubit must be specified");
  }
  return success();
}

#define GET_OP_CLASSES
#include "mqt/Dialect/QAssert/IR/QAssertOps.cpp.inc"
