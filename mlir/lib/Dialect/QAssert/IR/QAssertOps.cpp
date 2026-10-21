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

#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <memory>

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

static LogicalResult verifyMinQubits(Operation* op, ValueRange qubits,
                                     size_t minQubits) {
  if (qubits.size() < minQubits) {
    return op->emitOpError("number of qubits must be at least ")
           << minQubits << ", but got " << qubits.size();
  }
  return success();
}

LogicalResult SupOp::verify() {
  return verifyMinQubits(*this, getQubitsIn(), 1);
}

#define GET_OP_CLASSES
#include "mqt/Dialect/QAssert/IR/QAssertOps.cpp.inc"
