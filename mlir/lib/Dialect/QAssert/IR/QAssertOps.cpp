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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

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

LogicalResult EntOp::verify() {
  return verifyMinQubits(*this, getQubitsIn(), 2);
}

// The similarity scales the reference to length 1, so its norm does not change
// the result. The check catches typos, which scaling would turn into a
// different valid state. The tolerance accepts rounded amplitudes such as
// 0.707.
static constexpr double REFERENCE_NORM_TOLERANCE = 1e-2;

LogicalResult EqOp::verify() {
  if (failed(verifyMinQubits(*this, getQubitsIn(), 1))) {
    return failure();
  }

  auto reference = dyn_cast<DenseElementsAttr>(getReference());
  if (!reference) {
    return emitOpError("reference must use dense element storage");
  }
  auto type = dyn_cast<RankedTensorType>(reference.getType());
  if (!type || type.getRank() != 1) {
    return emitOpError("reference must be a rank-one tensor");
  }
  auto complexType = dyn_cast<ComplexType>(type.getElementType());
  if (!complexType || !complexType.getElementType().isF64()) {
    return emitOpError("reference elements must have type complex<f64>");
  }

  // The reference cannot hold 2^63 or more amplitudes, so larger qubit counts
  // fail the length check without computing the shift.
  const auto numQubits = getQubitsIn().size();
  const auto numAmplitudes = type.getNumElements();
  if (numQubits >= 63 ||
      std::cmp_not_equal(numAmplitudes, uint64_t{1} << numQubits)) {
    return emitOpError("reference must have 2^n amplitudes for n = ")
           << numQubits << " qubits, but got " << numAmplitudes;
  }

  double sumOfSquares = 0.0;
  for (const auto amplitude : reference.getValues<std::complex<double>>()) {
    if (!std::isfinite(amplitude.real()) || !std::isfinite(amplitude.imag())) {
      return emitOpError("reference amplitudes must be finite");
    }
    sumOfSquares += std::norm(amplitude);
  }
  if (std::abs(sumOfSquares - 1.0) > REFERENCE_NORM_TOLERANCE) {
    return emitOpError("squared reference amplitudes must sum to 1 within ")
           << REFERENCE_NORM_TOLERANCE << ", but sum to " << sumOfSquares;
  }

  const auto threshold = getSimilarityThreshold().convertToDouble();
  if (!(threshold >= 0.0 && threshold <= 1.0)) {
    return emitOpError("similarity threshold must be in [0, 1], but got ")
           << threshold;
  }
  return success();
}

LogicalResult SpanOp::verify() {
  if (failed(verifyMinQubits(*this, getQubitsIn(), 1))) {
    return failure();
  }

  auto states = getStates();
  if (states.empty()) {
    return emitOpError("states must contain at least one bitstring");
  }

  const auto numQubits = getQubitsIn().size();
  llvm::SmallDenseSet<StringRef, 8> seen;
  for (auto state : states.getAsValueRange<StringAttr>()) {
    if (state.size() != numQubits ||
        state.find_first_not_of("01") != StringRef::npos) {
      return emitOpError("state '")
             << state << "' must have one '0' or '1' for each of the "
             << numQubits << " qubits";
    }
    if (!seen.insert(state).second) {
      return emitOpError("state '") << state << "' appears more than once";
    }
  }

  return success();
}

#define GET_OP_CLASSES
#include "mqt/Dialect/QAssert/IR/QAssertOps.cpp.inc"
