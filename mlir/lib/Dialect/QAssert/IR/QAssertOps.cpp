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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
// Custom Directives
//===----------------------------------------------------------------------===//

// Parses a real number. Decimal integers such as `0` are accepted as well,
// because `parseFloat` rejects them. Hexadecimal literals are rejected: the
// integer parser would read them as integers, but MLIR uses them for the bit
// pattern of a float.
static ParseResult parseReal(OpAsmParser& parser, double& value) {
  const bool negative = succeeded(parser.parseOptionalMinus());
  const auto loc = parser.getCurrentLocation();
  const char* text = loc.getPointer();
  if (text[0] == '0' && text[1] == 'x') {
    return parser.emitError(loc, "expected a decimal number");
  }
  int64_t integer = 0;
  const auto integerResult = parser.parseOptionalInteger(integer);
  if (integerResult.has_value()) {
    if (failed(*integerResult)) {
      return failure();
    }
    value = static_cast<double>(integer);
  } else if (parser.parseFloat(value)) {
    return failure();
  }
  if (negative) {
    value = -value;
  }
  return success();
}

// Parses an amplitude: a real number or a `(re, im)` pair.
static ParseResult parseAmplitude(OpAsmParser& parser,
                                  std::complex<double>& amplitude) {
  double real = 0.0;
  double imag = 0.0;
  if (succeeded(parser.parseOptionalLParen())) {
    if (parseReal(parser, real) || parser.parseComma() ||
        parseReal(parser, imag) || parser.parseRParen()) {
      return failure();
    }
  } else if (parseReal(parser, real)) {
    return failure();
  }
  amplitude = {real, imag};
  return success();
}

// A ket term creates a reference with 2^n amplitudes, so a short ket could
// otherwise request a huge tensor. 2^20 amplitudes take 16 MiB.
static constexpr size_t MAX_KET_QUBITS = 20;

// Parses the bitstring of a ket term after `|`, up to and including `>`. The
// lexer reads `01` as the integer 1, so the bitstring is taken from the source
// text to keep its leading zeros.
static ParseResult parseKetBits(OpAsmParser& parser, StringRef& bits) {
  const auto loc = parser.getCurrentLocation();
  APInt ignored;
  if (!parser.parseOptionalInteger(ignored).has_value()) {
    return parser.emitError(loc, "expected a bitstring of '0' and '1'");
  }
  const char* begin = loc.getPointer();
  const char* end = parser.getCurrentLocation().getPointer();
  bits = StringRef(begin, end - begin).rtrim();
  if (bits.find_first_not_of("01") != StringRef::npos) {
    return parser.emitError(loc,
                            "expected a bitstring of '0' and '1', but got '")
           << bits << "'";
  }
  if (bits.size() > MAX_KET_QUBITS) {
    return parser.emitError(loc, "ket bitstrings can have at most ")
           << MAX_KET_QUBITS << " qubits, but '" << bits << "' has "
           << bits.size();
  }
  return parser.parseGreater();
}

// Parses ket terms such as `0.707|00> - (0, 0.707)|11>`. A missing coefficient
// is 1, missing bitstrings get amplitude 0, and repeated bitstrings add up.
static ParseResult parseKet(OpAsmParser& parser,
                            SmallVector<std::complex<double>>& amplitudes) {
  SmallVector<std::pair<uint64_t, std::complex<double>>> terms;
  std::optional<size_t> numQubits;
  bool negative = succeeded(parser.parseOptionalMinus());
  while (true) {
    std::complex<double> coefficient = 1.0;
    if (failed(parser.parseOptionalVerticalBar()) &&
        (parseAmplitude(parser, coefficient) || parser.parseVerticalBar())) {
      return failure();
    }
    const auto loc = parser.getCurrentLocation();
    StringRef bits;
    if (parseKetBits(parser, bits)) {
      return failure();
    }
    if (!numQubits) {
      numQubits = bits.size();
    } else if (bits.size() != *numQubits) {
      return parser.emitError(loc, "all ket bitstrings must have ")
             << *numQubits << " qubits, but '" << bits << "' has "
             << bits.size();
    }
    uint64_t index = 0;
    bits.getAsInteger(2, index);
    terms.emplace_back(index, negative ? -coefficient : coefficient);

    if (succeeded(parser.parseOptionalPlus())) {
      negative = false;
    } else if (succeeded(parser.parseOptionalMinus())) {
      negative = true;
    } else {
      break;
    }
  }

  amplitudes.assign(size_t{1} << *numQubits, 0.0);
  for (const auto& [index, coefficient] : terms) {
    amplitudes[index] += coefficient;
  }
  return success();
}

// Parses `== [a0, a1, ...]` or `== c0|b0> + c1|b1> ...` into a rank-one
// `complex<f64>` tensor.
static ParseResult parseReference(OpAsmParser& parser,
                                  ElementsAttr& reference) {
  if (parser.parseEqual() || parser.parseEqual()) {
    return failure();
  }
  SmallVector<std::complex<double>> amplitudes;
  if (succeeded(parser.parseOptionalLSquare())) {
    const auto parseElement = [&]() -> ParseResult {
      return parseAmplitude(parser, amplitudes.emplace_back());
    };
    if (failed(parser.parseOptionalRSquare()) &&
        (parser.parseCommaSeparatedList(parseElement) ||
         parser.parseRSquare())) {
      return failure();
    }
  } else if (parseKet(parser, amplitudes)) {
    return failure();
  }

  auto* context = parser.getContext();
  auto type =
      RankedTensorType::get({static_cast<int64_t>(amplitudes.size())},
                            ComplexType::get(Float64Type::get(context)));
  reference = DenseElementsAttr::get(type, ArrayRef(amplitudes));
  return success();
}

// Prints a real amplitude as a plain number and a complex one as `(re, im)`.
static void printReference(OpAsmPrinter& printer, Operation* /*op*/,
                           ElementsAttr reference) {
  printer << "== ";
  auto dense = dyn_cast<DenseElementsAttr>(reference);
  auto complexType = dyn_cast<ComplexType>(reference.getElementType());
  if (!dense || !complexType || !complexType.getElementType().isF64()) {
    // Only reachable for invalid IR, which the verifier rejects.
    printer.printAttribute(reference);
    return;
  }
  printer << '[';
  llvm::interleaveComma(dense.getValues<std::complex<double>>(), printer,
                        [&](std::complex<double> amplitude) {
                          // Keep the sign of -0.0, so printing round-trips.
                          if (amplitude.imag() == 0.0 &&
                              !std::signbit(amplitude.imag())) {
                            printer.printFloat(APFloat(amplitude.real()));
                            return;
                          }
                          printer << '(';
                          printer.printFloat(APFloat(amplitude.real()));
                          printer << ", ";
                          printer.printFloat(APFloat(amplitude.imag()));
                          printer << ')';
                        });
  printer << ']';
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
