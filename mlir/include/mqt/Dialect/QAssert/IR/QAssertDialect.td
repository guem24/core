// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QASSERT_IR_QASSERTDIALECT_TD
#define MLIR_DIALECT_QASSERT_IR_QASSERTDIALECT_TD

include "mlir/IR/DialectBase.td"

def QAssertDialect : Dialect {
  let name = "qassert";
  let summary = "Quantum state assertions";
  let description = [{
        The QAssert dialect provides operations for asserting properties of
        quantum states at specific points in a quantum program: superposition
        (`sup`), pairwise correlation (`ent`), equality to a reference state
        (`eq`), and a basis-state span (`span`).
    }];
  let cppNamespace = "::mlir::qassert";
  let dependentDialects = ["::mlir::qco::QCODialect"];
}

#endif // MLIR_DIALECT_QASSERT_IR_QASSERTDIALECT_TD
