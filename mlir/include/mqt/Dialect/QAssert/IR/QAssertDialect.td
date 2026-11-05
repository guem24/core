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
        The QAssert dialect provides assertions for quantum programs. An
        assertion checks the expected quantum state at a point in the program
        and fails if the actual state differs. There are four assertions:
        superposition (`sup`), entanglement (`ent`), equality to a reference
        state (`eq`), and lying in a span of basis states (`span`).

        Assertions act on QCO qubits. They do not change the state and return
        their qubits in the same order. Since they are not gates, they cannot
        be used inside `qco.ctrl`, `qco.inv`, or `qco.pow` bodies.
    }];
  let cppNamespace = "::mlir::qassert";
  let dependentDialects = ["::mlir::qco::QCODialect"];
}

#endif // MLIR_DIALECT_QASSERT_IR_QASSERTDIALECT_TD
