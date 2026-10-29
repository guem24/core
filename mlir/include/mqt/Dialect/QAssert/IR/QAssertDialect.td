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
        The QAssert dialect provides assertions for quantum programs. The name
        "QAssert" stands for "Quantum Assertions."

        An assertion states what the quantum state should look like at a
        certain point in the program. If it fails, the program does not do
        what you expected at that point. There are four assertions:
        superposition (`sup`), entanglement (`ent`), equality to a reference
        state (`eq`), and which measurement results are possible (`span`).

        Assertions act on QCO qubits. An assertion does not change the state.
        It returns its qubits in the same order, so it stays at its point in
        the program. Assertions are not gates, so they cannot be used inside
        `qco.ctrl`, `qco.inv`, or `qco.pow` bodies.
    }];
  let cppNamespace = "::mlir::qassert";
  let dependentDialects = ["::mlir::qco::QCODialect"];
}

#endif // MLIR_DIALECT_QASSERT_IR_QASSERTDIALECT_TD
