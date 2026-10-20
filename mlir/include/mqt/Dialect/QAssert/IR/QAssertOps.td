// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QASSERT_IR_QASSERTOPS_TD
#define MLIR_DIALECT_QASSERT_IR_QASSERTOPS_TD

include "mqt/Dialect/QCO/IR/QCOTypes.td"
include "mqt/Dialect/QAssert/IR/QAssertDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QAssertOp<string mnemonic, list<Trait> traits = []>
    : Op<QAssertDialect, mnemonic, traits>;

def SupOp
    : QAssertOp<"sup", traits = [MemoryEffects<[MemWrite]>,
                                 RangedTypesMatchWith<
                                     "input and output qubit types match",
                                     "qubits_in", "qubits_out", "$_self">]> {
  let summary = "Assert that the given qubits are in superposition";
  let description = [{
    Asserts that measuring all `qubits_in` together in the computational
    basis can give more than one outcome. For several qubits, this holds
    if and only if at least one qubit on its own has no fixed outcome.
    To assert that each qubit is in superposition individually, use one
    `qassert.sup` per qubit.

    The operation does not change the state. It returns its qubits in the
    same order, which pins the assertion to this point in the program.

    Example:
    ```mlir
    %q1 = qco.h %q0 : !qco.qubit
    %q2 = qassert.sup %q1 : !qco.qubit
    ```
    }];

  let arguments =
      (ins Arg<Variadic<QubitType>, "the target qubits">:$qubits_in);
  let results = (outs Variadic<QubitType>:$qubits_out);
  let assemblyFormat = "$qubits_in attr-dict `:` type($qubits_in)";

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_QASSERT_IR_QASSERTOPS_TD
