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

class QAssertQubitsOp<string mnemonic, list<Trait> traits = []>
    : QAssertOp<mnemonic,
                !listconcat([MemoryEffects<[MemWrite]>,
                             RangedTypesMatchWith<
                                 "input and output qubit types match",
                                 "qubits_in", "qubits_out", "$_self">],
                            traits)> {
  let arguments =
      (ins Arg<Variadic<QubitType>, "the target qubits">:$qubits_in);
  let results = (outs Variadic<QubitType>:$qubits_out);
  let assemblyFormat = "$qubits_in attr-dict `:` type($qubits_in)";
  let hasVerifier = 1;
}

def SupOp : QAssertQubitsOp<"sup"> {
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
}

def EntOp : QAssertQubitsOp<"ent"> {
  let summary = "Assert that every pair of the given qubits is correlated";
  let description = [{
    Asserts that every pair of `qubits_in` is correlated: the two qubits
    are not independent, so a measurement on one of them can tell you
    something about the other. A Bell pair or a GHZ state passes. The state
    `|+>|+>` fails: each qubit is in superposition, but they are
    independent.

    Formally, for every pair (A, B), the quantum mutual information
    `S(rho_A) + S(rho_B) - S(rho_AB)` is greater than 0. `S` is the von
    Neumann entropy and `rho_X` is the state of `X` alone. This matches
    `assert-ent` in the MQT Debugger.

    Correlation is weaker than entanglement when the qubits are also
    entangled with other qubits. For example, after a GHZ state on three
    qubits, asserting two of them passes, although these two alone are
    not entangled.

    The operation does not change the state. It returns its qubits in the
    same order, which pins the assertion to this point in the program.

    Example:
    ```mlir
    %a1 = qco.h %a0 : !qco.qubit
    %a2, %b1 = qco.ctrl(%a1) targets (%t0 = %b0) {
      %t1 = qco.x %t0 : !qco.qubit
      qco.yield %t1 : !qco.qubit
    } : ({!qco.qubit}, {!qco.qubit})
    %a3, %b2 = qassert.ent %a2, %b1 : !qco.qubit, !qco.qubit
    ```
    }];
}

#endif // MLIR_DIALECT_QASSERT_IR_QASSERTOPS_TD
