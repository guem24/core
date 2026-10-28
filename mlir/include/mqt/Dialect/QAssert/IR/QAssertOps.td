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
                !listconcat([Pure, RangedTypesMatchWith<
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

def EqOp : QAssertQubitsOp<"eq"> {
  let summary = "Assert that the given qubits are in a reference state";
  let description = [{
    Asserts that `qubits_in`, taken together, are in the state given by
    `reference`. The assertion holds if `abs(<current|reference>)` is at
    least `similarity_threshold`, which lies in `[0, 1]` and defaults to 1.
    The global phase does not matter. This matches the state-vector form of
    `assert-eq` in the MQT Debugger.

    `reference` is a rank-one tensor of `2^n` `complex<f64>` amplitudes,
    where `n` is the number of qubits. Operand 0 labels the most-significant
    basis bit, as in `qco.unitary`. For example, `a = |1>` and `b = |0>`
    give `[0, 0, 1, 0]` for the operands `a, b`.

    The squared amplitudes must sum to 1 within `1e-2`, so hand-rounded
    amplitudes such as `0.707` are accepted. A rounded reference does not
    reach similarity 1, so it needs a threshold below 1, such as `0.999`.

    If the qubits are entangled with qubits outside the list, they have no
    state vector of their own, and the assertion cannot be evaluated. This
    is an error, not a failed assertion.

    The operation does not change the state. It returns its qubits in the
    same order, which pins the assertion to this point in the program.

    Example:
    ```mlir
    // (|00> + |11>) / sqrt(2)
    %a3, %b2 = qassert.eq dense<[(0.7071067811865476, 0.0), (0.0, 0.0),
                                 (0.0, 0.0), (0.7071067811865476, 0.0)]>
        : tensor<4xcomplex<f64>> %a2, %b1 : !qco.qubit, !qco.qubit
    ```
    }];

  let arguments = (ins ElementsAttr:$reference,
      DefaultValuedAttr<F64Attr, "1.0">:$similarity_threshold,
      Arg<Variadic<QubitType>, "the target qubits">:$qubits_in);
  let assemblyFormat = "$reference $qubits_in attr-dict `:` type($qubits_in)";
}

def SpanOp : QAssertQubitsOp<"span"> {
  let summary = "Assert that the given qubits can only be measured in the "
                "given basis states";
  let description = [{
    Asserts that measuring `qubits_in` in the computational basis gives one
    of the bitstrings in `states` with probability 1. Equivalently, every
    basis state with a nonzero amplitude is listed. The amplitudes and their
    phases do not matter, and the qubits may be entangled with other qubits.

    Character `i` of each bitstring belongs to operand `i`, so the first
    character is the most-significant bit, as in `qassert.eq`. For example,
    `a = |1>` and `b = |0>` give `["10"]` for the operands `a, b`. A single
    bitstring asserts a basis state, up to a global phase.

    Each bitstring has one `0` or `1` per qubit, and no bitstring appears
    twice.

    The operation does not change the state. It returns its qubits in the
    same order, which pins the assertion to this point in the program.

    Example:
    ```mlir
    // (|00> + |11>) / sqrt(2) can only be measured as 00 or 11
    %a3, %b2 = qassert.span ["00", "11"] %a2, %b1 : !qco.qubit, !qco.qubit
    ```
    }];

  let arguments = (ins StrArrayAttr:$states,
      Arg<Variadic<QubitType>, "the target qubits">:$qubits_in);
  let assemblyFormat = "$states $qubits_in attr-dict `:` type($qubits_in)";
}

#endif // MLIR_DIALECT_QASSERT_IR_QASSERTOPS_TD
