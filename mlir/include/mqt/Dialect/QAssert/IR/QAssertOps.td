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

    Example:
    ```mlir
    %q1 = qco.h %q0 : !qco.qubit
    %q2 = qassert.sup %q1 : !qco.qubit
    ```
    }];
}

def EntOp : QAssertQubitsOp<"ent"> {
  let summary = "Assert that the given qubits are entangled";
  let description = [{
    Asserts that `qubits_in` are entangled with each other. This means that
    you cannot split them into two groups that each have a state of their
    own. The entanglement may also include qubits that are not listed.

    For example, after preparing a GHZ state on `a, b, c`, asserting `a, b`
    holds, because `a` and `b` are entangled through the GHZ state. If `c` was
    never entangled with `a` and `b`, asserting `a, b, c` fails. The state
    `|+>|+>` also fails: both qubits are in superposition, but each one has a
    state of its own.

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
    `reference`. The reference is scaled to length 1, and the assertion holds
    if `abs(<current|reference>)` is at least `similarity_threshold`, which
    lies in `[0, 1]` and defaults to 1. The global phase does not matter.

    `reference` lists `2^n` amplitudes, where `n` is the number of qubits.
    Each amplitude is a real number or a `(real, imaginary)` pair. The
    reference is stored as a rank-one tensor of `complex<f64>`. Operand 0 labels the
    most-significant basis bit, as in `qco.unitary`. For example, `a = |1>`
    and `b = |0>` give `[0, 0, 1, 0]` for the operands `a, b`.

    Instead of a list, `reference` can also be written in ket notation, such
    as `0.707|00> + 0.707|11>`. The bitstrings use the same bit order. A
    missing coefficient is 1, bitstrings that do not appear have amplitude 0,
    and a repeated bitstring adds up. Ket notation supports at most 20 qubits.
    The reference is always printed as a list.

    The squared amplitudes must sum to 1 within `1e-2`. This catches typos in
    the amplitudes but accepts rounded values.

    Example:
    ```mlir
    // (|00> + |11>) / sqrt(2)
    %a3, %b2 = qassert.eq %a2, %b1 == [0.707, 0, 0, 0.707]
        : !qco.qubit, !qco.qubit

    // The same state in ket notation
    %a4, %b3 = qassert.eq %a3, %b2 == 0.707|00> + 0.707|11>
        : !qco.qubit, !qco.qubit

    // |+i> = (|0> + i|1>) / sqrt(2), with a threshold
    %q1 = qassert.eq %q0 == [0.707, (0, 0.707)] threshold 0.999 : !qco.qubit
    ```
    }];

  let arguments = (ins ElementsAttr:$reference,
      DefaultValuedAttr<F64Attr, "1.0">:$similarity_threshold,
      Arg<Variadic<QubitType>, "the target qubits">:$qubits_in);
  let assemblyFormat = [{
    $qubits_in custom<Reference>($reference)
    (`threshold` $similarity_threshold^)? attr-dict `:` type($qubits_in)
  }];
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
