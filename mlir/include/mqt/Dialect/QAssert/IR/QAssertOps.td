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
    Asserts that `qubits_in` are in superposition: measuring them in the
    computational basis can give more than one result. This is the case if
    at least one of the qubits has no fixed result. To check each
    qubit individually, use one `qassert.sup` per qubit.

    Example:
    ```mlir
    // Only a is in superposition, but the assertion still holds.
    %a1 = qco.h %a0 : !qco.qubit
    %a2, %b1 = qassert.sup %a1, %b0 : !qco.qubit, !qco.qubit
    ```
    }];
}

def EntOp : QAssertQubitsOp<"ent"> {
  let summary = "Assert that the given qubits are entangled";
  let description = [{
    Asserts that `qubits_in` are entangled with each other as one group. The
    entanglement can also include qubits that are not listed.

    No matter how you divide the listed qubits into two parts, the parts are
    always connected by entanglement, directly or through other qubits. For
    example:
    - After a GHZ state on `a, b, c`, asserting `a, b` shows that `a` and `b`
      are connected through the GHZ state.
    - If `a, b` are a Bell pair and `c` was never entangled with them,
      asserting `a, b, c` fails.
    - If `a, b` and `c, d` are two separate Bell pairs, asserting
      `a, b, c, d` fails, although every qubit is entangled, since the two
      pairs are not connected.
    - `|+>|+>` fails: both qubits are in superposition, but they are not
      entangled.

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
    Asserts that `qubits_in` are in the state given by `reference`. The
    reference is scaled to length 1, and the assertion holds if the
    similarity `abs(<current|reference>)` is at least the
    `similarity_threshold`. The threshold lies in `[0, 1]` and its default
    value is 1, which means the two states must be equal. The global phase
    does not matter.

    `reference` lists `2^n` amplitudes, where `n` is the number of qubits.
    Each amplitude is a real number or a `(real, imaginary)` pair. The
    reference is stored as a rank-one tensor of `complex<f64>` and is always
    printed as a list.

    Instead of a list, `reference` can also be written in ket notation. The
    first qubit is the most-significant bit, so `|10>` means the first qubit
    is `|1>` and the second is `|0>`.

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
  let summary = "Assert that the given qubits can only be measured as one of "
                "the given bitstrings";
  let description = [{
    Asserts that measuring `qubits_in` can only give one of the provided
    bitstrings. Only the possible results count, not their probabilities.
    Qubits may also be entangled with other qubits.

    For example, the Bell state `(|00> + |11>) / sqrt(2)` can only be measured
    as `00` or `11`. So `["00", "11"]` holds, but `["00"]` fails, because `11`
    can also come out. A single bitstring such as `["10"]` asserts that the
    qubits are in that basis state.

    Each bitstring has one character per qubit. The first character belongs
    to the first qubit, which is the most-significant bit.

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
