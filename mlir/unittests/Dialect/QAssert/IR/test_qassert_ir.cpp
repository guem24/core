/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qassert_ir.cpp
/// Unit tests for the QAssert MLIR dialect.

#include "mqt/Dialect/QAssert/IR/QAssertDialect.h"
#include "mqt/Dialect/QAssert/IR/QAssertOps.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include <memory>
#include <string>

using namespace mlir;

namespace {
class QAssertIRTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<func::FuncDialect, qassert::QAssertDialect, qco::QCODialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  /// Expects that parsing @p source fails with a diagnostic containing
  /// @p expected.
  void expectRejected(StringRef source, StringRef expected) const {
    std::string diagnostics;
    ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& message) {
      diagnostics += message.str();
      diagnostics += '\n';
      return success();
    });
    EXPECT_FALSE(parse(source));
    EXPECT_TRUE(StringRef(diagnostics).contains(expected)) << diagnostics;
  }

  /// Returns the only operation of type @p OpT in @p moduleOp, or a null op.
  template <typename OpT> static OpT findSingle(ModuleOp moduleOp) {
    OpT found;
    size_t count = 0;
    moduleOp->walk([&](OpT op) {
      found = op;
      ++count;
    });
    return count == 1 ? found : OpT();
  }
};

TEST_F(QAssertIRTest, ParsesSupSingleQubitAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit
        %q2 = qassert.sup %q1 : !qco.qubit
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto sup = findSingle<qassert::SupOp>(*moduleOp);
  ASSERT_TRUE(sup);
  EXPECT_EQ(sup.getQubitsIn().size(), 1);
  EXPECT_EQ(sup.getQubitsOut().size(), 1);
}

TEST_F(QAssertIRTest, ParsesSupMultiQubitAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qco.h %a0 : !qco.qubit
        %a2, %b1 = qco.ctrl(%a1) targets (%t0 = %b0) {
          %t1 = qco.x %t0 : !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
        %a3, %b2 = qassert.sup %a2, %b1 : !qco.qubit, !qco.qubit
        qco.sink %a3 : !qco.qubit
        qco.sink %b2 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto sup = findSingle<qassert::SupOp>(*moduleOp);
  ASSERT_TRUE(sup);
  EXPECT_EQ(sup.getQubitsIn().size(), 2);
  EXPECT_EQ(sup.getQubitsOut().size(), 2);
}

TEST_F(QAssertIRTest, RejectsSupZeroQubits) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        "qassert.sup"() : () -> ()
        return
      }
    }
  )mlir",
                 "number of qubits must be at least 1");
}

TEST_F(QAssertIRTest, RejectsSupResultCountMismatch) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %r = "qassert.sup"(%a, %b) : (!qco.qubit, !qco.qubit) -> !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir",
                 "input and output qubit types match");
}

TEST_F(QAssertIRTest, ParsesEntAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %c0 = qco.alloc : !qco.qubit
        %a1, %b1, %c1 = qassert.ent %a0, %b0, %c0 : !qco.qubit, !qco.qubit, !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        qco.sink %c1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto ent = findSingle<qassert::EntOp>(*moduleOp);
  ASSERT_TRUE(ent);
  EXPECT_EQ(ent.getQubitsIn().size(), 3);
  EXPECT_EQ(ent.getQubitsOut().size(), 3);
}

TEST_F(QAssertIRTest, RejectsEntOneQubit) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q = qco.alloc : !qco.qubit
        %r = qassert.ent %q : !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir",
                 "number of qubits must be at least 2");
}

TEST_F(QAssertIRTest, RejectsEntResultCountMismatch) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %r = "qassert.ent"(%a, %b) : (!qco.qubit, !qco.qubit) -> !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir",
                 "input and output qubit types match");
}

TEST_F(QAssertIRTest, ParsesEqAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1, %b1 = qassert.eq dense<[(0.7071067811865476, 0.0), (0.0, 0.0),
                                     (0.0, 0.0), (0.7071067811865476, 0.0)]>
            : tensor<4xcomplex<f64>> %a0, %b0 : !qco.qubit, !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto eq = findSingle<qassert::EqOp>(*moduleOp);
  ASSERT_TRUE(eq);
  EXPECT_EQ(eq.getQubitsIn().size(), 2);
  EXPECT_EQ(eq.getQubitsOut().size(), 2);
  EXPECT_EQ(eq.getReference().getNumElements(), 4);
  EXPECT_DOUBLE_EQ(eq.getSimilarityThreshold().convertToDouble(), 1.0);
}

TEST_F(QAssertIRTest, ParsesEqRoundedReference) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[(0.707, 0.0), (0.707, 0.0)]>
            : tensor<2xcomplex<f64>> %q0 {similarity_threshold = 0.999 : f64}
            : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto eq = findSingle<qassert::EqOp>(*moduleOp);
  ASSERT_TRUE(eq);
  EXPECT_DOUBLE_EQ(eq.getSimilarityThreshold().convertToDouble(), 0.999);
}

TEST_F(QAssertIRTest, RejectsEqZeroQubits) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        "qassert.eq"() <{reference = dense<[(1.0, 0.0)]> : tensor<1xcomplex<f64>>}> : () -> ()
        return
      }
    }
  )mlir",
                 "number of qubits must be at least 1");
}

TEST_F(QAssertIRTest, RejectsEqRankTwoReference) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[[(1.0, 0.0)], [(0.0, 0.0)]]>
            : tensor<2x1xcomplex<f64>> %q0 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "reference must be a rank-one tensor");
}

TEST_F(QAssertIRTest, RejectsEqNonComplexReference) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[1.0, 0.0]> : tensor<2xf64> %q0 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "reference elements must have type complex<f64>");
}

TEST_F(QAssertIRTest, RejectsEqWrongReferenceLength) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]>
            : tensor<4xcomplex<f64>> %q0 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "reference must have 2^n amplitudes");
}

TEST_F(QAssertIRTest, RejectsEqNaNReference) {
  // NaN, not infinity: an infinite amplitude already fails the norm check.
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[(0x7FF8000000000000, 0.0), (0.0, 0.0)]>
            : tensor<2xcomplex<f64>> %q0 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "reference amplitudes must be finite");
}

TEST_F(QAssertIRTest, RejectsEqUnnormalizedReference) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[(1.0, 0.0), (1.0, 0.0)]>
            : tensor<2xcomplex<f64>> %q0 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "squared reference amplitudes must sum to 1");
}

TEST_F(QAssertIRTest, RejectsEqThresholdAboveOne) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[(1.0, 0.0), (0.0, 0.0)]>
            : tensor<2xcomplex<f64>> %q0 {similarity_threshold = 1.5 : f64}
            : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "similarity threshold must be in [0, 1]");
}

TEST_F(QAssertIRTest, RejectsEqNaNThreshold) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.eq dense<[(1.0, 0.0), (0.0, 0.0)]>
            : tensor<2xcomplex<f64>> %q0
            {similarity_threshold = 0x7FF8000000000000 : f64} : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "similarity threshold must be in [0, 1]");
}

TEST_F(QAssertIRTest, RejectsEqResultCountMismatch) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %r = "qassert.eq"(%a, %b) <{reference = dense<[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]> : tensor<4xcomplex<f64>>}> : (!qco.qubit, !qco.qubit) -> !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir",
                 "input and output qubit types match");
}

TEST_F(QAssertIRTest, ParsesSpanAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1, %b1 = qassert.span ["00", "11"] %a0, %b0 : !qco.qubit, !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto span = findSingle<qassert::SpanOp>(*moduleOp);
  ASSERT_TRUE(span);
  EXPECT_EQ(span.getQubitsIn().size(), 2);
  EXPECT_EQ(span.getQubitsOut().size(), 2);
  ASSERT_EQ(span.getStates().size(), 2);
  EXPECT_EQ(cast<StringAttr>(span.getStates()[0]).getValue(), "00");
}

TEST_F(QAssertIRTest, ParsesSpanBasisState) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.x %q0 : !qco.qubit
        %q2 = qassert.span ["1"] %q1 : !qco.qubit
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto span = findSingle<qassert::SpanOp>(*moduleOp);
  ASSERT_TRUE(span);
  EXPECT_EQ(span.getStates().size(), 1);
}

TEST_F(QAssertIRTest, RejectsSpanZeroQubits) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        "qassert.span"() <{states = ["0"]}> : () -> ()
        return
      }
    }
  )mlir",
                 "number of qubits must be at least 1");
}

TEST_F(QAssertIRTest, RejectsSpanEmptyStates) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.span [] %q0 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "states must contain at least one bitstring");
}

TEST_F(QAssertIRTest, RejectsSpanWrongBitstringLength) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1, %b1 = qassert.span ["0"] %a0, %b0 : !qco.qubit, !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "state '0' must have one '0' or '1'");
}

TEST_F(QAssertIRTest, RejectsSpanInvalidCharacter) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1, %b1 = qassert.span ["02"] %a0, %b0 : !qco.qubit, !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "state '02' must have one '0' or '1'");
}

TEST_F(QAssertIRTest, RejectsSpanDuplicateState) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1, %b1 = qassert.span ["00", "00"] %a0, %b0 : !qco.qubit, !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "state '00' appears more than once");
}

TEST_F(QAssertIRTest, RejectsSpanResultCountMismatch) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %r = "qassert.span"(%a, %b) <{states = ["00"]}> : (!qco.qubit, !qco.qubit) -> !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir",
                 "input and output qubit types match");
}

// QCO rejects assertions in all modifier bodies with one shared check, so one
// assertion per modifier is enough.
TEST_F(QAssertIRTest, RejectsAssertionInControlBody) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1, %b1 = qco.ctrl(%a0) targets (%t0 = %b0) {
          %t1 = qassert.sup %t0 : !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "body must contain only unitary operations");
}

TEST_F(QAssertIRTest, RejectsAssertionInInverseBody) {
  expectRejected(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.inv(%t0 = %q0) {
          %t1 = qassert.sup %t0 : !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "body must contain only unitary operations");
}

TEST_F(QAssertIRTest, RejectsAssertionInPowerBody) {
  expectRejected(R"mlir(
    module {
      func.func @main(%exponent : f64) {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.pow(%exponent) (%t0 = %q0) {
          %t1 = qassert.sup %t0 : !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : {!qco.qubit} -> {!qco.qubit}
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir",
                 "body must contain only unitary operations");
}

TEST_F(QAssertIRTest, AssertionsArePure) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qassert.sup %a0 : !qco.qubit
        %a2, %b1 = qassert.ent %a1, %b0 : !qco.qubit, !qco.qubit
        %b2 = qassert.eq dense<[(1.0, 0.0), (0.0, 0.0)]>
            : tensor<2xcomplex<f64>> %b1 : !qco.qubit
        %b3 = qassert.span ["0"] %b2 : !qco.qubit
        qco.sink %a2 : !qco.qubit
        qco.sink %b3 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto sup = findSingle<qassert::SupOp>(*moduleOp);
  auto ent = findSingle<qassert::EntOp>(*moduleOp);
  auto eq = findSingle<qassert::EqOp>(*moduleOp);
  auto span = findSingle<qassert::SpanOp>(*moduleOp);
  ASSERT_TRUE(sup);
  ASSERT_TRUE(ent);
  ASSERT_TRUE(eq);
  ASSERT_TRUE(span);
  EXPECT_TRUE(isPure(sup));
  EXPECT_TRUE(isPure(ent));
  EXPECT_TRUE(isPure(eq));
  EXPECT_TRUE(isPure(span));
}

} // namespace
