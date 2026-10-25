/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qassert_optimization.cpp
/// Unit tests for QAssert operations in QCO optimization passes.

#include "mqt/Dialect/QAssert/IR/QAssertDialect.h"
#include "mqt/Dialect/QAssert/IR/QAssertOps.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include <cstddef>
#include <memory>
#include <utility>

using namespace mlir;

namespace {
class QAssertOptimizationTest : public ::testing::Test {
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

  [[nodiscard]] LogicalResult runPass(ModuleOp moduleOp,
                                      std::unique_ptr<Pass> pass) const {
    PassManager pm(context.get());
    pm.addPass(std::move(pass));
    return pm.run(moduleOp);
  }

  /// Returns the number of operations of type @p OpT in @p moduleOp.
  template <typename OpT> static size_t count(ModuleOp moduleOp) {
    size_t n = 0;
    moduleOp->walk([&](OpT) { ++n; });
    return n;
  }
};

TEST_F(QAssertOptimizationTest, CanonicalizeRemovesUnobservedQubit) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit
        %q2 = qco.h %q1 : !qco.qubit
        qco.sink %q2 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, createCanonicalizerPass()).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 0);
  EXPECT_EQ(count<qco::AllocOp>(*moduleOp), 0);
}

TEST_F(QAssertOptimizationTest, CanonicalizeKeepsSupAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit
        %q2 = qco.h %q1 : !qco.qubit
        %q3 = qassert.sup %q2 : !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, createCanonicalizerPass()).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 0);
  EXPECT_EQ(count<qassert::SupOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::AllocOp>(*moduleOp), 1);
}

TEST_F(QAssertOptimizationTest, RemoveDeadGatesKeepsSupObservedGates) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit
        %q2 = qassert.sup %q1 : !qco.qubit
        %q3 = qco.x %q2 : !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, qco::createRemoveDeadGates()).succeeded());

  EXPECT_EQ(count<qco::XOp>(*moduleOp), 0);
  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
  ASSERT_EQ(count<qassert::SupOp>(*moduleOp), 1);

  qassert::SupOp sup;
  moduleOp->walk([&](qassert::SupOp op) { sup = op; });
  EXPECT_TRUE(sup.getQubitsIn().front().getDefiningOp<qco::HOp>());
}

TEST_F(QAssertOptimizationTest, CanonicalizeKeepsEntAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qco.h %a0 : !qco.qubit
        %a2 = qco.h %a1 : !qco.qubit
        %a3, %b1 = qassert.ent %a2, %b0 : !qco.qubit, !qco.qubit
        qco.sink %a3 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, createCanonicalizerPass()).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 0);
  EXPECT_EQ(count<qassert::EntOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::AllocOp>(*moduleOp), 2);
}

TEST_F(QAssertOptimizationTest, RemoveDeadGatesKeepsEntObservedGates) {
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
        %a3, %b2 = qassert.ent %a2, %b1 : !qco.qubit, !qco.qubit
        %a4 = qco.y %a3 : !qco.qubit
        qco.sink %a4 : !qco.qubit
        qco.sink %b2 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, qco::createRemoveDeadGates()).succeeded());

  EXPECT_EQ(count<qco::YOp>(*moduleOp), 0);
  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::CtrlOp>(*moduleOp), 1);
  ASSERT_EQ(count<qassert::EntOp>(*moduleOp), 1);

  qassert::EntOp ent;
  moduleOp->walk([&](qassert::EntOp op) { ent = op; });
  for (Value qubit : ent.getQubitsIn()) {
    EXPECT_TRUE(qubit.getDefiningOp<qco::CtrlOp>());
  }
}

TEST_F(QAssertOptimizationTest, CanonicalizeKeepsEqAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit
        %q2 = qco.h %q1 : !qco.qubit
        %q3 = qassert.eq dense<[(1.0, 0.0), (0.0, 0.0)]>
            : tensor<2xcomplex<f64>> %q2 : !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, createCanonicalizerPass()).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 0);
  EXPECT_EQ(count<qassert::EqOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::AllocOp>(*moduleOp), 1);
}

TEST_F(QAssertOptimizationTest, RemoveDeadGatesKeepsEqObservedGates) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit
        %q2 = qassert.eq dense<[(0.7071067811865476, 0.0),
                                (0.7071067811865476, 0.0)]>
            : tensor<2xcomplex<f64>> %q1 : !qco.qubit
        %q3 = qco.x %q2 : !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, qco::createRemoveDeadGates()).succeeded());

  EXPECT_EQ(count<qco::XOp>(*moduleOp), 0);
  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
  ASSERT_EQ(count<qassert::EqOp>(*moduleOp), 1);

  qassert::EqOp eq;
  moduleOp->walk([&](qassert::EqOp op) { eq = op; });
  EXPECT_TRUE(eq.getQubitsIn().front().getDefiningOp<qco::HOp>());
}

TEST_F(QAssertOptimizationTest, CanonicalizeKeepsSpanAssertion) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qco.h %q0 : !qco.qubit
        %q2 = qco.h %q1 : !qco.qubit
        %q3 = qassert.span ["0"] %q2 : !qco.qubit
        qco.sink %q3 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, createCanonicalizerPass()).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 0);
  EXPECT_EQ(count<qassert::SpanOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::AllocOp>(*moduleOp), 1);
}

TEST_F(QAssertOptimizationTest, RemoveDeadGatesKeepsSpanObservedGates) {
  // Two qubits: on one qubit, |+> needs ["0", "1"], which is a full span.
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qco.h %a0 : !qco.qubit
        %a2, %b1 = qassert.span ["00", "10"] %a1, %b0 : !qco.qubit, !qco.qubit
        %a3 = qco.x %a2 : !qco.qubit
        qco.sink %a3 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, qco::createRemoveDeadGates()).succeeded());

  EXPECT_EQ(count<qco::XOp>(*moduleOp), 0);
  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
  ASSERT_EQ(count<qassert::SpanOp>(*moduleOp), 1);

  qassert::SpanOp span;
  moduleOp->walk([&](qassert::SpanOp op) { span = op; });
  EXPECT_TRUE(span.getQubitsIn().front().getDefiningOp<qco::HOp>());
}

} // namespace
