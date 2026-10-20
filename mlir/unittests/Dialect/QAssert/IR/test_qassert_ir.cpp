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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include <memory>

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

  /// Returns the only `qassert.sup` in @p moduleOp, or a null op.
  static qassert::SupOp findSingleSupOp(ModuleOp moduleOp) {
    qassert::SupOp found;
    size_t count = 0;
    moduleOp->walk([&](qassert::SupOp op) {
      found = op;
      ++count;
    });
    return count == 1 ? found : qassert::SupOp();
  }
};

TEST_F(QAssertIRTest, ParsesSingleQubitAssertion) {
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

  auto sup = findSingleSupOp(*moduleOp);
  ASSERT_TRUE(sup);
  EXPECT_EQ(sup.getQubitsIn().size(), 1);
  EXPECT_EQ(sup.getQubitsOut().size(), 1);
}

TEST_F(QAssertIRTest, ParsesMultiQubitAssertion) {
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

  auto sup = findSingleSupOp(*moduleOp);
  ASSERT_TRUE(sup);
  EXPECT_EQ(sup.getQubitsIn().size(), 2);
  EXPECT_EQ(sup.getQubitsOut().size(), 2);
}

TEST_F(QAssertIRTest, RejectsZeroQubits) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        "qassert.sup"() : () -> ()
        return
      }
    }
  )mlir"));
}

TEST_F(QAssertIRTest, RejectsResultCountMismatch) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %r = "qassert.sup"(%a, %b) : (!qco.qubit, !qco.qubit) -> !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir"));
}

TEST_F(QAssertIRTest, ReportsWriteEffect) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = qassert.sup %q0 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto sup = findSingleSupOp(*moduleOp);
  ASSERT_TRUE(sup);

  SmallVector<MemoryEffects::EffectInstance> effects;
  sup.getEffects(effects);
  ASSERT_EQ(effects.size(), 1);
  EXPECT_TRUE(isa<MemoryEffects::Write>(effects.front().getEffect()));
}

} // namespace
