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
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        "qassert.sup"() : () -> ()
        return
      }
    }
  )mlir"));
}

TEST_F(QAssertIRTest, RejectsSupResultCountMismatch) {
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

TEST_F(QAssertIRTest, RejectsSupInControlBody) {
  EXPECT_FALSE(parse(R"mlir(
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
  )mlir"));
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
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %q = qco.alloc : !qco.qubit
        %r = qassert.ent %q : !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir"));
}

TEST_F(QAssertIRTest, RejectsEntResultCountMismatch) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %r = "qassert.ent"(%a, %b) : (!qco.qubit, !qco.qubit) -> !qco.qubit
        qco.sink %r : !qco.qubit
        return
      }
    }
  )mlir"));
}

TEST_F(QAssertIRTest, RejectsEntInControlBody) {
  EXPECT_FALSE(parse(R"mlir(
    module {
      func.func @main() {
        %c0 = qco.alloc : !qco.qubit
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %c1, %a1, %b1 = qco.ctrl(%c0) targets (%t0 = %a0, %u0 = %b0) {
          %t1, %u1 = qassert.ent %t0, %u0 : !qco.qubit, !qco.qubit
          qco.yield %t1, %u1 : !qco.qubit, !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit, !qco.qubit})
        qco.sink %c1 : !qco.qubit
        qco.sink %a1 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir"));
}

TEST_F(QAssertIRTest, AssertionsArePure) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qassert.sup %a0 : !qco.qubit
        %a2, %b1 = qassert.ent %a1, %b0 : !qco.qubit, !qco.qubit
        qco.sink %a2 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  auto sup = findSingle<qassert::SupOp>(*moduleOp);
  auto ent = findSingle<qassert::EntOp>(*moduleOp);
  ASSERT_TRUE(sup);
  ASSERT_TRUE(ent);
  EXPECT_TRUE(isPure(sup));
  EXPECT_TRUE(isPure(ent));
}

} // namespace
