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

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <memory>
#include <string>
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
    func::registerInlinerExtension(registry);
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

  /// Runs canonicalize, then remove-dead-gates, on @p moduleOp.
  [[nodiscard]] LogicalResult runCleanup(ModuleOp moduleOp) const {
    PassManager pm(context.get());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(qco::createRemoveDeadGates());
    return pm.run(moduleOp);
  }

  /// Returns the number of operations of type @p OpT in @p moduleOp.
  template <typename OpT> static size_t count(ModuleOp moduleOp) {
    size_t n = 0;
    moduleOp->walk([&](OpT) { ++n; });
    return n;
  }

  /// Returns the operations of type @p OpT in @p moduleOp in program order.
  template <typename OpT> static SmallVector<OpT> collect(ModuleOp moduleOp) {
    SmallVector<OpT> ops;
    moduleOp->walk([&](OpT op) { ops.push_back(op); });
    return ops;
  }
};

//===----------------------------------------------------------------------===//
// Single assertions
//===----------------------------------------------------------------------===//

/// One assertion kind. Every case asserts on the same two qubits, so the tests
/// below share one program.
struct AssertionCase {
  const char* name;
  /// The operation name, such as `qassert.sup`.
  const char* opName;
  /// The assertion. It takes `%a1, %b0` and defines `%a2, %b1`.
  const char* assertion;
};

class QAssertSingleAssertionTest
    : public QAssertOptimizationTest,
      public ::testing::WithParamInterface<AssertionCase> {
protected:
  /// Returns @p program with its `ASSERTION` line replaced by the assertion of
  /// this case.
  [[nodiscard]] static std::string withAssertion(StringRef program) {
    constexpr StringRef placeholder = "ASSERTION";
    const size_t pos = program.find(placeholder);
    return (program.take_front(pos) + GetParam().assertion +
            program.drop_front(pos + placeholder.size()))
        .str();
  }

  /// Returns the operations named like the assertion of this case.
  [[nodiscard]] static SmallVector<Operation*> assertions(ModuleOp moduleOp) {
    SmallVector<Operation*> ops;
    moduleOp->walk([&](Operation* op) {
      if (op->getName().getStringRef() == GetParam().opName) {
        ops.push_back(op);
      }
    });
    return ops;
  }
};

TEST_P(QAssertSingleAssertionTest, CanonicalizeKeepsAssertion) {
  auto moduleOp = parse(withAssertion(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %ah = qco.h %a0 : !qco.qubit
        %a1 = qco.h %ah : !qco.qubit
        ASSERTION
        qco.sink %a2 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir"));
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, createCanonicalizerPass()).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 0);
  EXPECT_EQ(assertions(*moduleOp).size(), 1);
  EXPECT_EQ(count<qco::AllocOp>(*moduleOp), 2);
}

TEST_P(QAssertSingleAssertionTest, RemoveDeadGatesKeepsObservedGates) {
  auto moduleOp = parse(withAssertion(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qco.h %a0 : !qco.qubit
        ASSERTION
        %a3 = qco.x %a2 : !qco.qubit
        qco.sink %a3 : !qco.qubit
        qco.sink %b1 : !qco.qubit
        return
      }
    }
  )mlir"));
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, qco::createRemoveDeadGates()).succeeded());

  EXPECT_EQ(count<qco::XOp>(*moduleOp), 0);
  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
  auto ops = assertions(*moduleOp);
  ASSERT_EQ(ops.size(), 1);
  EXPECT_TRUE(ops.front()->getOperand(0).getDefiningOp<qco::HOp>());
}

TEST_P(QAssertSingleAssertionTest, RemoveDeadGatesKeepsAssertionInIf) {
  // The if contains only the assertion, so it has no memory effects.
  auto moduleOp = parse(withAssertion(R"mlir(
    module {
      func.func @main() -> i1 {
        %m0 = qco.alloc : !qco.qubit
        %p0 = qco.alloc : !qco.qubit
        %q0 = qco.alloc : !qco.qubit
        %m1, %c = qco.measure %m0 : !qco.qubit
        %p1 = qco.h %p0 : !qco.qubit
        %p2, %q1 = qco.if %c args(%a1 = %p1, %b0 = %q0)
            -> (!qco.qubit, !qco.qubit) {
          ASSERTION
          qco.yield %a2, %b1 : !qco.qubit, !qco.qubit
        } else args(%a1 = %p1, %b0 = %q0) {
          qco.yield %a1, %b0 : !qco.qubit, !qco.qubit
        }
        qco.sink %m1 : !qco.qubit
        qco.sink %p2 : !qco.qubit
        qco.sink %q1 : !qco.qubit
        return %c : i1
      }
    }
  )mlir"));
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, qco::createRemoveDeadGates()).succeeded());

  EXPECT_EQ(count<qco::IfOp>(*moduleOp), 1);
  EXPECT_EQ(assertions(*moduleOp).size(), 1);
  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
}

INSTANTIATE_TEST_SUITE_P(
    AllAssertions, QAssertSingleAssertionTest,
    ::testing::Values(
        AssertionCase{"Sup", "qassert.sup",
                      "%a2, %b1 = qassert.sup %a1, %b0 : !qco.qubit, "
                      "!qco.qubit"},
        AssertionCase{"Ent", "qassert.ent",
                      "%a2, %b1 = qassert.ent %a1, %b0 : !qco.qubit, "
                      "!qco.qubit"},
        AssertionCase{"Eq", "qassert.eq",
                      "%a2, %b1 = qassert.eq %a1, %b0 == [1, 0, 0, 0] : "
                      "!qco.qubit, !qco.qubit"},
        AssertionCase{"Span", "qassert.span",
                      "%a2, %b1 = qassert.span [\"00\"] %a1, %b0 : "
                      "!qco.qubit, !qco.qubit"}),
    [](const ::testing::TestParamInfo<AssertionCase>& info) {
      return std::string(info.param.name);
    });

//===----------------------------------------------------------------------===//
// Several assertions
//===----------------------------------------------------------------------===//

TEST_F(QAssertOptimizationTest, ChainedAssertionsSurviveCleanup) {
  // Bell pair, then all four assertions directly after each other.
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
        %a4, %b3 = qassert.ent %a3, %b2 : !qco.qubit, !qco.qubit
        %a5, %b4 = qassert.eq %a4, %b3 == [0.707, 0, 0, 0.707]
            : !qco.qubit, !qco.qubit
        %a6, %b5 = qassert.span ["00", "11"] %a5, %b4 : !qco.qubit, !qco.qubit
        qco.sink %a6 : !qco.qubit
        qco.sink %b5 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runCleanup(*moduleOp).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::CtrlOp>(*moduleOp), 1);
  auto sups = collect<qassert::SupOp>(*moduleOp);
  auto ents = collect<qassert::EntOp>(*moduleOp);
  auto eqs = collect<qassert::EqOp>(*moduleOp);
  auto spans = collect<qassert::SpanOp>(*moduleOp);
  ASSERT_EQ(sups.size(), 1);
  ASSERT_EQ(ents.size(), 1);
  ASSERT_EQ(eqs.size(), 1);
  ASSERT_EQ(spans.size(), 1);

  // The chain keeps its order and every qubit keeps its position.
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(sups[0].getQubitsIn()[i].getDefiningOp<qco::CtrlOp>());
    EXPECT_EQ(ents[0].getQubitsIn()[i], sups[0].getQubitsOut()[i]);
    EXPECT_EQ(eqs[0].getQubitsIn()[i], ents[0].getQubitsOut()[i]);
    EXPECT_EQ(spans[0].getQubitsIn()[i], eqs[0].getQubitsOut()[i]);
  }
}

TEST_F(QAssertOptimizationTest, OverlappingAssertionsKeepWires) {
  // GHZ on a, b, c. Each qubit flows through several assertions.
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %c0 = qco.alloc : !qco.qubit
        %a1 = qco.h %a0 : !qco.qubit
        %a2, %b1 = qco.ctrl(%a1) targets (%t0 = %b0) {
          %t1 = qco.x %t0 : !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
        %b2, %c1 = qco.ctrl(%b1) targets (%u0 = %c0) {
          %u1 = qco.x %u0 : !qco.qubit
          qco.yield %u1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
        %a3, %b3 = qassert.ent %a2, %b2 : !qco.qubit, !qco.qubit
        %b4, %c2 = qassert.ent %b3, %c1 : !qco.qubit, !qco.qubit
        %a4, %c3 = qassert.sup %a3, %c2 : !qco.qubit, !qco.qubit
        %a5, %b5, %c4 = qassert.span ["000", "111"] %a4, %b4, %c3
            : !qco.qubit, !qco.qubit, !qco.qubit
        qco.sink %a5 : !qco.qubit
        qco.sink %b5 : !qco.qubit
        qco.sink %c4 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runCleanup(*moduleOp).succeeded());

  EXPECT_EQ(count<qco::HOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::CtrlOp>(*moduleOp), 2);
  auto ents = collect<qassert::EntOp>(*moduleOp);
  auto sups = collect<qassert::SupOp>(*moduleOp);
  auto spans = collect<qassert::SpanOp>(*moduleOp);
  ASSERT_EQ(ents.size(), 2);
  ASSERT_EQ(sups.size(), 1);
  ASSERT_EQ(spans.size(), 1);

  auto entAB = ents[0];
  auto entBC = ents[1];
  // b leaves the first ent at position 1 and enters the second at position 0.
  EXPECT_EQ(entBC.getQubitsIn()[0], entAB.getQubitsOut()[1]);
  // sup takes a from the first ent and c from the second.
  EXPECT_EQ(sups[0].getQubitsIn()[0], entAB.getQubitsOut()[0]);
  EXPECT_EQ(sups[0].getQubitsIn()[1], entBC.getQubitsOut()[1]);
  // span gets a, b, c back in that order.
  EXPECT_EQ(spans[0].getQubitsIn()[0], sups[0].getQubitsOut()[0]);
  EXPECT_EQ(spans[0].getQubitsIn()[1], entBC.getQubitsOut()[0]);
  EXPECT_EQ(spans[0].getQubitsIn()[2], sups[0].getQubitsOut()[1]);
}

TEST_F(QAssertOptimizationTest, ReorderedOperandsKeepWires) {
  // a = |1>, b = |0>. The second span lists the qubits as (b, a), so its
  // bitstring is "01"; eq is back in the order (a, b), so "10" is index 2.
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qco.x %a0 : !qco.qubit
        %a2, %b1 = qassert.span ["10"] %a1, %b0 : !qco.qubit, !qco.qubit
        %b2, %a3 = qassert.span ["01"] %b1, %a2 : !qco.qubit, !qco.qubit
        %a4, %b3 = qassert.eq %a3, %b2 == [0, 0, 1, 0] : !qco.qubit, !qco.qubit
        qco.sink %a4 : !qco.qubit
        qco.sink %b3 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runCleanup(*moduleOp).succeeded());

  EXPECT_EQ(count<qco::XOp>(*moduleOp), 1);
  auto spans = collect<qassert::SpanOp>(*moduleOp);
  auto eqs = collect<qassert::EqOp>(*moduleOp);
  ASSERT_EQ(spans.size(), 2);
  ASSERT_EQ(eqs.size(), 1);

  auto straight = spans[0];
  auto swapped = spans[1];
  EXPECT_TRUE(straight.getQubitsIn()[0].getDefiningOp<qco::XOp>());
  EXPECT_EQ(swapped.getQubitsIn()[0], straight.getQubitsOut()[1]);
  EXPECT_EQ(swapped.getQubitsIn()[1], straight.getQubitsOut()[0]);
  EXPECT_EQ(eqs[0].getQubitsIn()[0], swapped.getQubitsOut()[1]);
  EXPECT_EQ(eqs[0].getQubitsIn()[1], swapped.getQubitsOut()[0]);
}

TEST_F(QAssertOptimizationTest, InterleavedAssertionsBlockGateCancellation) {
  // h, cx, cx, h is the identity, so without the assertions the cleanup
  // removes every gate. Each assertion observes the state between two gates of
  // a pair, so the pairs must not cancel. Only the final y is unobserved.
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() {
        %a0 = qco.alloc : !qco.qubit
        %b0 = qco.alloc : !qco.qubit
        %a1 = qco.h %a0 : !qco.qubit
        %a2 = qassert.sup %a1 : !qco.qubit
        %a3, %b1 = qco.ctrl(%a2) targets (%t0 = %b0) {
          %t1 = qco.x %t0 : !qco.qubit
          qco.yield %t1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
        %a4, %b2 = qassert.ent %a3, %b1 : !qco.qubit, !qco.qubit
        %a5, %b3 = qco.ctrl(%a4) targets (%u0 = %b2) {
          %u1 = qco.x %u0 : !qco.qubit
          qco.yield %u1 : !qco.qubit
        } : ({!qco.qubit}, {!qco.qubit})
        %a6 = qco.h %a5 : !qco.qubit
        %a7, %b4 = qassert.span ["00"] %a6, %b3 : !qco.qubit, !qco.qubit
        %a8 = qco.y %a7 : !qco.qubit
        qco.sink %a8 : !qco.qubit
        qco.sink %b4 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runCleanup(*moduleOp).succeeded());

  EXPECT_EQ(count<qco::YOp>(*moduleOp), 0);
  EXPECT_EQ(count<qco::HOp>(*moduleOp), 2);
  EXPECT_EQ(count<qco::CtrlOp>(*moduleOp), 2);
  auto sups = collect<qassert::SupOp>(*moduleOp);
  auto ents = collect<qassert::EntOp>(*moduleOp);
  auto spans = collect<qassert::SpanOp>(*moduleOp);
  ASSERT_EQ(sups.size(), 1);
  ASSERT_EQ(ents.size(), 1);
  ASSERT_EQ(spans.size(), 1);

  // Each assertion still sits between the same gates.
  EXPECT_TRUE(sups[0].getQubitsIn()[0].getDefiningOp<qco::HOp>());
  EXPECT_TRUE(ents[0].getQubitsIn()[0].getDefiningOp<qco::CtrlOp>());
  EXPECT_TRUE(ents[0].getQubitsIn()[1].getDefiningOp<qco::CtrlOp>());
  EXPECT_TRUE(spans[0].getQubitsIn()[0].getDefiningOp<qco::HOp>());
  EXPECT_TRUE(spans[0].getQubitsIn()[1].getDefiningOp<qco::CtrlOp>());
}

//===----------------------------------------------------------------------===//
// Inlining
//===----------------------------------------------------------------------===//

TEST_F(QAssertOptimizationTest, RemoveDeadGatesKeepsAssertionInNestedIf) {
  auto moduleOp = parse(R"mlir(
    module {
      func.func @main() -> i1 {
        %m0 = qco.alloc : !qco.qubit
        %a0 = qco.alloc : !qco.qubit
        %m1, %c = qco.measure %m0 : !qco.qubit
        %a1 = qco.if %c args(%x0 = %a0) -> (!qco.qubit) {
          %x1 = qco.x %x0 : !qco.qubit
          %x2 = qco.if %c args(%y0 = %x1) -> (!qco.qubit) {
            %y1 = qassert.sup %y0 : !qco.qubit
            qco.yield %y1 : !qco.qubit
          } else args(%y0 = %x1) {
            qco.yield %y0 : !qco.qubit
          }
          qco.yield %x2 : !qco.qubit
        } else args(%x0 = %a0) {
          qco.yield %x0 : !qco.qubit
        }
        qco.sink %m1 : !qco.qubit
        qco.sink %a1 : !qco.qubit
        return %c : i1
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, qco::createRemoveDeadGates()).succeeded());

  EXPECT_EQ(count<qco::IfOp>(*moduleOp), 2);
  EXPECT_EQ(count<qassert::SupOp>(*moduleOp), 1);
  EXPECT_EQ(count<qco::XOp>(*moduleOp), 1);
}

TEST_F(QAssertOptimizationTest, InlinerInlinesFunctionWithAssertion) {
  // Placement and routing and the QIR export need all calls that carry qubits
  // inlined.
  auto moduleOp = parse(R"mlir(
    module {
      func.func @check(%q : !qco.qubit) -> !qco.qubit {
        %h = qco.h %q : !qco.qubit
        %a = qassert.sup %h : !qco.qubit
        return %a : !qco.qubit
      }
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %q1 = func.call @check(%q0) : (!qco.qubit) -> !qco.qubit
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir");
  ASSERT_TRUE(moduleOp);

  ASSERT_TRUE(runPass(*moduleOp, createInlinerPass()).succeeded());

  auto main = moduleOp->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  size_t calls = 0;
  size_t sups = 0;
  main.walk([&](func::CallOp) { ++calls; });
  main.walk([&](qassert::SupOp sup) {
    ++sups;
    EXPECT_TRUE(sup.getQubitsIn()[0].getDefiningOp<qco::HOp>());
  });
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(sups, 1);
}

} // namespace
