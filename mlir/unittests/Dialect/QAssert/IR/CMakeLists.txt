# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qassert_ir_target mqt-core-mlir-unittest-qassert-ir)
add_executable(${qassert_ir_target} test_qassert_ir.cpp test_qassert_optimization.cpp)
target_link_libraries(
  ${qassert_ir_target}
  PRIVATE GTest::gtest_main
          MLIRFuncDialect
          MLIRQAssertDialect
          MLIRQCODialect
          MLIRParser
          MLIRSupport
          MLIRQCOTransforms
          MLIRTransforms
          MLIRPass)
mqt_mlir_configure_unittest_target(${qassert_ir_target})

gtest_discover_tests(${qassert_ir_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT 60)
