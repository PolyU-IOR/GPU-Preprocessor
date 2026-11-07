# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

CMAKE ?= cmake
CTEST ?= ctest
BUILD_DIR ?= build
BUILD_JOBS ?= 4
CUDA_ARCH ?=
CMAKE_ARGS ?=
GPU ?= 0

.PHONY: all test help

all:
	"$(CMAKE)" -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release $(if $(strip $(CUDA_ARCH)),"-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)") $(CMAKE_ARGS)
	"$(CMAKE)" --build "$(BUILD_DIR)" --parallel "$(BUILD_JOBS)"

test:
	$(MAKE) all CMAKE_ARGS="$(CMAKE_ARGS) -DGPUPREPROCESSOR_BUILD_TESTS=ON"
	CTEST_BIN="$(CTEST)" bash tools/test.sh "$(GPU)" "$(BUILD_DIR)"

help:
	@echo 'make                         Build LP and QP with CUDA'
	@echo 'make test GPU=0              Build and run tests on an idle GPU'
	@echo 'Options: BUILD_DIR, BUILD_JOBS, CUDA_ARCH, CMAKE_ARGS'
