# Copyright (c) 2026 Kaihuang Chen, Jiawei Lu, Defeng Sun, Yancheng Yuan, Guojun Zhang, Xiangyu Zhang, Xinyuan Zhao
# SPDX-License-Identifier: MIT

CMAKE ?= cmake
BUILD_DIR ?= build
CUDA_ARCH ?=
CMAKE_ARGS ?=
BUILD_JOBS ?=

.PHONY: all
all:
	"$(CMAKE)" -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release $(if $(strip $(CUDA_ARCH)),"-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)") $(CMAKE_ARGS)
	"$(CMAKE)" --build "$(BUILD_DIR)" $(if $(strip $(BUILD_JOBS)),--parallel "$(BUILD_JOBS)")
