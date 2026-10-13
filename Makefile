CMAKE ?= cmake
BUILD_DIR ?= build
CUDA_ARCH ?= native
CMAKE_ARGS ?=

.PHONY: all
all:
	$(CMAKE) -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)" $(CMAKE_ARGS)
	$(CMAKE) --build "$(BUILD_DIR)" --parallel
