# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Standalone reference tool for exercising SchedulerConfig::use_chunked_kv_cache end-to-end.
# Not part of the product samples list; configure/build it against an already-built
# openvino.genai tree, e.g.:
#
#   cmake -S tools/kv_cache_chunked_ref -B tools/kv_cache_chunked_ref/build \
#       -DOpenVINOGenAI_DIR=/path/to/openvino.genai/build
#   cmake --build tools/kv_cache_chunked_ref/build
#
# Run:
#   ./tools/kv_cache_chunked_ref/build/kv_cache_chunked_ref <MODEL_DIR> "<PROMPT>" GPU 1 16 128

cmake_minimum_required(VERSION 3.15)
project(kv_cache_chunked_ref CXX)

find_package(OpenVINOGenAI REQUIRED)

add_executable(kv_cache_chunked_ref kv_cache_chunked_ref.cpp)
target_link_libraries(kv_cache_chunked_ref PRIVATE openvino::genai)
set_target_properties(kv_cache_chunked_ref PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
