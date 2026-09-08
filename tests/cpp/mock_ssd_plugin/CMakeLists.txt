# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Deliberately does NOT link openvino::runtime/genai: a real vendor SSD plugin only needs to compile
# openvino/genai/c/ssd_plugin_interface.h, proving the C ABI boundary is genuinely dependency-free.

set(SRC mock_ssd_plugin.cpp)

add_library(mock_ssd_plugin MODULE ${SRC})
target_include_directories(mock_ssd_plugin PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/include")

# A second copy that reports an incompatible api_version, for testing that KVCacheOffloadSSDPluginBackend
# rejects a version-mismatched plugin instead of loading it anyway.
add_library(mock_ssd_plugin_bad_version MODULE ${SRC})
target_include_directories(mock_ssd_plugin_bad_version PRIVATE "${OpenVINOGenAI_SOURCE_DIR}/src/cpp/include")
target_compile_definitions(mock_ssd_plugin_bad_version PRIVATE MOCK_SSD_PLUGIN_BAD_API_VERSION)

# A loadable library that is not a valid plugin at all (missing the required entry symbol), for negative
# testing of the loader.
add_library(not_a_plugin MODULE not_a_plugin.cpp)

foreach(target_name IN ITEMS mock_ssd_plugin mock_ssd_plugin_bad_version not_a_plugin)
    if(DEFINED TEST_TARGET_NAME)
        set_target_properties(${target_name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:${TEST_TARGET_NAME}>"
            RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:${TEST_TARGET_NAME}>"
        )
    endif()

    install(TARGETS ${target_name}
            COMPONENT tests
            LIBRARY DESTINATION tests
            RUNTIME DESTINATION tests
            ARCHIVE DESTINATION tests)
endforeach()
