// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// A real, loadable shared library that deliberately does NOT export `ov_genai_get_ssd_plugin`, used to
// test that KVCacheOffloadSSDPluginBackend distinguishes "the file exists and loads fine, but isn't a
// valid plugin" from "the file could not be loaded at all".

#ifdef _WIN32
#    define NOT_A_PLUGIN_EXPORT __declspec(dllexport)
#else
#    define NOT_A_PLUGIN_EXPORT
#endif

extern "C" {
NOT_A_PLUGIN_EXPORT int not_an_ssd_plugin_marker() {
    return 0;
}
}
