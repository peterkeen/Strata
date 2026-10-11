#pragma once

namespace strata::program::adapt_fetch {
// Configuration admission only: an enabled mode still needs a fetchable layer,
// a nonzero PCIe share in that window and a suitable non-routed victim.
// Keep the existing exclusion order, including every batch configuration.
struct Config {
    bool adaptive_tier = false;
    bool residency = false;
    bool spec_split = false;
    bool layer_split = false;
    bool peer = false;
    bool helpers = false;
    bool async = false;
    bool pipeline = false;
    bool batch = false;
    bool copy_mode = true;
    bool gpu_plan = false;
    bool pcie_share = false;
    bool resident_ram = false;
};

enum class Reason {
    none, disabled, adaptive_tier_off, no_residency_table, spec_split,
    layer_split, peer, helper_gpus, adapt_async, pipeline_windows, batch,
    pcie_mode, no_gpu_plan, no_pcie_share, resident_ram
};

struct State {
    int requested;   // normalized STRATA_ADAPT_FETCH (0..2), including the existing default
    int effective;
    Reason reason;
};

constexpr State evaluate(int requested, const Config& c) {
    Reason r = Reason::none;
    if (requested <= 0) r = Reason::disabled;
    else if (!c.adaptive_tier) r = Reason::adaptive_tier_off;
    else if (!c.residency) r = Reason::no_residency_table;
    else if (c.spec_split) r = Reason::spec_split;
    else if (c.layer_split) r = Reason::layer_split;
    else if (c.peer) r = Reason::peer;
    else if (c.helpers) r = Reason::helper_gpus;
    else if (c.async) r = Reason::adapt_async;
    else if (c.pipeline) r = Reason::pipeline_windows;
    else if (c.batch) r = Reason::batch;
    else if (!c.copy_mode) r = Reason::pcie_mode;
    else if (!c.gpu_plan) r = Reason::no_gpu_plan;
    else if (!c.pcie_share) r = Reason::no_pcie_share;
    else if (c.resident_ram) r = Reason::resident_ram;
    return {requested, r == Reason::none ? requested : 0, r};
}

// Single-word values for the native INFO protocol and /metrics.engine.
constexpr const char* code(Reason r) {
    switch (r) {
        case Reason::none: return "none";
        case Reason::disabled: return "disabled";
        case Reason::adaptive_tier_off: return "adaptive_tier_off";
        case Reason::no_residency_table: return "no_residency_table";
        case Reason::spec_split: return "spec_split";
        case Reason::layer_split: return "layer_split";
        case Reason::peer: return "peer";
        case Reason::helper_gpus: return "helper_gpus";
        case Reason::adapt_async: return "adapt_async";
        case Reason::pipeline_windows: return "pipeline_windows";
        case Reason::batch: return "batch";
        case Reason::pcie_mode: return "pcie_mode";
        case Reason::no_gpu_plan: return "no_gpu_plan";
        case Reason::no_pcie_share: return "no_pcie_share";
        case Reason::resident_ram: return "resident_ram";
    }
    return "unknown";
}

// Preserve the existing human-readable diagnostics as well.
constexpr const char* description(Reason r, bool serving) {
    switch (r) {
        case Reason::none: return "none";
        case Reason::disabled: return "disabled";
        case Reason::adaptive_tier_off: return "the adaptive tier is off";
        case Reason::no_residency_table: return "no residency table";
        case Reason::spec_split: return "--spec-split";
        case Reason::layer_split: return "a layer split";
        case Reason::peer: return serving ? "--peer-device" : "a peer tier";
        case Reason::helper_gpus: return "helper GPUs";
        case Reason::adapt_async: return "--adapt-async";
        case Reason::pipeline_windows: return "--pipeline-windows";
        case Reason::batch: return "--batch";
        case Reason::pcie_mode: return "--pcie-mode dma / direct";
        case Reason::no_gpu_plan: return "no GPU plan";
        case Reason::no_pcie_share: return "no PCIe share";
        case Reason::resident_ram: return "the resident RAM mode";
    }
    return "unknown";
}
} // namespace strata::program::adapt_fetch
