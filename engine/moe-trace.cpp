// moe-trace: dump MoE expert-routing decisions (selected expert id + normalized
// gate weight) for every token of a single prefill pass, one CSV row per
// (token position, layer, slot).
//
// Not an upstream llama.cpp tool: this is local research tooling used to
// measure expert-routing overlap across consecutive tokens.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <clocale>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

// the graph builder names per-layer tensors "<base>-<il>" (see
// llama_context::graph_get_cb in src/llama-context.cpp), so the layer index
// is read straight back out of the tensor name rather than tracked with a
// counter. this is exact even when some layers are dense (no MoE tensors
// emitted) instead of drifting like a plain increment-per-eval counter would.
static int parse_layer_index(const char * name) {
    const char * dash = strrchr(name, '-');
    if (!dash || !dash[1]) {
        return -1;
    }
    for (const char * p = dash + 1; *p; ++p) {
        if (!isdigit((unsigned char) *p)) {
            return -1;
        }
    }
    return atoi(dash + 1);
}

struct layer_moe_data {
    std::vector<int32_t> topk;      // flattened [token][slot], expert ids
    std::vector<float>   weights;   // flattened [token][slot], normalized gate weights
    int64_t n_slot = 0;
    int64_t n_tok  = 0;
    bool has_topk    = false;
    bool has_weights = false;
};

struct moe_trace_ctx {
    std::map<int, layer_moe_data> layers;
    std::vector<uint8_t>          scratch; // reused for non-host tensor copies
};

static bool moe_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = (moe_trace_ctx *) user_data;
    const bool is_topk    = strstr(t->name, "ffn_moe_topk")         != nullptr;
    const bool is_weights = strstr(t->name, "ffn_moe_weights_norm") != nullptr;
    if (!is_topk && !is_weights) {
        // not a tensor we care about: on ask, decline observation (scheduler can
        // batch it away). ask=false returning false would cancel the whole graph
        // compute, so never do that here.
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }

    const int il = parse_layer_index(t->name);
    if (il < 0) {
        LOG_ERR("%s: could not parse layer index from tensor name '%s', skipping\n", __func__, t->name);
        return true;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        const size_t n_bytes = ggml_nbytes(t);
        ctx->scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, ctx->scratch.data(), 0, n_bytes);
        data = ctx->scratch.data();
    }

    const int64_t n_slot = t->ne[0];
    const int64_t n_tok  = t->ne[1];
    layer_moe_data & ld = ctx->layers[il];
    ld.n_slot = n_slot;

    // APPEND, never overwrite. A prompt longer than the micro-batch size is prefilled in
    // several llama_decode calls, so this callback fires once per layer PER MICRO-BATCH.
    // Overwriting here would silently keep only the last chunk, which looks like a valid
    // capture and is wrong. Chunked prefill is required on this machine: a single-shot
    // prefill of ~19+ tokens touches enough of the 75 GB expert bank to exhaust RAM.
    if (is_topk) {
        const size_t base = ld.topk.size();
        ld.topk.resize(base + (size_t)(n_slot * n_tok));
        for (int64_t i1 = 0; i1 < n_tok; ++i1) {
            for (int64_t i0 = 0; i0 < n_slot; ++i0) {
                const uint8_t * p = data + i1 * t->nb[1] + i0 * t->nb[0];
                ld.topk[base + i1 * n_slot + i0] = *(const int32_t *) p;
            }
        }
        ld.has_topk = true;
    } else {
        const size_t base = ld.weights.size();
        ld.weights.resize(base + (size_t)(n_slot * n_tok));
        for (int64_t i1 = 0; i1 < n_tok; ++i1) {
            for (int64_t i0 = 0; i0 < n_slot; ++i0) {
                const uint8_t * p = data + i1 * t->nb[1] + i0 * t->nb[0];
                ld.weights[base + i1 * n_slot + i0] = *(const float *) p;
            }
        }
        ld.has_weights = true;
    }

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens - provide a prompt with -f\n", __func__);
        return false;
    }
    LOG_INF("%s: prompt tokens = %zu\n", __func__, tokens.size());

    // single prefill of the whole prompt - no generation loop
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("%s: llama_decode failed\n", __func__);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // -o / --output / --output-csv is handled by hand and stripped from argv
    // before common_params_parse, because the shared "-o" option (arg.cpp) is
    // scoped to a fixed set of examples that does not include the generic
    // LLAMA_EXAMPLE_COMMON parser this tool uses, and adding a new example
    // enum value would mean editing tracked llama.cpp source.
    std::string output_path;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-o" || arg == "--output" || arg == "--output-csv") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires an argument\n", argv[i]);
                return 1;
            }
            output_path = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    if (output_path.empty()) {
        fprintf(stderr, "error: -o OUTPUT_CSV is required\n");
        return 1;
    }

    const int filtered_argc = (int) filtered_argv.size();

    moe_trace_ctx trace;
    common_params params;
    common_init();
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = moe_trace_cb_eval;
    params.cb_eval_user_data = &trace;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    LOG_INF("\n%s\n\n", common_params_get_system_info(params).c_str());

    if (!run(ctx, params)) {
        return 1;
    }

    // --- write CSV + summary ---

    std::ofstream csv(output_path);
    if (!csv) {
        LOG_ERR("%s: could not open output file '%s'\n", __func__, output_path.c_str());
        return 1;
    }
    csv << std::setprecision(8);
    csv << "token_pos,layer,slot,expert_id,gate_weight\n";
    size_t   rows            = 0;
    int      layers_seen     = 0;
    int      layers_no_gate  = 0;
    int32_t  min_expert      = std::numeric_limits<int32_t>::max();
    int32_t  max_expert      = std::numeric_limits<int32_t>::min();
    float    min_weight      = std::numeric_limits<float>::infinity();
    float    max_weight      = -std::numeric_limits<float>::infinity();
    std::set<int64_t> token_positions_seen;
    for (const auto & kv : trace.layers) {
        const int il = kv.first;
        const layer_moe_data & ld = kv.second;
        if (!ld.has_topk) {
            continue;
        }
        layers_seen++;
        if (!ld.has_weights) {
            layers_no_gate++;
        }
        const int64_t n_tok_total = ld.n_slot > 0 ? (int64_t) (ld.topk.size() / ld.n_slot) : 0;
        for (int64_t tok = 0; tok < n_tok_total; ++tok) {
            token_positions_seen.insert(tok);
            for (int64_t slot = 0; slot < ld.n_slot; ++slot) {
                const size_t idx = (size_t) (tok * ld.n_slot + slot);
                const int32_t expert_id  = ld.topk[idx];
                const float   gate_weight = (ld.has_weights && idx < ld.weights.size())
                    ? ld.weights[idx]
                    : std::numeric_limits<float>::quiet_NaN();
                csv << tok << "," << il << "," << slot << "," << expert_id << "," << gate_weight << "\n";
                rows++;
                min_expert = std::min(min_expert, expert_id);
                max_expert = std::max(max_expert, expert_id);
                if (ld.has_weights && std::isfinite(gate_weight)) {
                    min_weight = std::min(min_weight, gate_weight);
                    max_weight = std::max(max_weight, gate_weight);
                }
            }
        }
    }
    csv.close();

    LOG_INF("\n%s: === summary ===\n", __func__);
    LOG_INF("%s: rows written        = %zu\n", __func__, rows);
    LOG_INF("%s: distinct layers     = %d\n", __func__, layers_seen);
    LOG_INF("%s: distinct token pos  = %zu\n", __func__, token_positions_seen.size());
    if (rows > 0) {
        LOG_INF("%s: expert id range     = [%d, %d]\n", __func__, min_expert, max_expert);
        if (layers_no_gate < layers_seen) {
            LOG_INF("%s: gate weight range   = [%.6f, %.6f]\n", __func__, min_weight, max_weight);
        }
    } else {
        LOG_WRN("%s: no MoE routing tensors captured - is this an MoE model?\n", __func__);
    }
    if (layers_no_gate > 0) {
        LOG_WRN("%s: %d layer(s) had no ffn_moe_weights_norm tensor (gate_weight written as nan) "
                "- model likely does not normalize gate weights\n", __func__, layers_no_gate);
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
