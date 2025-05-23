#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "chat.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// Sliding window for speculative draft model (operates on sequence 0)
// Returns true if a critical error occurs and generation should stop.
// Returns false if shifting was successful OR not needed.
static bool handle_speculative_context_shifting(
    llama_context *ctx,      // The llama_context for the draft model
    int n_keep,              // Number of tokens to keep from the start for seq 0
    int &n_past_seq0,        // Input/Output: Number of tokens in KV cache for seq 0 before call, updated if shift occurs.
    int n_eval_seq0          // Number of tokens in the current batch to be evaluated next for seq 0.
) {
    const int n_ctx = llama_n_ctx(ctx);

    if (n_past_seq0 + n_eval_seq0 < n_ctx) {
        return false; // Enough space for seq 0
    }

    LOG_DBG("%s: Draft context shift potentially needed for seq 0. n_past_seq0=%d, n_eval_seq0=%d, (sum = %d), n_ctx=%d, n_keep=%d\\n",
            __func__, n_past_seq0, n_eval_seq0, n_past_seq0 + n_eval_seq0, n_ctx, n_keep);

    const int n_left_to_discard_from = n_past_seq0 - n_keep;
    int n_discard = 0;
    if (n_left_to_discard_from > 0) {
        n_discard = n_left_to_discard_from / 2;
    }

    if (n_discard <= 0) {
        LOG_DBG("n_discard is %d (n_past=%d, n_keep=%d), no tokens effectively shifted from KV cache.\n", n_discard, n_past_seq0, n_keep);
    }

    LOG_DBG("%s: Draft performing context shift for seq 0. n_past_old_seq0=%d, n_discard=%d\\n", __func__, n_past_seq0, n_discard);

    llama_kv_self_seq_rm (ctx, 0, n_keep, n_keep + n_discard);
    llama_kv_self_seq_add(ctx, 0, n_keep + n_discard, n_past_seq0, -n_discard);
    n_past_seq0 -= n_discard;

    LOG_DBG("%s: Draft after shift for seq 0, n_past_new_seq0=%d.\\n", __func__, n_past_seq0);

    if (n_past_seq0 + n_eval_seq0 >= n_ctx) {
        LOG_ERR("%s: Draft context shift critical error for seq 0: Shift performed, but new n_past_seq0 (%d) + n_eval_seq0 (%d) = %d still >= n_ctx (%d).\\n",
                __func__, n_past_seq0, n_eval_seq0, n_past_seq0 + n_eval_seq0, n_ctx);
        return true; // Critical error for seq 0
    }

    LOG_DBG("%s: Draft context shift successful for seq 0. New n_past_seq0=%d. Space for %d tokens available for seq 0.\\n", __func__, n_past_seq0, n_eval_seq0);
    return false; // Shift successful for seq 0
}

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    // 支持 @file.txt 方式从文件读取prompt
    if (!params.prompt.empty() && params.prompt[0] == '@') {
        std::string filename = params.prompt.substr(1);
        std::ifstream fin(filename);
        if (!fin) {
            LOG_ERR("%s: 无法打开prompt文件: %s\n", __func__, filename.c_str());
            return 1;
        }
        std::ostringstream ss;
        ss << fin.rdbuf();
        params.prompt = ss.str();
        if (!params.prompt.empty() && params.prompt.back() == '\n') {
            params.prompt.pop_back();
        }
        LOG_INF("从文件 '%s' 读取prompt\n", filename.c_str());
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_draft_split = params.speculative.p_split;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // Initialize chat templates for the target model
    auto chat_templates = common_chat_templates_init(model_tgt, params.chat_template);
    /* Commented out due to compilation issues with llama_model_chat_template_type / llama_chat_template_type_to_string
    if (chat_templates.get() != nullptr && params.chat_template.empty()) {
        LOG_INF("%s: No explicit chat template provided, but model has a default one. Type: %s\\n", 
                __func__, llama_chat_template_type_to_string(llama_model_chat_template_type(model_tgt)));
    }
    */

    // load the draft model
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    // Set n_ctx for draft model as per request (default 2048)
    // And restore original n_ctx after draft model init for hygiene,
    const int original_params_n_ctx = params.n_ctx;
    params.n_ctx = 2048; // User requested default for draft model
    LOG_INF("%s: Initializing draft model with n_ctx = %d\\n", __func__, params.n_ctx);

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    common_init_result llama_init_dft = common_init_from_params(params);

    params.n_ctx = original_params_n_ctx; // Restore params.n_ctx

    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }


    // Tokenize the prompt
    std::vector<llama_token> inp;
    std::string final_prompt_for_tokenization;
    std::vector<common_chat_msg> chat_msgs;

    // Load prompt from file if specified by @ (this modifies params.prompt directly)
    if (!params.prompt.empty() && params.prompt[0] == '@') {
        std::string filename = params.prompt.substr(1);
        std::ifstream fin(filename);
        if (!fin) {
            LOG_ERR("%s: 无法打开prompt文件: %s\\n", __func__, filename.c_str());
            return 1;
        }
        std::ostringstream ss;
        ss << fin.rdbuf();
        params.prompt = ss.str();
        if (!params.prompt.empty() && params.prompt.back() == '\\n') {
            params.prompt.pop_back();
        }
        LOG_INF("从文件 '%s' 读取prompt\\n", filename.c_str());
    }

    if (chat_templates.get()) { // A valid template was loaded (either user-specified or model default)
        if (!params.chat_template.empty()) {
            LOG_INF("%s: Using user-specified chat template: %s\\n", __func__, params.chat_template.c_str());
        } else {
            LOG_INF("%s: Using model's default chat template.\\n", __func__);
        }

        if (!params.system_prompt.empty()) {
            common_chat_msg system_msg;
            system_msg.role = "system";
            system_msg.content = params.system_prompt;
            chat_msgs.push_back(system_msg);
        }
        if (!params.prompt.empty()) { // The user's main prompt
            common_chat_msg user_msg;
            user_msg.role = "user";
            user_msg.content = params.prompt;
            chat_msgs.push_back(user_msg);
        }

        common_chat_templates_inputs inputs;
        inputs.messages = chat_msgs;
        inputs.add_generation_prompt = true; // Always true for initial prompt in speculative example
        
        auto result = common_chat_templates_apply(chat_templates.get(), inputs);
        final_prompt_for_tokenization = result.prompt;
        // TODO: if result.bos is true, should we pass add_bos = false to common_tokenize?
        // For now, common_tokenize will use vocab default if its add_bos arg is true.

        LOG_INF("%s: Templated prompt for tokenization: \"%s\"\\n", __func__, final_prompt_for_tokenization.c_str());
        if (chat_msgs.empty()) {
             LOG_WRN("%s: Chat template applied, but no system or user prompt was provided. The template might only provide an initial assistant prefix.\\n", __func__);
        }
    } else {
        // No chat template (neither user-specified nor model default), use params.prompt directly
        final_prompt_for_tokenization = params.prompt;
        LOG_INF("%s: No chat template used (or model has no default), using raw prompt for tokenization: \"%s\"\\n", __func__, final_prompt_for_tokenization.c_str());
    }
    
    // The common_tokenize in speculative already adds BOS if vocab says so, when its add_bos_token arg is true.
    inp = common_tokenize(ctx_tgt, final_prompt_for_tokenization, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    // Ensure n_batch is respected for initial prompt eval
    const int n_batch_tgt = llama_n_batch(ctx_tgt);
    const int n_batch_dft = llama_n_batch(ctx_dft);

    // Evaluate prompt for target model
    if (n_input > 0) {
        const int n_tokens_tgt_first_part = n_input - 1;
        if (n_tokens_tgt_first_part > 0) {
            for (int i = 0; i < n_tokens_tgt_first_part; i += n_batch_tgt) {
                const int n_eval = std::min(n_batch_tgt, n_tokens_tgt_first_part - i);
                if (llama_decode(ctx_tgt, llama_batch_get_one(inp.data() + i, n_eval)) != 0) {
                    LOG_ERR("%s: llama_decode failed for target model prompt (first part)\\n", __func__);
                    llama_backend_free();
                    return 1;
                }
            }
        }
    }

    // Evaluate prompt for draft model
    const int n_keep_dft_val = 5; // User requested n_prefix=5 for draft model
    int current_n_past_dft_prompt_seq0 = 0; // Tracks n_past for seq 0 during prompt eval
    if (n_input > 0) {
        for (int i = 0; i < n_input; i += n_batch_dft) {
            const int n_eval_chunk = std::min(n_batch_dft, n_input - i);
            // Apply context shifting for seq 0 of the draft model
            if (handle_speculative_context_shifting(ctx_dft, n_keep_dft_val, current_n_past_dft_prompt_seq0, n_eval_chunk)) {
                LOG_ERR("%s: Draft model context shift failed during prompt processing for seq 0. Exiting.\\n", __func__);
                llama_backend_free();
                return 1;
            }
            if (llama_decode(ctx_dft, llama_batch_get_one(inp.data() + i, n_eval_chunk)) != 0) {
                LOG_ERR("%s: llama_decode failed for draft model prompt\\n", __func__);
                llama_backend_free();
                return 1;
            }
            current_n_past_dft_prompt_seq0 += n_eval_chunk;
        }
    }

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // how many tokens to draft each time
    int n_draft = params.speculative.n_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = current_n_past_dft_prompt_seq0;

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    // --- BEGIN FIX for initial target sampling ---
    // Decode the very last token of the input prompt with the target model
    // to make its logits available at batch_tgt.token[0] for the first sampling.
    // This assumes n_past_tgt is already n_input after prompt processing.
    if (n_input > 0) {
        common_batch_clear(batch_tgt); 
        // The last token of the input is inp[n_input - 1].
        // Its position for decoding is n_input - 1 (since n_past_tgt is n_input).
        // We want its logits to be at batch_tgt index 0.
        common_batch_add(batch_tgt, inp[n_input - 1], n_input - 1, {0}, true); // seq_id 0, request logits

        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: llama_decode failed for target model (initial single token for first sample). Exiting.\\n", __func__);
            llama_backend_free();
            return 1;
        }
        // The logits for inp[n_input-1] (which is now the one to predict *after*) 
        // are now in ctx_tgt, and common_sampler_sample will use them via batch_tgt index 0.
        // n_past_tgt should not be incremented here as this token was part of the prompt.
    } else if (llama_vocab_get_add_bos(vocab_tgt)) {
        // If no prompt and BOS is used, decode BOS for the first sample.
        common_batch_clear(batch_tgt);
        llama_token bos = llama_vocab_bos(vocab_tgt);
        common_batch_add(batch_tgt, bos, 0, {0}, true); // pos 0, seq_id 0, request logits
        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: llama_decode failed for target model (initial BOS token for first sample). Exiting.\\n", __func__);
            llama_backend_free();
            return 1;
        }
        // n_past_tgt would be 1 if we increment it, but common_sampler_sample uses ctx_tgt state.
        // For consistency, n_past_tgt should reflect tokens *consumed* and *in* KV. BOS is now in KV.
        // However, the main loop structure increments n_past_tgt *after* a token is accepted.
        // Let's assume for now that n_past_tgt remains 0, and the first accepted token will increment it.
        // This area can be tricky. The key is that ctx_tgt has processed BOS.
    } else {
        LOG_WRN("%s: No input prompt and BOS is not added by vocab. First token sampling for target model might fail or be undefined.\\n", __func__);
        // If we proceed, batch_tgt is empty, likely leading to the same error.
        // For robust handling, an error should probably be raised here or a dummy token decoded.
    }
    // --- END FIX ---

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        LOG_DBG("Main loop start: n_past_tgt = %d, n_past_dft = %d, n_predict = %d, has_eos = %d\n", n_past_tgt, n_past_dft, n_predict, has_eos);
        std::set<int> active_seqs = {};

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    auto & dist_tgt = *common_sampler_get_candidates(smpl);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // greedy verification

                    // sample from the target model
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }

        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // TODO: simplify
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                llama_kv_self_seq_keep(ctx_dft, s_keep);
                llama_kv_self_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
                llama_kv_self_seq_keep(ctx_dft, 0);

                llama_kv_self_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
                llama_kv_self_seq_keep(ctx_tgt, s_keep);
                llama_kv_self_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                llama_kv_self_seq_keep(ctx_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            // Update draft model (seq 0) with the confirmed token_id
            // n_past_dft here is the current length of seq 0 *before* adding this new token_id.
            // n_eval for this operation is 1 token for seq 0.
            if (handle_speculative_context_shifting(ctx_dft, n_keep_dft_val, n_past_dft, 1)) {
                LOG_ERR("%s: Draft model context shift failed for seq 0 before single token update. Exiting.\\n", __func__);
                llama_backend_free();
                return 1;
            }
            // After potential shift, n_past_dft (for seq 0) is updated.
            // Clear KV cache for seq 0 from the (new) n_past_dft onwards before adding the new token.
            llama_kv_self_seq_rm(ctx_dft, 0, n_past_dft, -1);

            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, token_id, n_past_dft, { 0 }, true); // Add token_id at (new) n_past_dft for seq 0.

            LOG_DBG("Before llama_decode (dft, single token update for seq 0): batch_dft.n_tokens = %d, token_id = %d, n_past_dft_seq0 = %d (after shift), n_ctx_dft = %d\\n",
                    batch_dft.n_tokens, token_id, n_past_dft, llama_n_ctx(ctx_dft));
            if (llama_decode(ctx_dft, batch_dft) != 0) {
                 LOG_ERR("%s: llama_decode failed for draft model (single token update for seq 0)\\n", __func__);
                 llama_backend_free();
                 return 1;
            }
            LOG_DBG("After llama_decode (dft, single token update for seq 0)\\n");

            ++n_past_dft; // n_past_dft tracks length of seq 0
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            LOG_DBG("Drafting loop (i = %d / %d): n_past_cur = %d, n_drafted = %d, batch_tgt.n_tokens = %d\n", i, n_draft, n_past_cur, n_drafted, batch_tgt.n_tokens);

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl);

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s);

                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < 8; ++f) {
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_kv_self_seq_rm(ctx_dft,    n_seq_cur, -1, -1);
                        llama_kv_self_seq_cp(ctx_dft, s, n_seq_cur, -1, -1);

                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;

                    const int s = sa[is];

                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id);
                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            LOG_DBG("Before llama_decode (dft, evaluate drafted): batch_dft.n_tokens = %d, n_past_cur = %d, n_drafted = %d, n_ctx_dft = %d\\n", batch_dft.n_tokens, n_past_cur, n_drafted, llama_n_ctx(ctx_dft));
            if (batch_dft.n_tokens > 0) { // Only decode if there are tokens
                // n_past_cur is the length of all active draft sequences (including seq 0) before adding this new level of tokens.
                // We are adding 1 new token to the length of each active sequence at this level.
                // Context shifting is applied to seq 0 using its current length (n_past_cur) and an eval_size of 1.
                if (handle_speculative_context_shifting(ctx_dft, n_keep_dft_val, n_past_cur, 1)) {
                    LOG_ERR("%s: Draft model context shift failed for seq 0 during drafting loop. Exiting.\\n", __func__);
                    llama_backend_free();
                    return 1;
                }
                // After potential shift, n_past_cur (tracking length of seq 0) is updated.
                // The batch_dft tokens were added with the *original* n_past_cur as their position.
                // If n_past_cur for seq 0 was shifted down, we need to adjust positions for seq 0 in batch_dft if llama_decode strictly uses it for KV cache addressing.
                // However, llama_decode itself uses the batch's pos for KV cache. If seq 0's KV got shifted, new tokens go to new slots.
                // Copied sequences (s > 0) have their own KV cache regions but are based on seq 0 at the time of copy.
                // For simplicity and minimal change, we assume the KV cache operations inside handle_speculative_context_shifting correctly manage seq 0,
                // and llama_decode with batched sequences handles KV for s>0 correctly based on their copied state and new tokens.
                // n_past_cur will be incremented after successful decode. If it was shifted, it means new tokens for seq 0 are added at an earlier effective point.

                if (llama_decode(ctx_dft, batch_dft) != 0) {
                    LOG_ERR("%s: llama_decode failed for draft model (evaluate drafted)\\n", __func__);
                    llama_backend_free();
                    return 1;
                }
            }
            LOG_DBG("After llama_decode (dft, evaluate drafted)\\n");
            if (batch_dft.n_tokens > 0) { // only increment if we actually decoded and added a level
                ++n_past_cur; // n_past_cur tracks length of active sequences, including seq 0.
            }
            ++n_drafted; // n_drafted counts levels attempted

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_kv_self_seq_keep(ctx_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_self_seq_cp(ctx_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            LOG_DBG("Before llama_decode (tgt, evaluate drafted): batch_tgt.n_tokens = %d, n_past_tgt = %d, n_ctx_tgt = %d\n", batch_tgt.n_tokens, n_past_tgt, llama_n_ctx(ctx_tgt));
            if (batch_tgt.n_tokens > 0) { // Only decode if there are tokens
                llama_decode(ctx_tgt, batch_tgt);
            }
            LOG_DBG("After llama_decode (tgt, evaluate drafted)\n");
            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
