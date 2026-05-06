#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

#include "blake3.h"
#include "llama.h"
#include "words.h"

const std::string MODEL_PATH = "llama/models/meta-llama-3.1-8b-instruct-q4_0.gguf";
const int GPU_OFFLOAD_LAYERS = 99;
const int N_PREDICT = 128;
const float TEMP = 1.2f;
const int TOP_K = 5000;
const bool DEBUG = false;

static void null_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static std::array<uint8_t, BLAKE3_OUT_LEN> hash(const std::vector<uint8_t> &data) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data.data(), data.size());

    std::array<uint8_t, BLAKE3_OUT_LEN> out;
    blake3_hasher_finalize(&hasher, out.data(), out.size());

    if (DEBUG) {
        fprintf(stderr, "blake3: ");
        for (size_t i = 0; i < out.size(); i++) {
            fprintf(stderr, "%02x", out[i]);
        }
        fprintf(stderr, "\n");
    }
    return out;
}

static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "%s: error: failed to open '%s'\n", __func__, path.c_str());
        std::exit(1);
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (size > 0 && !file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        fprintf(stderr, "%s: error: failed to read '%s'\n", __func__, path.c_str());
        std::exit(1);
    }
    return buffer;
}

static uint32_t seed_from_digest(const std::array<uint8_t, BLAKE3_OUT_LEN> &d) {
    return  uint32_t(d[0])
         | (uint32_t(d[1]) <<  8)
         | (uint32_t(d[2]) << 16)
         | (uint32_t(d[3]) << 24);
}

static std::string build_prompt(const std::array<uint8_t, BLAKE3_OUT_LEN> &d) {
    const char *w[3];
    for (int i = 0; i < 3; i++) {
        size_t off = 4 + 2 * i;
        size_t idx = (size_t(d[off]) | (size_t(d[off+1]) << 8)) % WORDS_LEN;
        w[i] = WORDS[idx];
    }
    return std::string("Write a short poem of five lines total. The poem has to include the following words: ")
        + w[0] + ", " + w[1] + ", " + w[2]
        + ". Only output the poem, and nothing else.";
}

static std::string apply_template(llama_model *model, const std::string &prompt) {
    bool fallback = false;
    int alloc_size = 512;
    std::vector<char> buf(alloc_size);
    std::vector<llama_chat_message> chat = {{"user", prompt.c_str()}};

    int32_t res = llama_chat_apply_template(
        model, nullptr, chat.data(), chat.size(), true, buf.data(), buf.size());
    if (res < 0) {
        // use chatml template if that didn't work
        res = llama_chat_apply_template(nullptr, "chatml", chat.data(), chat.size(), true, buf.data(), buf.size());
        fallback = true;
    }
    if ((size_t) res > buf.size()) {
        printf("resizing buffer!");
        buf.resize(res);
        res = llama_chat_apply_template(
            fallback ? nullptr : model,
            fallback ? "chatml" : nullptr,
            chat.data(), chat.size(), true, buf.data(), buf.size());
    }

    return std::string(buf.data(), res);
}

static llama_sampler *make_sampler(uint32_t seed) {
    // TODO use proper sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = !DEBUG;
    llama_sampler *chain = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(chain, llama_sampler_init_top_k(TOP_K));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(TEMP, 0.f, 1.f));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    return chain;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file>\n", argv[0]);
        return 1;
    }

    auto data = read_file(argv[1]);
    auto digest = hash(data);
    uint32_t seed = seed_from_digest(digest);
    std::string prompt = build_prompt(digest);

    llama_log_set(null_log_callback, NULL);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = GPU_OFFLOAD_LAYERS;

    llama_model *model = llama_load_model_from_file(MODEL_PATH.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    auto prompt_fmt = apply_template(model, prompt);

    // find the number of tokens in the prompt
    int n_prompt = -llama_tokenize(model, prompt_fmt.c_str(), prompt_fmt.size(), NULL, 0, false, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(model, prompt_fmt.c_str(), prompt_fmt.size(), prompt_tokens.data(), prompt_tokens.size(), false, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + N_PREDICT - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = !DEBUG;

    llama_context *ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    if (DEBUG) {
        printf("Prompt:");
        for (auto id : prompt_tokens) {
            char buf[128];
            int n = llama_token_to_piece(model, id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
        }
        printf("\n");
    }

    auto sampler = make_sampler(seed);

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + N_PREDICT; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(sampler, ctx, -1);

            // is it an end of generation?
            if (llama_token_is_eog(model, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(model, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    if (DEBUG) {
        fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

        fprintf(stderr, "\n");
        llama_perf_sampler_print(sampler);
        llama_perf_context_print(ctx);
        fprintf(stderr, "\n");
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
