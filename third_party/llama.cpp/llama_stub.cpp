#include "include/llama.h"
#include <string>
#include <cstring>

struct llama_model {
    std::string path;
};

struct llama_context {
    llama_model* model;
    std::string prompt;
    int idx;
};

llama_model* llama_load_model_from_file(const char* path) {
    llama_model* m = new llama_model();
    m->path = path ? path : "";
    return m;
}

llama_context* llama_new_context_with_model(llama_model* model) {
    llama_context* c = new llama_context();
    c->model = model;
    c->idx = 0;
    return c;
}

void llama_eval(llama_context* ctx, int gpu, int n_threads, int n_ctx,
                const char* prompt, int prompt_len, void* cb, void* user_data) {
    (void)gpu; (void)n_threads; (void)n_ctx; (void)cb; (void)user_data; (void)prompt_len;
    if (ctx) ctx->prompt = prompt ? prompt : "";
}

int llama_tokenize(llama_context* ctx, const char* str, int len) {
    if (!ctx || ctx->idx >= (int)ctx->prompt.size()) return LAMBDA_token_invalid;
    return ctx->prompt[ctx->idx++];
}

const char* llama_token_to_str(llama_model* model, int token) {
    (void)model;
    static char buf[2];
    buf[0] = (char)token;
    buf[1] = '\0';
    return buf;
}

void llama_free(llama_context* ctx) {
    delete ctx;
}

