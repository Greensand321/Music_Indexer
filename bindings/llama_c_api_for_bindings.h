#pragma once

// Complete type for pybind11
struct llama_model {
    void *opaque;  // dummy payload
};

struct llama_context; // still opaque

extern "C" {
    // C API signatures (no re-declaration of llama_model)
    llama_model*    llama_load_model_from_file(const char* path);
    llama_context*  llama_new_context_with_model(llama_model* model);
    void            llama_eval(
                       llama_context* ctx,
                       int             gpu,
                       int             n_threads,
                       int             n_ctx,
                       const char*     prompt,
                       int             prompt_len,
                       void*           cb,
                       void*           user_data);
    int             llama_tokenize(llama_context* ctx, const char* str, int len);
    const char*     llama_token_to_str(llama_model* model, int token);
    void            llama_free(llama_context* ctx);

    #define LAMBDA_token_invalid (-1)
}
