#include "llama_c_api_for_bindings.h"   // our binding-only header
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;
using ModelPtr = llama_model*;

// Load a GGUF model from disk
ModelPtr load_model(const std::string &path) {
    return llama_load_model_from_file(path.c_str());
}

// Run a single-pass generation
std::string chat(ModelPtr model, const std::string &prompt) {
    // create a fresh context for each call
    llama_context *ctx = llama_new_context_with_model(model);
    llama_eval(ctx,
               /*gpu*/       0,
               /*n_threads*/ 6,
               /*n_ctx*/     2048,
               prompt.c_str(),
               (int)prompt.size(),
               /*cb*/        nullptr,
               /*user_data*/ nullptr);

    std::string out;
    int token;
    // use the stub's invalid‐token constant
    while ((token = llama_tokenize(ctx, nullptr, 0)) != LAMBDA_token_invalid) {
        out += llama_token_to_str(model, token);
    }

    llama_free(ctx);
    return out;
}

// Expose to Python
PYBIND11_MODULE(llama_bindings, m) {
    m.doc() = "Pybind11 llama.cpp bindings";

    m.def("load_model", &load_model,
          "Load a GGUF model from file and return a handle");

    m.def("chat", &chat,
          "Generate text from a prompt using the loaded model");
}
