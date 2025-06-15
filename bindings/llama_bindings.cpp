#include <pybind11/pybind11.h>
#include "../third_party/llama.cpp/include/llama.h"
namespace py = pybind11;

using ModelPtr = llama_model*;

ModelPtr load_model(const std::string &path) {
    return llama_load_model_from_file(path.c_str());
}

std::string chat(ModelPtr model, const std::string &prompt) {
    llama_context *ctx = llama_new_context_with_model(model);
    llama_eval(ctx, 0, /*n_threads*/ 6, /*n_ctx*/ 2048,
               prompt.c_str(), prompt.size(), nullptr, nullptr);
    std::string out;
    int token;
    while ((token = llama_tokenize(ctx, nullptr, 0)) != LAMBDA_token_invalid) {
        out += llama_token_to_str(model, token);
    }
    llama_free(ctx);
    return out;
}

PYBIND11_MODULE(llama_bindings, m) {
    m.doc() = "Pybind11 llama.cpp bindings";
    m.def("load_model", &load_model);
    m.def("chat", &chat);
}
