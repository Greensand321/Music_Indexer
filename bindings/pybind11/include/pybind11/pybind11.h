#pragma once
#include <string>
namespace pybind11 {
class module_ {
public:
    template <typename... Args>
    module_(Args&&...) {}
    template <typename Func>
    void def(const char*, Func) {}
    void attr(const char*, const char*) {}
};
}
#define PYBIND11_MODULE(name, variable) extern "C" void init_##name() {}
