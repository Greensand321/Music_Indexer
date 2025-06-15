#pragma once
#include <Python.h>
#include <string>
#include <vector>

namespace pybind11 {
class module_ {
    const char* name;
    std::vector<PyMethodDef> methods;
public:
    explicit module_(const char* name) : name(name) {}
    struct doc_proxy { void operator=(const char*) {} };
    doc_proxy doc() { return {}; }
    template <typename Func>
    void def(const char* n, Func f, const char* = nullptr) {
        methods.push_back({n, reinterpret_cast<PyCFunction>(f), METH_VARARGS, nullptr});
    }
    void attr(const char*, const char*) {}
    PyObject* create() {
        methods.push_back({nullptr, nullptr, 0, nullptr});
        static PyModuleDef def;
        def = {PyModuleDef_HEAD_INIT, name, nullptr, -1, methods.data()};
        return PyModule_Create(&def);
    }
};
}
#define PYBIND11_MODULE(name, variable) \
    extern "C" PyObject* PyInit_##name(); \
    PyObject* PyInit_##name()
