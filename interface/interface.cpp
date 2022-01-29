//
#include "Python.h"
// #include <numpy/arrayobject.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//
#include "interface.h"

static PyObject *INTERFACE_error = NULL;

static PyObject *INTERFACE_version(PyObject *self) {
    return Py_BuildValue("s", "INTERFACE version 0.1");
}

static inline PyObject* nullReturnValue() {
    return Py_BuildValue("i", 0);
}

// Register the methods to be made available Python side
static PyMethodDef INTERFACE_methods[] = {
    //
    {"createAndTrain", INTERFACE_createAndTrain, METH_VARARGS,
     "Blablabla"
     "parameter."},
    //
    {"version", (PyCFunction)INTERFACE_version, METH_VARARGS,
     "Returns the version of the INTERFACE library."},

    {NULL, NULL, 0, NULL}};

//
static PyModuleDef INTERFACE_module = {
    PyModuleDef_HEAD_INIT, "INTERFACE", "INTERFACE blabla", -1,
    INTERFACE_methods};

//
PyMODINIT_FUNC PyInit_INTERFACE() {
    PyObject *obj = PyModule_Create(&INTERFACE_module);

    if (!obj) return NULL;

    INTERFACE_error = PyErr_NewException("INTERFACE.error", NULL, NULL);
    Py_XINCREF(INTERFACE_error);

    if (PyModule_AddObject(obj, "error", INTERFACE_error) < 0) {
        Py_XDECREF(INTERFACE_error);
        Py_CLEAR(INTERFACE_error);
        Py_DECREF(obj);
        return NULL;
    }

    return obj;
}

static parameters_t pyObjectToParameters_t(PyObject* pyObject, Py_buffer* pyView) {
    return (parameters_t) {};
}

static PyObject* INTERFACE_createAndTrain(PyObject *self, PyObject *args) {
    PyObject *parameters_object;
    Py_buffer parameters_view;
    if (!PyArg_ParseTuple(args, "s", &parameters_object)) {
        fprintf(stderr, "<!> INTERFACE_createAndTrain(): PyArg_ParseTuple\n");
        return NULL;
    }
    printParameters(pyObjectToParameters_t(parameters_object, &parameters_view));


    /* Build the output tuple */
    return nullReturnValue();
}

int main(void) {
    return 0;
}