#include "NetworkInterface.hpp"
#include <functional>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#pragma region NteworkInterface Class binding
std::string getVersion() { return "0.1a"; }

void initNetworkInterface(py::module &m) {
  py::class_<NetworkInterface>(m, "NetworkInterface")
          .def(py::init<std::string>(), py::arg("parameter_filepath"))
          .def(py::init<>())
          .def("printJSONConfig",
               py::overload_cast<>(&NetworkInterface::printJSONConfig, py::const_))
          .def("onPrecisionChanged", py::overload_cast<const std::function<void(float)> &>(
                                             &NetworkInterface::onPrecisionChanged, py::const_))
          .def("createAndTrain", py::overload_cast<>(&NetworkInterface::createAndTrain));
}

#pragma endregion functions

PYBIND11_MODULE(kreps, m) {
  m.def("getVersion", &getVersion, "Current interface version");
  initNetworkInterface(m);
}
