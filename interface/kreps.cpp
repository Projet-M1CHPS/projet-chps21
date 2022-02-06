#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "NetworkInterface.hpp"

namespace py = pybind11;

#pragma region functions
std::string getVersion() { return "0.1a"; }

void initNetworkInterface(py::module &m) {
  py::class_<NetworkInterface>(m, "NetworkInterface")
          .def(py::init<std::string>(), py::arg("parameter_filepath"))
          .def("printJSONConfig", py::overload_cast<>(&NetworkInterface::printJSONConfig, py::const_))
          .def("createAndTrain", py::overload_cast<>(&NetworkInterface::createAndTrain));
}

#pragma endregion functions

PYBIND11_MODULE(kreps, m) {
  m.def("getVersion", &getVersion, "Current interface version");
  initNetworkInterface(m);
}
