#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "kalman.h"
#include "gating.h"
#include "association.h"

// The glue file
// Allows Python to call all the C++ files

namespace py = pybind11;

PYBIND11_MODULE(tracker_cpp, m) {
    m.doc() = "C++ acceleration module for radar tracking";

    // Kalman functions
    m.def("kalman_predict", &kalman_predict,
          "Kalman prediction step");

    m.def("kalman_update", &kalman_update,
          "Kalman update step");

    // Gating
    m.def("gate_measurements", &gate_measurements,
          "Return indices of measurements inside the gate");

    // Nearest Neighbor
    m.def("associate_nn", &associate_nn,
          "Nearest Neighbor association");

    // PDA
    py::class_<PDAResult>(m, "PDAResult")
        .def_readonly("z_fused", &PDAResult::z_fused)
        .def_readonly("betas", &PDAResult::betas);

    m.def("associate_pda", &associate_pda,
          "Probabilistic Data Association");
}
