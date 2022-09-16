#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cone_differentials.h"
#include "cone_projections.h"
#include "deriv.h"
#include "linop.h"
#include "lsqr.h"
#include "refine.h"


namespace py = pybind11;

PYBIND11_MODULE(_coneref, m) {
  m.doc() = "Refinement of conic linear programs, C++ Extension";
 
  py::class_<Cone>(m, "Cone")
      .def(py::init<ConeType, const std::vector<int> &>())
      .def_readonly("type", &Cone::type)
      .def_readonly("sizes", &Cone::sizes);
  py::enum_<ConeType>(m, "ConeType")
      .value("ZERO", ConeType::ZERO)
      .value("POS", ConeType::POS)
      .value("SOC", ConeType::SOC)
      .value("PSD", ConeType::PSD)
      .value("EXP", ConeType::EXP)
      .value("EXP_DUAL", ConeType::EXP_DUAL);

  m.def("SOC_Pi", &SOC_Pi);
  m.def("PSD_Pi", &PSD_Pi);
  m.def("exp_primal_Pi", &exp_primal_Pi);
  m.def("exp_dual_Pi", &exp_dual_Pi);
  m.def("prod_cone_Pi", &prod_cone_Pi);
  m.def("embedded_cone_Pi", &embedded_cone_Pi);

  m.def("SOC_Pi_diff", &SOC_Pi_diff);
  m.def("SDP_Pi_diff", &SDP_Pi_diff);
  m.def("exp_primal_Pi_diff", &exp_primal_Pi_diff);
  m.def("exp_dual_Pi_diff", &exp_dual_Pi_diff);
  m.def("prod_cone_Pi_diff", &prod_cone_Pi_diff);

  m.def("residual_map", &residual_map);
  m.def("residual_map_python_friendly", &residual_map_python_friendly);


  m.def("DR_operator", &DR_operator_memory_optimized);
  m.def("DN_operator", &DN_operator_optimized_memory);
  m.def("Q_operator", &Q_operator);
  m.def("refine", &refine);

}
