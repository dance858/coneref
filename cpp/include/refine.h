#pragma once
#include "utils.h"
#include "eigen_includes.h"

/**
 * Refines an approximate solution to the conic linear program
 * 
 *  min. c^T x 
 *  s.t Ax + s = b
 *      s \in \mathcal{K}.
 * 
 * @param[in] A              Constraint matrix.
 * @param[in] b              Right-hand side of constraints.
 * @param[in] c              Cost vector.
 * @param[in] cones          The cones of the problem.
 * @param[in] z              The approximate solution.
 * @param[in] n              The number of x-variables.
 * @param[in] m              The number of linear constraints.
 * @param[in] ref_iter       Number of refinement steps.
 * @param[in] lsqr_iter      Number of LSQR iterations each refinement step.
 * 
 * @return ....
 * 
 * TODO: It should return a lot of information about the refinement procedure.
 */
Vector refine(const SparseMatrix &A, const Vector &b, const Vector &c, 
              const std::vector<Cone> &cones, Vector &z, int n, int m,
              int ref_iter, int lsqr_iter, bool verbose);
