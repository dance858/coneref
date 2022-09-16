#pragma once
#include "eigen_includes.h"

#define CONE_TOL (1e-8)
#define CONE_THRESH (1e-6)
#define EXP_CONE_MAX_ITERS (200)
const double EulerConstant = std::exp(1.0);
const double sqrt_two = std::sqrt(2.0);

enum ConeType { ZERO = 0, POS, SOC, PSD, EXP, EXP_DUAL };

class Cone {
public:
  ConeType type;
  std::vector<int> sizes;

  Cone(ConeType type, const std::vector<int> &sizes)
      : type(type), sizes(sizes){};
};


bool in_exp(const Eigen::Vector3d &x);
bool in_exp_dual(const Eigen::Vector3d &x);

/* Reconstructs the entire matrix based on it's lower triangular part.
*   If the function is called with x = [1, 2, 3, 4, 5, 6], then the matrix
*  [1            *           *]
*  [2/sqrt(2)    4           *]
*  [3/sqrt(2)    5/sqrt(2)   6]
*/
Matrix matrix_from_lower_triangular(const Vector &lower_tri);

/** Extracts the lower diagonal part of a matrix. If the function is called with 
*   the matrix
*   [1    *   *]
*   [2    4   *]
*   [3    5   6],
*   then x = [1, 2* sqrt(2), 3*sqrt(2), 4, 5*sqrt(2), 6]
*   is returned.
*/
Vector lower_triangular_from_matrix(const Matrix &matrix);

/** Returns the number of elements in the vectorization of a 
* n x n symmetric matrix. 
*/
int vectorized_psd_size(int n);
