#include "eigen_includes.h"
#include "utils.h"


/* Returns the number of elements in the vectorization of a 
 n x n symmetric matrix. 
*/
int vectorized_psd_size(int n) { return n * (n + 1) / 2; }

/* Extracts the lower diagonal part of a matrix. If the function is called with 
   the matrix
   [1    *   *]
   [2    4   *]
   [3    5   6],
   then x = [1, 2* sqrt(2), 3*sqrt(2), 4, 5*sqrt(2), 6]
   is returned.
*/
Vector lower_triangular_from_matrix(const Matrix &matrix) {
  int n = matrix.rows();
  Vector lower_tri = Vector::Zero(vectorized_psd_size(n));
  int offset = 0;
  for (int col = 0; col < n; ++col) {
    for (int row = col; row < n; ++row) {
      if (row != col) {
        lower_tri[offset] = matrix(row, col) * sqrt_two;
      } else {
        lower_tri[offset] = matrix(row, col);
      }
      ++offset;
    }
  }
  return lower_tri;
}

/* Reconstructs the entire matrix based on it's lower triangular part.
   If the function is called with x = [1, 2, 3, 4, 5, 6], then the matrix
  [1            *           *]
  [2/sqrt(2)    4           *]
  [3/sqrt(2)    5/sqrt(2)   6]
*/
Matrix matrix_from_lower_triangular(const Vector &lower_tri) {
  int n = static_cast<int>(std::sqrt(2 * lower_tri.size()));
  Matrix matrix = Matrix::Zero(n, n);
  int offset = 0;
  for (int col = 0; col < n; ++col) {
    for (int row = col; row < n; ++row) {
      if (row != col) {
        matrix(row, col) = lower_tri[offset] / sqrt_two;
        matrix(col, row) = lower_tri[offset] / sqrt_two;
      } else {
        matrix(row, col) = lower_tri[offset];
      }
      ++offset;
    }
  }
  return matrix;
}

bool in_exp(const Eigen::Vector3d &x) {
  return (x[0] <= 0 && std::abs(x[1]) <= CONE_THRESH && x[2] >= 0) ||
         (x[1] > 0 && x[1] * exp(x[0] / x[1]) - x[2] <= CONE_THRESH);
}

bool in_exp_dual(const Eigen::Vector3d &x) {
  return (std::abs(x[0]) <= CONE_THRESH && x[1] >= 0 && x[2] >= 0) ||
         (x[0] < 0 &&
          -x[0] * exp(x[1] / x[0]) - EulerConstant * x[2] <= CONE_THRESH);
}
