#pragma once
#include "eigen_includes.h"
#include <functional>
#include <vector>

using VecFn = std::function<Vector(const Vector &)>;

class LinearOperator {
  /**
   * m x n linear operator
   */
public:
  const int m;
  const int n;
  
  // Note that these are not constant. The reason is that we want to be able
  // to modify some operators in place (see eg., the function DN_memory_optimized).
  // Modifying them in place gives a huge improvement in refinement time 
  // for small problem instances.
  VecFn matvec;                         
  VecFn rmatvec;                        

  explicit LinearOperator(int rows, int cols, const VecFn &matvec_in,
                          const VecFn &rmatvec_in)
      : m(rows), n(cols), matvec(matvec_in), rmatvec(rmatvec_in){};
  LinearOperator operator+(const LinearOperator &obj) const;
  LinearOperator operator-(const LinearOperator &obj) const;
  LinearOperator operator*(const LinearOperator &obj) const;
  LinearOperator transpose() const {
    return LinearOperator(n, m, rmatvec, matvec);
  }

  Vector apply_matvec(const Vector &x) const { return matvec(x); }
  Vector apply_rmatvec(const Vector &x) const { return rmatvec(x); }
};

LinearOperator block_diag(const std::vector<LinearOperator> &linear_operators);
LinearOperator aslinearoperator(const Matrix &A);
LinearOperator aslinearoperator(const SparseMatrix &A);
LinearOperator zero(int m, int n);
LinearOperator identity(int n);
LinearOperator diag(const Array &coefficients);
LinearOperator scalar(double x);

// Scalar multiplication for operator.
LinearOperator scalar_mult(const double &y, const LinearOperator &obj);

// Modifies op1 to become y*op1 - op2
void scalar_mult_and_subtraction_in_place(const double &y, LinearOperator &op1,
                                          const LinearOperator &op2);

// Modifies op2 to become op1(op2(x)) + op3(x).
void mult_of_op_and_addition(const LinearOperator &op1, LinearOperator &op2, 
                             const LinearOperator &op3);

/* Modifies op to become the operator y*op - scal*e^T. */
void DN_operation(const double &y, LinearOperator &op, const Vector &scal, 
                  const int &N);

/* Modifies op to become (Q-I)*op + I*/
void DR_operation(const LinearOperator &Q, LinearOperator &op);