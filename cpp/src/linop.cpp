#include "linop.h"
#include <assert.h>

LinearOperator LinearOperator::operator+(const LinearOperator &obj) const {
  assert(m == obj.m);
  assert(n == obj.n);

  const LinearOperator this_op = *this;
  const VecFn result_matvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.matvec(x) + obj.matvec(x);
  };
  const VecFn result_rmatvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.rmatvec(x) + obj.rmatvec(x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator-(const LinearOperator &obj) const {
  assert(m == obj.m);
  assert(n == obj.n);
  
  const LinearOperator this_op = *this;
  const VecFn result_matvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.matvec(x) + obj.matvec(-x);
  };
  const VecFn result_rmatvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.rmatvec(x) + obj.rmatvec(-x);
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}

LinearOperator LinearOperator::operator*(const LinearOperator &obj) const {
  assert(n == obj.m);

  const LinearOperator this_op = *this;
  const VecFn result_matvec = [this_op, obj](const Vector &x) -> Vector {
    return this_op.matvec(obj.matvec(x));
  };
  const VecFn result_rmatvec = [this_op, obj](const Vector &x) -> Vector {
    return obj.rmatvec(this_op.rmatvec(x));
  };
  return LinearOperator(m, obj.n, result_matvec, result_rmatvec);
}

LinearOperator block_diag(const std::vector<LinearOperator> &linear_operators) {
  assert(linear_operators.size() > 0);

  int rows = 0;
  int cols = 0;

  for (const LinearOperator &linop : linear_operators) {
    rows += linop.m;
    cols += linop.n;
  }

  const VecFn result_matvec = [linear_operators, rows,
                               cols](const Vector &x) -> Vector {
    assert(x.size() == cols);
    Vector out = Vector::Zero(rows);
    int i = 0;
    int j = 0;
    for (const LinearOperator &linop : linear_operators) {
      out.segment(i, linop.m) = linop.matvec(x.segment(j, linop.n));
      i += linop.m;
      j += linop.n;
    }
    return out;
  };
  const VecFn result_rmatvec = [linear_operators, rows,
                                cols](const Vector &x) -> Vector {
    assert(x.size() == rows);
    Vector out = Vector::Zero(cols);
    int i = 0;
    int j = 0;
    for (const LinearOperator &linop : linear_operators) {
      out.segment(i, linop.n) = linop.rmatvec(x.segment(j, linop.m));
      i += linop.n;
      j += linop.m;
    }
    return out;
  };

  return LinearOperator(rows, cols, result_matvec, result_rmatvec);
}

LinearOperator aslinearoperator(const Matrix &A) {
  const VecFn result_matvec = [A](const Vector &x) -> Vector { return A * x; };
  const VecFn result_rmatvec = [A](const Vector &x) -> Vector {
    return A.transpose() * x;
  };
  return LinearOperator(A.rows(), A.cols(), result_matvec, result_rmatvec);
}

LinearOperator aslinearoperator(const SparseMatrix &A) {
  const VecFn result_matvec = [A](const Vector &x) -> Vector { return A * x; };
  const VecFn result_rmatvec = [A](const Vector &x) -> Vector {
    return A.transpose() * x;
  };
  return LinearOperator(A.rows(), A.cols(), result_matvec, result_rmatvec);
}

LinearOperator zero(int m, int n) {
  const VecFn matvec = [](const Vector &x) -> Vector {
    return Vector::Zero(x.size());
  };
  return LinearOperator(m, n, matvec, matvec);
}

LinearOperator identity(int n) {
  const VecFn matvec = [](const Vector &x) -> Vector { return x; };
  return LinearOperator(n, n, matvec, matvec);
}

LinearOperator diag(const Array &coefficients) {
  const VecFn matvec = [coefficients](const Vector &x) -> Vector {
    return (coefficients * x.array()).matrix();
  };
  return LinearOperator(coefficients.size(), coefficients.size(), matvec,
                        matvec);
}

LinearOperator scalar(double num) {
  const VecFn matvec = [num](const Vector &x) -> Vector {
    Vector result = Vector::Zero(1);
    result[0] = num * x[0];
    return result;
  };
  return LinearOperator(1, 1, matvec, matvec);
}

/** Multiplies an operator by a scalar in the sense that the result from the
 *  operators matvec/rmatvec is multiplied by the scalar.
 */ 
LinearOperator scalar_mult(const double &y, const LinearOperator &obj) {

  const VecFn result_matvec = [y, obj](const Vector &x) -> Vector {
    return y*obj.matvec(x);
  };

   const VecFn result_rmatvec = [y, obj](const Vector &x) -> Vector {
    return y*obj.rmatvec(x);
  };

  return LinearOperator(obj.m, obj.n, result_matvec, result_rmatvec);
}

/** Modifies op1 to become y*op1(x) - op2(x)  */
void scalar_mult_and_subtraction_in_place(const double &y, LinearOperator &op1, 
                                          const LinearOperator &op2) {
  assert(op1.m == op2.m);
  assert(op1.n == op2.n);

  const VecFn result_matvec = [y, op1, op2](const Vector &x) -> Vector {
    return y*op1.matvec(x) - op2.matvec(x);
  };

   const VecFn result_rmatvec = [y, op1, op2](const Vector &x) -> Vector {
    return y*op1.rmatvec(x) - op2.rmatvec(x);
  };

  op1.matvec = result_matvec;
  op1.rmatvec = result_rmatvec;
}


/*  Modifies op2 to become the operator op1(op2(x)) + op3(x).*/
void mult_of_op_and_addition(const LinearOperator &op1, LinearOperator &op2, 
                             const LinearOperator &op3){
  assert(op1.n == op2.m);
  assert(op1.m == op3.m);

  const VecFn result_matvec = [op1, op2, op3](const Vector &x) -> Vector {
    return op1.matvec(op2.matvec(x)) + op3.matvec(x);
  };
  const VecFn result_rmatvec = [op1, op2, op3](const Vector &x) -> Vector {
    // Note order of op1 and op2. Classic transpose rule. Thank god I took that 
    // linear algebra class.
    return op2.rmatvec(op1.rmatvec(x)) + op3.rmatvec(x);  
  };
  
  op2.matvec = result_matvec;
  op2.rmatvec = result_rmatvec;
}

/* Modifies op to become the operator y*op - scal*e^T. */
void DN_operation(const double &y, LinearOperator &op, const Vector &scal, 
                  const int &N){
  
  const VecFn result_matvec = [y, op, scal, N](const Vector &x) -> Vector {
    return y*op.matvec(x) - x[N-1]*scal;
  };

  Vector e = Vector::Zero(N);
  e[N-1] = 1;                       
   const VecFn result_rmatvec = [y, op, scal, e](const Vector &x) -> Vector {
    return y*op.rmatvec(x) - scal.dot(x)*e;
  };

  op.matvec = result_matvec;
  op.rmatvec = result_rmatvec;
}

/* Modifies op to become (Q-I)*op + I*/
void DR_operation(const LinearOperator &Q, LinearOperator &op){
  
  const VecFn result_matvec = [Q, op](const Vector &x) -> Vector {
    Vector proj = op.matvec(x);
    return Q.matvec(proj) - proj + x;
  };
                       
   const VecFn result_rmatvec = [Q, op](const Vector &x) -> Vector {
    return op.rmatvec(Q.rmatvec(x) - x) + x;
  };
  
  op.matvec = result_matvec;
  op.rmatvec = result_rmatvec;
}
