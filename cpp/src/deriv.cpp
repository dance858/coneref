#include "deriv.h"
#include "cone_differentials.h"
#include "cone_projections.h"

// Greater or equal to function.
inline double gt(double x, double t) {
  if (x >= t) {
    return 1.0;
  } else {
    return 0.0;
  }
}

// Sign function 
int _sign(float x) {
    return (x > 0) - (x < 0);
}

/** Returns the differential of the operator that projects onto the embedded
 *  cone.
 *
 * @param[in] u z[0:n]
 * @param[in] v z[n:n+m-1]
 * @param[in] w z[n+m]
 *        
 * @return A linear operator representing the differential.
 * 
 * TODO: (dance858) Subtracting prod_cone_Pi_diff from identity can be refactored. 
 *                  Profile before refactor.        
 */
LinearOperator dpi_with_cache_support(const Vector &u, const Vector &v, double w,
                   const std::vector<Cone> &cones, const Vector &q_cache, 
                   const Vector &cache_evals, const Vector &cache_evecs,
                   const Vector &ep_cache, const Vector &ed_cache){
  
  LinearOperator eye = identity(u.size());
 
  LinearOperator D_proj = identity(v.size()) - 
                          prod_cone_Pi_diff(-v, cones, q_cache,
                                                    cache_evals, cache_evecs,
                                                    ep_cache, ed_cache);
  
  LinearOperator last = scalar(gt(w, 0.0));

  std::vector<LinearOperator> linops{eye, D_proj, last};

  return block_diag(linops);
}


LinearOperator DR_operator(const LinearOperator &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w, 
                          const Vector &q_cache, const Vector &cache_evals, 
                          const Vector &cache_evecs, const Vector &ep_cache, 
                          const Vector &ed_cache) {
  int N = u.size() + v.size() + 1;

  return (Q - identity(N)) * 
         dpi_with_cache_support(u, v, w, cones, q_cache, cache_evals,
                                cache_evecs, ep_cache, ed_cache) +
         identity(N);
}

LinearOperator DR_operator_memory_optimized
                         (const LinearOperator &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w, 
                          const Vector &q_cache, const Vector &cache_evals, 
                          const Vector &cache_evecs, const Vector &ep_cache, 
                          const Vector &ed_cache) {

  //int N = u.size() + v.size() + 1;

  LinearOperator DR = dpi_with_cache_support(u, v, w, cones, q_cache, cache_evals,
                                             cache_evecs, ep_cache, ed_cache);
  // Modify DR to become (Q-I)*DR + I.
  DR_operation(Q, DR);
  return DR;
}

/** To simplify the construction of DN. This function
 *  forms the second term in the expression for DN, see conic refinement paper. 
 *  m is number of rows of operator.
 *  n is number of columns of operator.
 * 
 * @note This function is used in the function "DN_operator", but not in the more
 *       efficient function "DN_operator_optimized_memory."
 */
LinearOperator vector_scaled_by_last_component_of_other_vector(const Vector &y, 
                                                               int m, int n){
  assert(y.size() == m);

  const VecFn result_matvec = [y, n](const Vector &x) -> Vector {
    return x[n-1]*y;
  };

  Vector e = Vector::Zero(n);
  e[n-1] = 1; 
   const VecFn result_rmatvec = [y, n, e](const Vector &x) -> Vector {
    return y.dot(x)*e;
  };
  return LinearOperator(m, n, result_matvec, result_rmatvec);
}



LinearOperator DN_operator(const LinearOperator &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w,
                          const Vector &residual, const Vector &q_cache, 
                          const Vector &cache_evals, const Vector &cache_evecs,
                          const Vector &ep_cache, const Vector &ed_cache) {
  if(w == 0){
    throw std::runtime_error("Construction of DN(z) failed. Prevented division by zero.");
  }
  int N = residual.size();

  return scalar_mult(1/std::abs(w),
                     DR_operator(Q, cones, u, v, w, q_cache, cache_evals, 
                                 cache_evecs, ep_cache, ed_cache)) - 
  vector_scaled_by_last_component_of_other_vector(_sign(w)/(w*w)*residual, N, N);
}

LinearOperator DN_operator_optimized_memory(const LinearOperator &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w,
                          const Vector &residual, const Vector &q_cache, 
                          const Vector &cache_evals, const Vector &cache_evecs,
                          const Vector &ep_cache, const Vector &ed_cache) {
  if(w == 0){
    throw std::runtime_error("Construction of DN(z) failed. Prevented division by zero.");
  }
  
  int N = residual.size();

  LinearOperator DN =
    DR_operator_memory_optimized(Q, cones, u, v, w, q_cache, cache_evals,
                                 cache_evecs, ep_cache, ed_cache);
  DN_operation(1/std::abs(w), DN, _sign(w)/(w*w)*residual, N);
  return DN;
}

LinearOperator Q_operator(const SparseMatrix &A, const Vector &b, const Vector &c,
                          int n, int m){

  const VecFn result_matvec = [A, b, c, n, m](const Vector &z) -> Vector {
    Vector result(n+m+1);
    const Vector &x = z.segment(0, n);
    const Vector &y = z.segment(n, m);
    result(Eigen::seqN(0, n)) = (y.transpose() * A).transpose() + z[n+m]*c;
    result(Eigen::seqN(n, m)) = -A*x + z[n+m]*b; 
    result[n+m] = -(c.dot(x) + b.dot(y));
    return result;
  };

  const VecFn result_rmatvec = [A, b, c, n, m](const Vector &z) -> Vector {
    Vector result(n+m+1);
    const Vector &x = z.segment(0, n);
    const Vector &y = z.segment(n, m);
    result(Eigen::seqN(0, n)) = (y.transpose() * A).transpose() + z[n+m]*c;
    result(Eigen::seqN(n, m)) = -A*x + z[n+m]*b; 
    result[n+m] = -(c.dot(x) + b.dot(y));
    return -result;
  };

  return LinearOperator(m+n+1, m+n+1, result_matvec, result_rmatvec);
} 

Vector residual_map(const LinearOperator &Q, const std::vector<Cone> &cones, 
                             const Vector &z, Vector &q_cache,  Vector &eval_cache,
                             Vector &evec_cache, Vector &ep_cache, Vector &ed_cache,
                             int n, int m){
  Vector u = embedded_cone_Pi(z, cones, q_cache, eval_cache, 
                              evec_cache, ep_cache, ed_cache, n, m);
  //Vector v = u - z;           // For cancellation
  //return Q.matvec(u) - v;
  return Q.matvec(u) - (u-z);
}

// Just for simplfying visualization in Python. 
Vector residual_map_python_friendly(const SparseMatrix &A, const Vector &b,
                                    const Vector &c, const std::vector<Cone> &cones,
                                    const Vector &z, Vector &q_cache, 
                                    Vector &eval_cache, Vector &evec_cache,
                                    Vector &ep_cache, Vector &ed_cache, int n,
                                    int m){
        
  LinearOperator Q = Q_operator(A, b, c, n, m);
  return residual_map(Q, cones, z, q_cache, eval_cache, evec_cache, ep_cache,
                      ed_cache, n, m);
}
