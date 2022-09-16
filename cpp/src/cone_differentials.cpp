#include <numeric>
#include "cone_differentials.h"


LinearOperator exp_primal_Pi_diff(const Eigen::Vector3d &x,
                                  const Eigen::Vector3d &ep_cache) {
  
    // Case 1.
    if (in_exp(x)) {
      return identity(3);
    } 
    // Case 2. Note the minus. We also add an extra check for nondifferentiability.
    // If x[1] == 0, then we return the zero operator. This is a heuristic choice.
    else if (in_exp_dual(-x)) {
      return zero(3, 3);
    }
    // Case 3.
    else if (x[0] < 0 && x[1] < 0) {
      const VecFn matvec = [x](const Vector &y) -> Vector {
        Eigen::Vector3d out;
        double last_component = 0;
        if (x[2] >= 0) {
          last_component = y[2];
        }
        out << y[0], 0, last_component;
        return out;
      };
      return LinearOperator(3, 3, matvec, matvec);
    }
    // Case 4. 
    else { 
      double _x = ep_cache[0];
      double _y = ep_cache[1];

      // Heuristic choice in case of non-differentiability.
      if(_y == 0){
        return zero(3, 3);
      }

      double mu = ep_cache[2]; 
      double alpha = std::exp(_x / _y);
      double beta = mu * _x / (_y * _y) * alpha;

      Eigen::Matrix<double, 4, 4> J_inv;                                              
      J_inv << alpha, (-_x + _y) / _y * alpha, -1, 0,                      // Row 4 in refinement paper.
               1 + mu / _y * alpha, -beta, 0, alpha,                       // Row 1 in refinement paper. 
               -beta, 1 + beta * _x / _y, 0, (1 - _x / _y) * alpha,        // Row 2 in refinement paper. 
               0, 0, 1, -1;                                                // Row 3 in refinement paper.
      
      // extract a 3x3 subblock, with top-left corner at row 0, column 1
      const Matrix J = J_inv.inverse().block<3, 3>(0, 1);
      return aslinearoperator(J);
    }
  }

LinearOperator exp_dual_Pi_diff(const Eigen::Vector3d &x, 
                                const Eigen::Vector3d &ed_cache){
   return identity(3) - exp_primal_Pi_diff(-x, ed_cache);
}

LinearOperator SDP_Pi_diff(const Vector &x, const Vector &cache_eval, 
                           const Vector &cache_evec){

  // Represent the eigenvectors in matrix form.
  int num_of_evals = cache_eval.size();
  Matrix Q = cache_evec.reshaped(num_of_evals, num_of_evals);
  
  // Check if all eigenvalues are >= 0.
  if (cache_eval[0] >= 0) {
    return identity(x.size());
  }

  // k is the number of negative eigenvalues in X minus ONE
  int k = -1;
  for (int i = 0; i < num_of_evals; ++i) {
    if (cache_eval[i] < 0) {
      k += 1;
    } else {
      break;
    }
  }

  // Define the differential.
  const VecFn matvec = [cache_eval, Q, k](const Vector &y) -> Vector {
    Matrix tmp = Q.transpose() * matrix_from_lower_triangular(y) * Q;
    // Componentwise multiplication by the matrix `B` from refinement paper.
    for (int i = 0; i < tmp.rows(); ++i) {
      for (int j = 0; j < tmp.cols(); ++j) {
        if (i <= k && j <= k) {
          tmp(i, j) = 0;
        } else if (i > k && j <= k) {
          double lambda_i_pos = std::max(cache_eval[i], 0.0);
          double lambda_j_neg = -std::min(cache_eval[j], 0.0);
          tmp(i, j) *= lambda_i_pos / (lambda_j_neg + lambda_i_pos);
        } else if (i <= k && j > k) {
          double lambda_i_neg = -std::min(cache_eval[i], 0.0);
          double lambda_j_pos = std::max(cache_eval[j], 0.0);
          tmp(i, j) *= lambda_j_pos / (lambda_i_neg + lambda_j_pos);
        }
      }
    }
    Matrix result = Q * tmp * Q.transpose();
    return lower_triangular_from_matrix(result);
  };
  return LinearOperator(x.size(), x.size(), matvec, matvec);
}


LinearOperator SOC_Pi_diff(const Vector &z, const Vector &q_cache) {
  // Point at which we differentiate.
  int n = z.size();
  const double t = z[0];
  const Vector &x = z.segment(1, n - 1);
  const double norm_x = x.norm();

  // Simple cases.
  if (norm_x <= t) {
    return identity(n);
  } else if (norm_x <= -t) {
    return zero(n, n);
  } else {
  // Use cached information.
    const double s = q_cache[0];
    const Vector &y = q_cache.segment(1, n - 1);        

    const VecFn matvec = [s, y, x, t, n](const Vector &dz) -> Vector {
      double dt = dz[0];
      const Vector &dx = dz.segment(1, n - 1);
      double alpha = 2*s - t;
      if (alpha == 0){
        throw std::runtime_error("Second order cone derivative error.");
      }
    
      const Vector b = 2 * y - x;
      const double c = dt * s + dx.dot(y);
      const Vector d = dt * y + s*dx;
      const double denom = (alpha - b.squaredNorm()/alpha);
      if (denom == 0){
        throw std::runtime_error("Second order cone derivative error.");
      }

      const double ds = (c - b.dot(d) / alpha) / denom;
      const Vector dy = (d - ds * b) / alpha;

      Vector result(n);
      result << ds, dy;
      return result;
    };
    return LinearOperator(n, n, matvec, matvec);
  }
}

LinearOperator _dprojection_pos(const Vector &x) {
  const Array sign = x.cwiseSign();
  return diag(0.5 * (sign + 1));
}

LinearOperator _dprojection_zero(const Vector &x, bool dual) {
  int n = x.size();
  return dual ? identity(n) : zero(n, n);
}

/**
 * TODO: (dance858) At the moment many temporary vectors representing
 *       temporary caches are created. This can refactored, but profiling 
 *       shows that it is just microoptimization.
*/
LinearOperator prod_cone_Pi_diff(const Vector &x, const std::vector<Cone> &cones,
                                 const Vector &q_cache, const Vector &cache_evals, 
                                 const Vector &cache_evecs, const Vector &ep_cache, 
                                 const Vector &ed_cache){
  
  std::vector<LinearOperator> lin_ops;

  // Offset used for parsing x.
  int offset = 0;
  
  for (const Cone &cone : cones) {
    const ConeType &type = cone.type;
    const std::vector<int> &sizes = cone.sizes;
    if (std::accumulate(sizes.begin(), sizes.end(), 0) == 0) {
      continue;
    }

    if (type == ZERO){ 
      lin_ops.emplace_back(_dprojection_zero(x.segment(offset, sizes[0]), false));                
      offset += sizes[0];
      
    } else if (type == POS){
        lin_ops.emplace_back(_dprojection_pos(x.segment(offset, sizes[0])));
        offset += sizes[0];
    } else if (type == SOC){
      int second_order_cache_offset = 0;
      for (int size : sizes){
        lin_ops.emplace_back(SOC_Pi_diff(x.segment(offset, size), 
                               q_cache.segment(second_order_cache_offset, size)));
        offset += size;
        second_order_cache_offset += size;
      }
    } else if (type == PSD){
      int eval_cache_offset = 0;
      int evec_cache_offset = 0;
      int vectorized_size;
      for(int size : sizes){
        vectorized_size = vectorized_psd_size(size);
        lin_ops.emplace_back(SDP_Pi_diff(x.segment(offset, vectorized_size), 
                              cache_evals.segment(eval_cache_offset, size), 
                              cache_evecs.segment(evec_cache_offset, size*size)));
        offset += vectorized_size;
        eval_cache_offset += size;
        evec_cache_offset += size*size;
      }
    } else if (type == EXP){
      int ep_cache_offset = 0;
      for (int i = 0; i < sizes[0]; i++){
        lin_ops.emplace_back(exp_primal_Pi_diff(x.segment(offset, 3), 
                               ep_cache.segment(ep_cache_offset, 3)));
        offset += 3;
        ep_cache_offset += 3;
      }
    } else if (type == EXP_DUAL) {
      int ed_cache_offset = 0;
      for (int i = 0; i < sizes[0]; i++){
          lin_ops.emplace_back(exp_dual_Pi_diff(x.segment(offset, 3), 
                               ed_cache.segment(ed_cache_offset, 3)));
          offset += 3;
          ed_cache_offset += 3;
       } 
    } else{
       throw std::invalid_argument("Unknown cone");
    }
  }
  
  return block_diag(lin_ops);
}