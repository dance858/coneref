#include <algorithm>
#include <numeric>
#include "cone_projections.h"
#include "linop.h"
#include "utils.h"

double exp_newton_one_d(double rho, double y_hat, double z_hat) {
  double t = std::max(-z_hat, 1e-6);
  double f, fp;
  int i;
  for (i = 0; i < EXP_CONE_MAX_ITERS; ++i) {
    f = t * (t + z_hat) / rho / rho - y_hat / rho + log(t / rho) + 1;
    fp = (2 * t + z_hat) / rho / rho + 1 / t;

    t = t - f / fp;

    if (t <= -z_hat) {
      return 0;
    } else if (t <= 0) {
      return z_hat;
    } else if (std::abs(f) < CONE_TOL) {
      break;
    }
  }
  return t + z_hat;
}

void exp_solve_for_x_with_rho(double *v, double *x, double rho) {
  x[2] = exp_newton_one_d(rho, v[1], v[2]);
  x[1] = (x[2] - v[2]) * x[2] / rho;
  x[0] = v[0] - rho;
}

double exp_calc_grad(double *v, double *x, double rho) {
  exp_solve_for_x_with_rho(v, x, rho);
  if (x[1] <= 1e-12) {
    return x[0];
  }
  return x[0] + x[1] * log(x[1] / x[2]);
}

void exp_get_rho_ub(double *v, double *x, double *ub, double *lb) {
  *lb = 0;
  *ub = 0.125;
  while (exp_calc_grad(v, x, *ub) > 0) {
    *lb = *ub;
    (*ub) *= 2;
  }
}

/* project onto the exponential cone, v has dimension *exactly* 3 */
int _proj_exp_cone(double *v, double *rho) {
  int i;
  double ub, lb, g, x[3];
  double r = v[0], s = v[1], t = v[2];
  double tol = CONE_TOL;

  /* v in cl(Kexp) */
  if ((s * exp(r / s) - t <= CONE_THRESH && s > 0) ||
      (r <= 0 && std::abs(s) <= CONE_THRESH && t >= 0)) {
    return 0;
  }

  /* -v in Kexp^* */
  if ((-r < 0 && r * exp(s / r) + EulerConstant * t <= CONE_THRESH) ||
      (std::abs(r) <= CONE_THRESH && -s >= 0 && -t >= 0)) {
    memset(v, 0, 3 * sizeof(double));
    return 0;
  }

  /* special case with analytical solution */
  if (r < 0 && s < 0) {
    v[1] = 0.0;
    v[2] = std::max(v[2], 0.0);
    return 0;
  }

  /* iterative procedure to find projection, bisects on dual variable: */
  exp_get_rho_ub(v, x, &ub, &lb); /* get starting upper and lower bounds */
  for (i = 0; i < EXP_CONE_MAX_ITERS; ++i) {
    *rho = (ub + lb) / 2;          /* halfway between upper and lower bounds */
    g = exp_calc_grad(v, x, *rho); /* calculates gradient wrt dual var */
    if (g > 0) {
      lb = *rho;
    } else {
      ub = *rho;
    }
    if (ub - lb < tol) {
      break;
    }
  }

  v[0] = x[0];
  v[1] = x[1];
  v[2] = x[2];
  return 0;
}

/**
 * @note This function calls the function '_proj_exp_cone', which modifies
 *       rho. First I thought rho is the dual variable, but the value of rho
 *       returned by '_proj_exp_cone' does not satisfy the KKT-conditions.
 *       However, we can recover the correct multiplier as 
 *       mu = projection[2] - x[2] (see the third equation in eq. 25 in
 *       refinement paper). 
*/
Eigen::Vector3d exp_primal_Pi(const Eigen::Vector3d &x, Eigen::Vector3d &ep_cache){
  double v[3] = {x[0], x[1], x[2]};
  double rho = 0;
  int ret = _proj_exp_cone(v, &rho);
  if (ret != 0) {
    throw std::runtime_error("Projection onto exponential cone failed.");
  }
  Eigen::Vector3d projection;
  projection << v[0], v[1], v[2];               
  ep_cache << v[0], v[1], v[2] - x[2];           

  return projection;
}

Eigen::Vector3d exp_dual_Pi(const Eigen::Vector3d &x, Eigen::Vector3d &ed_cache){
  // Project onto K* using the Moreau decomposition Pi_K*(y) = y + Pi_K(-y).
  return x + exp_primal_Pi(-x, ed_cache);
}


/**
 * @note
 *  What is the purpose of the cache? The main function of this package is the 
 *  refine function. In the refine function we first evaluate the residual R(z).
 *  To evaluate R(z) we compute the projection onto the embedded cone 
 *  R^n x K* x R+. To project onto the dual cone K* inside the function 
 *  "embedded_cone_Pi" we use the Moreau decomposition Proj_K*(z) = z + Proj_K(-z). 
 *  In other words, we project (a part of) the point -z onto K (note the minus).
 *  The SOC-part of the quantity Proj_K(-z) is then stored in q_cache. This 
 *  quantity is reused when the differential of embedded cone is evaluated. In 
 *  other words, the purpose of q_cache is not to store the projection
 *  of a point x, but rather it should store the projection of -x.
 */

Vector SOC_Pi(const Vector &x, Vector &q_cache){
  int n = x.size();  
  const double t = x[0];
  const Vector &z = x.segment(1, n - 1);      
  const double norm_z = z.norm();                             
  
  if (norm_z <= t) {
    q_cache = x;
    return q_cache;
  } else if (norm_z <= -t) {
    q_cache = Vector::Zero(n);
    return q_cache;
  } else {
    q_cache << 1, z/norm_z;
    q_cache = 0.5*(t+norm_z)*q_cache;
    return q_cache;
    };
}

/**
 * @note
 *    1. It should be emphasized that this function modifies the caches.
 *    2. This function is called from project_on_embedded_cone using the Moreau decomposition.
 *       When the function is called this way it should always project on K (and not K*). Hence,
 *       the projection of the zero cones in 'cones' should always be on zero and not the 
 *       free cone. Similarily for the exponential cones.
 * 
 * TODO: (dance858) At the moment many temporary vectors representing
 *       temporary caches are created. This can refactored, but it does not 
 *       incur any significant cost.
 */
Vector prod_cone_Pi(const Vector &x, const std::vector<Cone> &cones, 
                    Vector &q_cache, Vector &cache_evals, Vector &cache_evecs, 
                    Vector &ep_cache, Vector &ed_cache){
  
  // Container for storing the projection.
  Vector result(x.size());

  // Offset used for parsing x.
  int offset = 0;
  
  for (const Cone &cone : cones) {
    const ConeType &type = cone.type;
    const std::vector<int> &sizes = cone.sizes;
    if (std::accumulate(sizes.begin(), sizes.end(), 0) == 0) {
      continue;
    }

    if (type == ZERO){
      result(Eigen::seqN(offset, sizes[0])) = Vector::Zero(sizes[0]);              
      offset += sizes[0];
      
    } else if (type == POS){
        result(Eigen::seqN(offset, sizes[0])) = x.segment(offset, sizes[0]).cwiseMax(0);
        offset += sizes[0];
      
    } else if (type == SOC){
      int second_order_cache_offset = 0;
      for (int size : sizes){
        Vector temp_cache(size);    
        result(Eigen::seqN(offset, size)) = SOC_Pi(x.segment(offset, size), temp_cache);
        q_cache(Eigen::seqN(second_order_cache_offset, size)) = temp_cache;
        offset += size;
        second_order_cache_offset += size;
      }
    } else if (type == PSD){
      int eval_cache_offset = 0;
      int evec_cache_offset = 0;
      int vectorized_size;
      for (int size : sizes){
        vectorized_size = vectorized_psd_size(size);
        Vector temp_eval_cache(size);
        Vector temp_evec_cache(size*size);
        Vector temp_result = PSD_Pi(x.segment(offset, vectorized_size), 
                                            temp_eval_cache, temp_evec_cache);

        result(Eigen::seqN(offset, vectorized_size)) = temp_result;
        cache_evals(Eigen::seqN(eval_cache_offset, size)) = temp_eval_cache;
        cache_evecs(Eigen::seqN(evec_cache_offset, size*size)) = temp_evec_cache;
        
        offset += vectorized_size;
        eval_cache_offset += size;
        evec_cache_offset += size*size;
      }
    } else if (type == EXP){
        int ep_cache_offset = 0;
        for(int i = 0; i < sizes[0]; i++){
          Eigen::Vector3d ep_cache_temp;
          result(Eigen::seqN(offset, 3)) = exp_primal_Pi(x.segment(offset, 3),
                                                         ep_cache_temp);
          ep_cache(Eigen::seqN(ep_cache_offset, 3)) = ep_cache_temp;
          offset += 3;
          ep_cache_offset +=3;
        }
    } else if (type == EXP_DUAL) {
       int ed_cache_offset = 0;
        for(int i = 0; i < sizes[0]; i++){
          Eigen::Vector3d ed_cache_temp;
          result(Eigen::seqN(offset, 3)) = exp_dual_Pi(x.segment(offset, 3), 
                                                       ed_cache_temp);
          ed_cache(Eigen::seqN(ed_cache_offset, 3)) = ed_cache_temp;
          offset += 3;
          ed_cache_offset +=3;
        }
    } else{
       throw std::invalid_argument("Unknown cone");
    }
  }
  return result;
}


Vector PSD_Pi(const Vector &x, Vector &cache_evals, Vector &cache_evecs){

  // Unvectorize x.
  //int n = x.size();
  const Matrix &X = matrix_from_lower_triangular(x);

  // Eigenvalue decomposition.
  Eigen::SelfAdjointEigenSolver<Matrix> eigen_solver(X.rows());
  eigen_solver.compute(X);
  
  // Store quantities in cache.
  cache_evals = eigen_solver.eigenvalues();
  cache_evecs = eigen_solver.eigenvectors().reshaped();
  //std::cout << "Eigenvectors: " << std::endl << cache_evecs << std::endl;

  // Compute the projection.
  Vector pos_evals = cache_evals.cwiseMax(0);
  Matrix proj = eigen_solver.eigenvectors() * 
                (pos_evals.asDiagonal() * eigen_solver.eigenvectors().transpose());


  return lower_triangular_from_matrix(proj);
}

/**
 * @note Observe that the projection of -x.segment(n, m) is stored in the caches, and 
 *       not the projection of x.segment(n, m). The reason is that we  base the 
 *       differentation in the function "_differential_proj_embedded_cone" on the 
 *       Moreau decomposition. 
 *       
 */
Vector embedded_cone_Pi(const Vector &x, const std::vector<Cone> &cones,
                        Vector &q_cache, Vector &eval_cache, Vector &evec_cache,
                        Vector &ep_cache, Vector &ed_cache, int n, int m){

  // Container for storing the projection onto the embedded cone.
  Vector projection(x.size());    

  // Project onto R^n.
  projection(Eigen::seqN(0, n)) = x.segment(0, n);

  // Project onto K* using the Moreau decomposition Pi_K*(y) = y + Pi_K(-y).
  projection(Eigen::seqN(n, m)) = x.segment(n, m) +
                        prod_cone_Pi(-x.segment(n, m), cones, q_cache,
                                     eval_cache, evec_cache, ep_cache, ed_cache);
  
  // Project onto R+. 
  projection(n+m) = std::max(x(n+m), 0.0);

  return projection;
}
