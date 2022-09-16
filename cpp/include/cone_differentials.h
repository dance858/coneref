#pragma once
#include <vector>
#include "eigen_includes.h"
#include "linop.h"
#include "utils.h"

/**
 * Computes the differential of the operator that projects onto the second-order
 * cone.
 * @param[in] x The point to compute the differential at.
 * @param[in] q_cache Stores the projection to the second-order cone. For more 
 *                    details, see the function "SOC_Pi". 
 * 
 * @return A linear operator representing the differential at x.
 * 
 * @note The cache must be full before this function is called.
*/
LinearOperator SOC_Pi_diff(const Vector &z, const Vector &q_cache);

/**
 * Computes the differential of the operator that projects onto the positive
 * semidefinite cone.
 * @param[in] x The point to compute the differential at.
 * @param[in] cache_eval Vector that stores eigenvalues.
 * @param[in] cache_evec Vector that stores eigenvectors (note that 
 *                       the eigenvectors are stored in vectorized format).
 * @return A linear operator representing the differential at x.
 * 
 * @note The cache must be full before this function is called.
*/
LinearOperator SDP_Pi_diff(const Vector &x, const Vector &cache_eval, const Vector &cache_evec);


/**
 * Computes the differential of the operator that projects onto the primal
 * exponential cone.
 * @param[in] x The point to compute the differential at.
 * @param[in] ep_cache Container that stores three quantities:
 *                     proj(x)[0], proj(x)[1] and the optimal multiplier.
 * @return A linear operator representing the differential at x.
 * 
 * @note The cache must be full before this function is called.
*/                                 
LinearOperator exp_primal_Pi_diff(const Eigen::Vector3d &x,
                                  const Eigen::Vector3d &ep_cache);

LinearOperator exp_dual_Pi_diff(const Eigen::Vector3d &x, 
                                const Eigen::Vector3d &ed_cache);



/**
 * Computes the differential of the operator that projects onto the cartesian
 * product of the cones specified by the argument 'cones'.
 * @param[in] x The point to compute the differential at.
 * @param[in] caches Cached information. 
 * @return A linear operator representing the differential at x.
 * 
 * @note  The cache must be full before this function is called.
*/
LinearOperator prod_cone_Pi_diff(const Vector &x, const std::vector<Cone> &cones,
                                 const Vector &q_cache, const Vector &cache_evals, 
                                 const Vector &cache_evecs, const Vector &ep_cache, 
                                 const Vector &ed_cache);