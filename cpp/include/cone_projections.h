#pragma once
#include <vector>
#include "eigen_includes.h"
#include "utils.h"


/**
 * Project onto second order cone.
 * @param[in] x The point to project onto the SOC.
 * @param[in] q_cache Container for storing the projection. See cpp file for 
 *                    more details.
 * @return The projection of x onto the SOC.
 */
Vector SOC_Pi(const Vector &x, Vector &q_cache);

/**
 * Project onto positive semidefinite cone.
 *
 * @param[in] x              The "matrix" to project. A symmetric matrix of size 
 *                           n x n is represented by a Vector of size n(n+1)/2. 
 *                           The off-diagonal elements have been scaled by sqrt(2).
 * @param[in] cones          Information about the cones in the problem.
 * @param[in] cache_evals    Container in which to cache the eigenvalues for the
 *                           projections onto the positive semidefinite cones.
 * @param[in] cache_evecs    Container in which to cache eigenvectors for the 
 *                           projections onto the positive semidefinite cones.           
 * @return The projection onto the positive semidefinite cone, represented by a Vector.
 */
Vector PSD_Pi(const Vector &x, Vector &cache_evals, Vector &cache_evecs);

/**
 * Computes the projection onto the primal exponential cone.
 * @param[in] x        The point to project.
 * @param[in] ep_cache A container to store information in. The cache stores 
 *                     the two first components of the projection (denoted by
 *                     x and y in the refinement paper), as well as the optimal
 *                     multiplier (denoted by mu in the refinement paper).
 * 
*/
Eigen::Vector3d exp_primal_Pi(const Eigen::Vector3d &x, Eigen::Vector3d &ep_cache);
Eigen::Vector3d exp_dual_Pi(const Eigen::Vector3d &x, Eigen::Vector3d &ed_cache);

/**
 * Project onto cartesian product of cones.
 *
 * @param[in] x              The point to project.
 * @param[in] cones          Information about the cones in the problem.
 * @param[in] q_cache        Container in which to cache the projection onto the 
 *                           second order cone.
 * @param[in] cache_evals    Container in which to cache the eigenvalues for the
 *                           projections onto the positive semidefinite cones.
 * @param[in] cache_evecs    Container in which to cache eigenvectors for the 
 *                           projections onto the positive semidefinite cones.
 * @param[in] ep_cache       Container of size number_of_primal_exponential_cones * 3.
 *                           For each exponential cone we cache the two first components
 *                           of the projection as well as the optimal multiplier.
 * @param[in] ed_cache       Cache for dual exponential cone.   
 * @return The projection of x onto the cartesian product.
 * 
 */
Vector prod_cone_Pi(const Vector &x, const std::vector<Cone> &cones, 
                    Vector &q_cache,  Vector &cache_evals, Vector &cache_evecs,
                    Vector &ep_cache, Vector &ed_cache);          
                    
/**
 * Project onto embedded cone.
 *
 * @param[in] x              The point to project. Should have size m+n+1.
 * @param[in] cones          Information about the cones in the problem.
 * @param[in] q_cache        Cache for second-order cones.
 * @param[in] cache_evals    Container in which to cache the eigenvalues for the
 *                           projections onto the positive semidefinite cones.
 * @param[in] cache_evecs    Container in which to cache eigenvectors for the 
 *                           projections onto the positive semidefinite cones.           
 * @return The projection onto the positive semidefinite cone, represented by a Vector.
 */      
Vector embedded_cone_Pi(const Vector &x, const std::vector<Cone> &cones,
                        Vector &q_cache, Vector &eval_cache, Vector &evec_cache,
                        Vector &ep_cache, Vector &ed_cache, int n, int m);
