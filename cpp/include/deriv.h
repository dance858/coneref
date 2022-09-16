#pragma once
#include "eigen_includes.h"
#include "linop.h"
#include <vector>
#include "utils.h"

/** 
 * Returns the differential of the residual map, equation 9 in the refinement paper.
 *
 * @param[in] u z[0:n]
 * @param[in] v z[n:n+m-1]
 * @param[in] w z[n+m]
 * @param[in] q_cache Full cache for the projection onto the second-order cones.
 * 
 * @return An abstract linear operator representing the differential of the residual map
 *         at the point z = [u, v, w].
 * 
 * @note
 *  The cache must be full before this function is called. 
 */
LinearOperator DR_operator(const LinearOperator &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w, 
                          const Vector &q_cache, const Vector &cache_evals, 
                          const Vector &cache_evecs, const Vector &ep_cache, 
                          const Vector &ed_cache);

/**
 * Same as DR_operator but much less overhead. Very significant difference for
 * small problems.
 */
LinearOperator DR_operator_memory_optimized
                         (const LinearOperator &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w, 
                          const Vector &q_cache, const Vector &cache_evals, 
                          const Vector &cache_evecs, const Vector &ep_cache, 
                          const Vector &ed_cache);

/** Differential of normalized residual map. 
 * @param[in] u z[0:n]
 * @param[in] v z[n:n+m-1]
 * @param[in] w z[n+m]
 * @param[in] residual R(z) 
 * @param[in] q_cache Full cache for the projection onto the second-order cones.
 * @return An abstract linear operator representing the differential of 
 *         the normalized residual map at the point z = [u, v, w].
 * 
 * @note
 *  The cache must be full before this function is called. 
 */
LinearOperator DN_operator(const LinearOperator &Q, const std::vector<Cone> &cones,
                          const Vector &u, const Vector &v, double w,
                          const Vector &residual, const Vector &q_cache, 
                          const Vector &cache_evals, const Vector &cache_evecs,
                          const Vector &ep_cache, const Vector &ed_cache);

/**
 * Same as "DN_operator" but much less overhead. Very significant difference for
 * small problems.
 */
LinearOperator DN_operator_optimized_memory(const LinearOperator &Q,
                          const std::vector<Cone> &cones, const Vector &u, 
                          const Vector &v, double w, const Vector &residual, 
                          const Vector &q_cache, const Vector &cache_evals,
                          const Vector &cache_evecs, const Vector &ep_cache, 
                          const Vector &ed_cache);

/** Returns a linear operator representing the skew-symmetric matrix. */                    
LinearOperator Q_operator(const SparseMatrix &A, const Vector &b, const Vector &c,
                          int n, int m);

/** Evaluates the residual map R(z), using eq. 7 of refinement paper.
 * @param[in] Q A liner operator representing the skew-symmetric matrix of the 
 *              embedding.
 * @param[in] n The number of constraints in the primal conic LP, ie. the number 
 *              of rows of the constraint matrix A.
 * @param[in] m The number of x-variables in the primal conic LP, ie. the number
 *              of columns of the constraint matrix A.
 * 
 * @return The vector R(z).
 * 
 * @note This function modifies the caches.
 *  
*/
Vector residual_map(const LinearOperator &Q, const std::vector<Cone> &cones,
                    const Vector &z, Vector &q_cache, Vector &eval_cache, 
                    Vector &evec_cache, Vector &ep_cache, Vector &ed_cache, 
                    int n, int m);

// For making the visualization of the performance of the refinement procedure 
// easier.
Vector residual_map_python_friendly(const SparseMatrix &A, const Vector &b,
                                    const Vector &c, const std::vector<Cone> &cones,
                                    const Vector &z, Vector &q_cache, 
                                    Vector &eval_cache, Vector &evec_cache,
                                    Vector &ep_cache, Vector &ed_cache, int n,
                                    int m);
