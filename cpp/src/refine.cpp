
#include "refine.h"
#include "lsqr.h"
#include "deriv.h"
#include <numeric>     
#include "linop.h"
#include <iostream>    // Remove this later.

/** Construct the cache for the exponential cones. */
Vector construct_q_cache(const std::vector<Cone> &cones){
    int q_cache_size = 0;         
    for (const Cone &cone : cones) {
        const ConeType &type = cone.type;
        const std::vector<int> &sizes = cone.sizes;
        if(type == SOC){
           q_cache_size = std::accumulate(sizes.begin(), sizes.end(), 0);
        } 
    }
    return Vector(q_cache_size);  
}

struct PSD_cache
{
    Vector cache_evals;
    Vector cache_evecs; 
};

/** Construct the cache for the PSD cones.*/
PSD_cache construct_s_cache(const std::vector<Cone> &cones){
    int cache_eval_size = 0;
    int cache_evec_size = 0;
    for (const Cone &cone : cones) {
        const ConeType &type = cone.type;
        const std::vector<int> &sizes = cone.sizes;
        if(type == PSD){
           for (int size : sizes){
               cache_eval_size += size;
               cache_evec_size += size*size;
           }
        } 
    }
    return PSD_cache{Vector(cache_eval_size), 
                     Vector(cache_evec_size)};
}

/** Construct the cache for the exponential cone or its dual.*/
Vector construct_exp_pri_or_dual_cache(const std::vector<Cone> &cones,
                                       const ConeType _conetype){
    int size = 0;
    for (const Cone &cone : cones) {
        const ConeType &type = cone.type;
        if(type == _conetype){
           size = cone.sizes[0];
           }
        } 
    return Vector::Zero(3*size);   
}

Vector refine(const SparseMatrix &A, const Vector &b, const Vector &c,
              const std::vector<Cone> &cones, Vector &z, int n, int m,
              int ref_iter, int lsqr_iter, bool verbose) {

    // Construct caches.
    Vector q_cache = construct_q_cache(cones);
    PSD_cache _PSD_cache = construct_s_cache(cones);
    Vector cache_eval = _PSD_cache.cache_evals;
    Vector cache_evec = _PSD_cache.cache_evecs;
    Vector ep_cache = construct_exp_pri_or_dual_cache(cones, EXP);
    Vector ed_cache = construct_exp_pri_or_dual_cache(cones, EXP_DUAL);

    // Construct operator representing the skew-symmetric matrix Q.
    LinearOperator Q_op = Q_operator(A, b, c, n, m);

    Vector residual = residual_map(Q_op, cones, z, q_cache, cache_eval,
                                   cache_evec, ep_cache, ed_cache, n, m);

    
    double normres = residual.norm()/std::abs(z[n+m]);
    if(verbose){
        std::cout << "Initial norm of N(z): " << normres << std::endl;
    }    
    double refined_normres = normres;
    // Refinement steps.
    double old_normres;
    Vector refined = z;
    Vector new_z(n+m+1);
    for (int iter = 0; iter < ref_iter; iter++) {
        
        
        LinearOperator DN =
          DN_operator_optimized_memory(Q_op, cones, z.segment(0, n), 
                                       z.segment(n, m), z[n+m], residual,       
                                       q_cache, cache_eval, cache_evec,
                                       ep_cache, ed_cache);

        Vector rhs = residual/std::abs(z[n+m]);
        LsqrResult result = lsqr(DN, rhs, 1e-8, 1e-8, 1e-8, 1e8, lsqr_iter);
        Vector dir = result.x;
        
        // Backtrack. Note the extra minus.
        old_normres = normres;
        for (int j = 0; j < 10; j++) {
            new_z = z - std::pow(2, -j) * dir;
            residual = residual_map(Q_op, cones, new_z, q_cache, cache_eval, 
                                             cache_evec, ep_cache, ed_cache, n, m);
            normres = residual.norm()/std::abs(new_z(n+m));
            
            if(normres < old_normres) {
                z = new_z;
                break; 
            }
        }
        if(normres < refined_normres){
            refined_normres = normres;
            refined = z/std::abs(z(n+m));                            
        }
        if(verbose){
            std::cout << "Norm of N(z) after refinement step: " << normres << std::endl;
        }
    };
    return refined;      
}

