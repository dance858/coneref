import numpy as np
import sys   
import _coneref
from time import time
#import os  For finding process ID for debugging the C++ code.

ZERO = 'z'
POS = "l"
SOC = "q"
PSD = "s"
EXP = "ep"
EXP_DUAL = "ed"

# The ordering of CONES matches SCS.
CONES = [ZERO, POS, SOC, PSD, EXP, EXP_DUAL]

# Map from Python cones to C++ format.
CONE_MAP = {
    'z': _coneref.ConeType.ZERO,
    "l": _coneref.ConeType.POS,
    "q": _coneref.ConeType.SOC,
    "s": _coneref.ConeType.PSD,
    "ep": _coneref.ConeType.EXP,
    "ed": _coneref.ConeType.EXP_DUAL
}

# The following two functions are used to convert a dictionary of cones
# to C++ format.
def parse_cone_dict_cpp(cone_list):
    return [_coneref.Cone(CONE_MAP[cone], [l] if not isinstance(l, (list, tuple)) else l)
            for cone, l in cone_list]
def parse_cone_dict(cone_dict):
    return [(cone, cone_dict[cone]) for cone in CONES if cone in cone_dict]

# Evaluates N(z) defined in eq. 8 of the refinement paper.
def evaluate_normalized_res_map(z, A, b, c, cones):
    _cones = parse_cone_dict(cones)
    _cones_parsed = parse_cone_dict_cpp(_cones)
    m, n = A.shape
    # These caches are not really needed.
    q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
         = make_prod_cone_cache(cones)
    normalized_res = _coneref.residual_map_python_friendly(A, b, c, _cones_parsed,
                      z, q_cache, eval_cache, evec_cache, ep_cache, ed_cache, n, m)

    return normalized_res

# Compute x, s, y, kappa, tau from z."""
#def _z2xsykappatau(z, cones, n, m):
#    _cones = parse_cone_dict(cones)
#    _cones_parsed = parse_cone_dict_cpp(_cones)

     # These caches are not actually needed.
#    q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
#        = make_prod_cone_cache(cones)
#    u = _coneref.projection_embedded_cone(z, _cones_parsed, q_cache, eval_cache, 
#                                        evec_cache, ep_cache, ed_cache, n, m)
#    v = u - z
#    x, s, y, tau, kappa = uv2xsytaukappa(u, v, n)
#    return x, s, y, tau, kappa

# The following functions are used to set up caches for all cone types.
def make_prod_cone_cache(dim_dict):
    return _make_prod_cone_cache(np.array(dim_dict['q'] if 'q'
                                          in dim_dict else [], dtype=np.int64),
                                 np.array(dim_dict['s'] if 's' in
                                          dim_dict else [], dtype=np.int64),
                                 np.array(dim_dict['ep'] if 'ep' in
                                          dim_dict else 0, dtype=np.int64),
                                 np.array(dim_dict['ed'] if 'ed' in
                                          dim_dict else 0, dtype=np.int64))

def _make_prod_cone_cache(second_ord, semi_def, ep, ed):
    q_cache = np.zeros(np.sum(second_ord))

    eval_cache = np.zeros(np.sum(semi_def))
    evec_cache = np.zeros(np.sum(semi_def**2))

    ep_cache = np.zeros((3*ep, ))
    ed_cache = np.zeros((3*ed, ))

    return q_cache, eval_cache, evec_cache, ep_cache, ed_cache

# Given x, s, y this function returns u = [x y tau], v = [0, s, kappa].
def xsy2uv(x, s, y, tau=1., kappa=0.):
    n = len(x)
    m = len(s)
    u = np.empty(m + n + 1)
    v = np.empty_like(u)
    u[:n] = x
    u[n:-1] = y
    u[-1] = tau
    v[:n] = 0
    v[n:-1] = s
    v[-1] = kappa
    return u, v

# Returns z = u - v"""
def xsy2z(x, s, y, tau=1., kappa=0.):
    u, v = xsy2uv(x, s, y, tau, kappa)
    return u - v


# Given u and v this function returns x, s, y, tau, kappa.
def uv2xsytaukappa(u, v, n):
    tau = np.float(u[-1])
    kappa = np.float(v[-1])
    x = u[:n] / tau if tau > 0 else u[:n] / kappa
    y = u[n:-1] / tau if tau > 0 else u[n:-1] / kappa
    s = v[n:-1] / tau if tau > 0 else v[n:-1] / kappa
    return x, s, y, tau, kappa

#  Refines an approximate solution to the conic linear program
#   min. c^T x subject to Ax + s = b, s \in \mathcal{K}.
def refine_py(A, b, c, cones, z, ref_iter = 2, lsqr_iter = 30, 
              verbose = False):
    _cones = parse_cone_dict(cones)
    _cones_parsed = parse_cone_dict_cpp(_cones)
    m, n = A.shape 
    q_cache, eval_cache, evec_cache, ep_cache, ed_cache \
        = make_prod_cone_cache(cones) # These caches are not actually needed.
    info = {}
    
    # Refine the solution
    tic = time()
    refined_z = _coneref.refine(A, b, c, _cones_parsed, z, n, m, ref_iter, lsqr_iter,
                             verbose)

    # Recover x, y, s from z. Requires an additional projection.
    u = _coneref.embedded_cone_Pi(refined_z, _cones_parsed, q_cache, eval_cache,
                                evec_cache, ep_cache, ed_cache, n, m)
    v = u - refined_z    
    x, s, y, tau, kappa = uv2xsytaukappa(u, v, n)
    
    
    # Store some quantities
    info['ref_time'] = time() - tic
    info['primal_residual'] = np.linalg.norm(A @ x + s - b)
    info['dual_residual'] = np.linalg.norm(A.T @ y + c)
    info['sTy'] = s @ y 
    info['duality_gap'] = c.T @ x + b.T @ y

    return refined_z, x, y, s, tau, kappa, info

def print_residuals(primal_residual = -1, dual_residual = -1, duality_gap = -1,
                    bTy = -1, ATy_norm = -1, Axs_norm = -1, cTx = -1,
                    primal_infeasible = False, dual_infeasible = False,
                    finite_opt_val = False):

    if finite_opt_val:                
        print("Primal residual/dual residual/duality_gap:",
                    "{:.4e}".format(primal_residual), " ",
                    "{:.4e}".format(dual_residual), " ",
                    "{:.4e}".format(duality_gap))
    elif primal_infeasible:
        print("b @ y:", bTy)
        print("||A^T y||:", ATy_norm)
    elif dual_infeasible:
        print("c @ x:", "{:.4e}".format(cTx))
        print("||Ax + s||:", "{:.4e}".format(Axs_norm))