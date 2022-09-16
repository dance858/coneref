import numpy as np
from time import time
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from .cones_utils import print_residuals, xsy2z, refine_py

""" 
@param[in] data      Dictionary that contains all problem data for a 
                     conic LP on standard form, ie. constraint matrix A,
                     right-hand side b, cost vector c and dictionary with 
                     cones.
@param[in] sol       TODO: 

@return              TODO:
"""
def cvxpy_scs_to_coneref(data, sol=None):

    A, b, c, dims = data['A'], data['b'], data['c'], data['dims']

    # Om ingen lösning skickas med. Ska denna tas bort?
    if sol is None:
        z = np.zeros(len(b) + len(c) + 1)
        z[-1] = 1.
    # Om en lösning skickas med. 
    else:
        # Hur vet vi att tau är 1 och kappa noll? Varför kan z bli isnan?
        z = xsy2z(sol['x'], sol['s'], sol['y'], tau=1., kappa=0.)

        # Vad är detta?
        if np.any(np.isnan(z)):  # certificate...

            x = np.zeros_like(sol['x']) \
                if np.any(np.isnan(sol['x'])) else sol['x']

            s = np.zeros_like(sol['s']) \
                if np.any(np.isnan(sol['s'])) else sol['s']

            y = np.zeros_like(sol['y']) \
                if np.any(np.isnan(sol['y'])) else sol['y']

            if np.allclose(y, 0.) and c@x < 0:
                obj = c@x
                # assert obj < 0
                x /= -obj
                s /= -obj
                # print('primal res:', np.linalg.norm(A@x + s))

            if np.allclose(s, 0.) and b@y < 0:
                obj = b@y
                # assert obj < 0
                y /= -obj
                # print('dual res:', np.linalg.norm(A.T@y))

            z = xsy2z(x, s, y, tau=0., kappa=1.)

    dims_dict = {}
    if int(dims.nonneg):
        dims_dict['l'] = int(dims.nonneg)
    if int(dims.zero):
        dims_dict['z'] = int(dims.zero)
    if int(dims.exp):
        dims_dict['ep'] = int(dims.exp)
    if len(dims.soc):
        dims_dict['q'] = list([int(el) for el in dims.soc])
    if len(dims.psd):
        dims_dict['s'] = list([int(el) for el in dims.psd])

    return A, b, c, z, dims_dict

def xsy2z_support_infeasibility_and_unboundedness(x, s, y, b, c):
   
    # If the problem is primal infeasible (x and s are nan, y is the certificate).
    if np.any(np.isnan(x)):
        x = np.zeros_like(x)       
        s = np.zeros_like(s) # Borde välja som c istället?
        y = -y/(b @ y)
        z = xsy2z(x, s, y, tau=0., kappa=1.)
    # If the problem is dual infeasible (y is nan, (x, s) is a certificate).
    elif np.any(np.isnan(y)):
        y = np.zeros_like(y)
        obj = c@x
        x /= -obj
        s /= -obj
        z = xsy2z(x, s, y, tau=0., kappa=1.) 
    # If the problem has a finite solution.
    else:
        z = xsy2z(x, s, y, tau=1., kappa=0.)

    return z

def cvxpy_scs_to_coneref(data):
    A, b, c, dims = data['A'], data['b'], data['c'], data['dims']

    dims_dict = {}
    if int(dims.nonneg):
        dims_dict['l'] = int(dims.nonneg)
    if int(dims.zero):
        dims_dict['z'] = int(dims.zero)
    if int(dims.exp):
        dims_dict['ep'] = int(dims.exp)
    if len(dims.soc):
        dims_dict['q'] = list([int(el) for el in dims.soc])
    if len(dims.psd):
        dims_dict['s'] = list([int(el) for el in dims.psd])

    return A, b, c, dims_dict

""" This is the main function of the python interface. It has been designed to
    avoid the need for recompiling a problem that has already been compiled.
    * If cvxpy_problem has already been solved, its solution is refined.
    * If cvxpy_problem has been solved and refined, the refined solution is
          refined again.
    * If the cvxpy_problem has not been solved, the problem is solved using
      SCS and then the solution is refined. 

@param[in] cvxpy_problem    A cvxpy problem.
@param[in] ref_iter         Number of refinement steps.
@param[in] lsqr_iter        Number of LSQR iterations used for each refinement 
                            step.
@param[in] verbose_ref1      If true the refinement procedure writes out the
                             following quantities:
                          1. Primal and dual residual.
                          2. The complementary slackness measure
                             s^T y (here s denotes the primal slack variable 
                             in the standard conic LP form and y denotes
                             the dual variable).
                          3. The duality gap c^T x + b^T y.
                          4. The total time for SCS, and the total time for 
                             the refinement procedure.
@param[in] verbose_ref2      If true the refinement procedure writes out
                             the normalized residual N(z) (see eq. 8 of 
                             refinement paper) after each refinement iteration.
@param[in] warmstart, verbose and scs_opts are settings for SCS.
"""


def cvxpy_solve(cvxpy_problem, ref_iter=2, lsqr_iter=500, verbose_scs=True, scs_opts = {}, verbose_ref1 = True, verbose_ref2 = False, warm_start=False):
    
    # If the solving chain for the cvxpy problem has not been built, we build it 
    # and store it. We also solve the problem (which is on standard conic LP form).
    # Finally, we construct z corresponding to the (x, y, s)-solution obtained
    # from SCS.
    if len(cvxpy_problem._solver_cache) == 0:
        _candidates = {'qp_solvers': [], 'conic_solvers': ['SCS']}
        solving_chain =  construct_solving_chain(cvxpy_problem, candidates=_candidates)
        data, inverse_data = solving_chain.apply(cvxpy_problem)
        cvxpy_problem._solver_cache['data'] = data 
        cvxpy_problem._solver_cache['inverse_data'] = inverse_data
        cvxpy_problem._solver_cache['solving_chain'] = solving_chain
        scs_solution = solving_chain.solve_via_data(cvxpy_problem,
                            data=data, warm_start=warm_start, verbose=verbose_scs,
                            solver_opts=scs_opts)
        
        # If the problem is infeasible the primal variables x and s will be 
        # nan. If the problem is unbounded, the dual variable will be nan.
        x = scs_solution['x']
        y = scs_solution['y']
        s = scs_solution['s']
        total_time = scs_solution['info']['solve_time'] + \
                     scs_solution['info']['setup_time']
        z = xsy2z_support_infeasibility_and_unboundedness(x, s, y, data['b'], data['c'])
        
        # If the problem has a finite optimal value we print the residuals.
        if not (np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(s))):
            A, b, c, dims_dict = cvxpy_scs_to_coneref(data)
            print("After solving the problem with SCS (total time = " + "{:.2e}".format(total_time/1000) + "s):")
            print_residuals(np.linalg.norm(A @ x + s - b), np.linalg.norm(A.T @ y + c),
                            s @ y, c.T @ x + b.T @ y)

    # If the solving chain for the cvxpy problem has been built we reuse it.
    # We also reuse z from the previous round of refinement. 
    else:
        data = cvxpy_problem._solver_cache['data']
        inverse_data =  cvxpy_problem._solver_cache['inverse_data']
        solving_chain = cvxpy_problem._solver_cache['solving_chain']
        z = cvxpy_problem._solver_cache['z']

    # Parse data standard cone LP format.
    A, b, c, dims_dict = cvxpy_scs_to_coneref(data)

    #print("Norm z before refinement:", np.linalg.norm(z))
    refined_z, x, y, s, tau, kappa, info = \
                                  refine_py(A, b, c, dims_dict, z, ref_iter=ref_iter,
                                            lsqr_iter=lsqr_iter, verbose1=verbose_ref1,
                                            verbose2 = verbose_ref2)

    # If the problem has a finite solution.
    if 'SCS' in cvxpy_problem._solver_cache.keys():
        #print("cvxpy_problem._solver_cache['SCS'].keys(): ", cvxpy_problem._solver_cache['SCS'].keys())
        cvxpy_problem._solver_cache['SCS']['x'] = x            
        cvxpy_problem._solver_cache['SCS']['y'] = y 
        cvxpy_problem._solver_cache['SCS']['s'] = s
        cvxpy_problem._solver_cache['z'] = refined_z

        # If SCS terminates because it has reached the maximum number of
        # iterations it seems like cvxpy_problem._solver_cache may not 
        # contain the info field. For the unpacking to work we must 
        # provide the info field with some information?
        # TODO: Add valid information to info field here.
        if ~('info' in cvxpy_problem._solver_cache['SCS'].keys()):
             cvxpy_problem._solver_cache['SCS']['info'] = \
                  {'status_val': 1, "solve_time": 1, "setup_time": 1, "iter": 1,
                   'pobj': x @ c if tau > 0 else np.nan}
        #print("cvxpy_problem._solver_cache['SCS']['info']:", cvxpy_problem._solver_cache['SCS']['info'])
        cvxpy_problem.unpack_results(cvxpy_problem._solver_cache['SCS'], 
                                 solving_chain, inverse_data)
    else:
        cvxpy_problem._solver_cache['SCS'] = {'x': x, 'y': y, 's': s}
        cvxpy_problem._solver_cache['z'] = refined_z
    
    #print("Norm z after refinement:", np.linalg.norm(refined_z))
    #print(" ")

