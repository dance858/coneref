import numpy as np
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from .cones_utils import print_residuals, xsy2z, refine_py

""" Transform (x, y, s) returned from SCS to z-space. """
def xsy2z_support_certificates(x, s, y, b, c):
    # This function is called with the output (x, s, y) from SCS.
    # If the problem has a finite optimal value, SCS returns a triple (x, y, s)
    # satisfying the KKT-conditions for the conic LP.
    # If the problem is primal infeasible, then SCS returns (x, s) = nan and
    # a vector y such that b @ y = -1, A^T @ y = 0.
    # If the problem is dual infeasible, then SCS returns y = nan and (x, s) 
    # satisfying Ax + s = 0, c @ x = -1.

    # If the problem is primal infeasible.
    if np.any(np.isnan(x)):
        x = np.zeros_like(x)       
        s = np.zeros_like(s) 
        #y = -y/(b @ y)    # Not needed if b @ y = -1, but we want to make sure that
                          # our approximate certificate really satisfies b @ y = -1.
        z = xsy2z(x, s, y, tau=0., kappa=1.)
    # If the problem is dual infeasible.
    elif np.any(np.isnan(y)):
        y = np.zeros_like(y)
        #obj = c@x         # The following two lines are not needed if c @ x = -1,
        #x /= -obj         # but we want to make sure that our approximate
        #s /= -obj         # certificate really satisfies c @ x = -1.
        z = xsy2z(x, s, y, tau=0., kappa=1.) 
    # If the problem has a finite solution.
    else:
        z = xsy2z(x, s, y, tau=1., kappa=0.)

    return z

""" Parses cvxpy data object to standard conic LP form. """
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
        cvxpy_problem._solver_cache['scs_total_time'] = total_time
        
        z = xsy2z_support_certificates(x, s, y, data['b'], data['c'])
        
        # Print residuals. There are residuals inside "scs_solution['info]"
        # but these residuals do not coincide with the true residuals. Maybe
        # they are computed for the scaled problem?
        A, b, c, dims_dict = cvxpy_scs_to_coneref(data)
        print("After SCS (total time = " + "{:.2e}".format(total_time/1000) + "s):")
        if np.any(np.isnan(x)):
            print_residuals(bTy=b@y, ATy_norm=np.linalg.norm(A.T @ y), 
                            primal_infeasible=True)
            cvxpy_problem._solver_cache['primal_infeasible'] = True
        elif np.any(np.isnan(y)):
            print_residuals(cTx=c@x, Axs_norm=np.linalg.norm(A @ x + s), 
                            dual_infeasible=True)
            cvxpy_problem._solver_cache['dual_infeasible'] = True
        else:
            cvxpy_problem._solver_cache['SCS']['info']['res_pri_before_ref'] = np.linalg.norm(A @ x + s - b)
            cvxpy_problem._solver_cache['SCS']['info']['res_dual_before_ref'] = np.linalg.norm(A.T @ y + c)
            cvxpy_problem._solver_cache['SCS']['info']['gap_before_ref'] = c.T @ x + b.T @ y
            print_residuals(primal_residual = cvxpy_problem._solver_cache['SCS']['info']['res_pri_before_ref'],
                           dual_residual = cvxpy_problem._solver_cache['SCS']['info']['res_dual_before_ref'],
                           duality_gap = cvxpy_problem._solver_cache['SCS']['info']['gap_before_ref'],
                           finite_opt_val = True)
            cvxpy_problem._solver_cache['finite_opt_val'] = True

    # If the solving chain for the cvxpy problem has been built we reuse it.
    # We also reuse z from the previous round of refinement. 
    else:
        data = cvxpy_problem._solver_cache['data']
        inverse_data =  cvxpy_problem._solver_cache['inverse_data']
        solving_chain = cvxpy_problem._solver_cache['solving_chain']
        z = cvxpy_problem._solver_cache['z']

    # Parse data standard cone LP format.
    A, b, c, dims_dict = cvxpy_scs_to_coneref(data)

    refined_z, x, y, s, tau, kappa, info = \
                                  refine_py(A, b, c, dims_dict, z, ref_iter=ref_iter,
                                            lsqr_iter=lsqr_iter, verbose = verbose_ref2)
    cvxpy_problem._solver_cache['refine_time'] = info['ref_time']*1000 # In milliseconds
    
    if verbose_ref1:
        print("After refinement (ref time =" + "{:.2e}".format(info['ref_time']) + "s):")
        if 'primal_infeasible' in cvxpy_problem._solver_cache:
            print_residuals(bTy=b@y, ATy_norm=np.linalg.norm(A.T @ y), 
                            primal_infeasible=True)
        elif 'dual_infeasible' in cvxpy_problem._solver_cache:
             print_residuals(cTx=c@x, Axs_norm=np.linalg.norm(A @ x + s), 
                            dual_infeasible=True)
        elif 'finite_opt_val' in cvxpy_problem._solver_cache:
            print_residuals(primal_residual = info['primal_residual'],
                           dual_residual = info['dual_residual'],
                           duality_gap = info['duality_gap'],
                           finite_opt_val = True)


    # If the problem seems to have a finite solution. 
    if 'finite_opt_val' in cvxpy_problem._solver_cache: 
        # If SCS terminates because it has reached the maximum number of
        # iterations, cvxpy_problem._solver_cache does not contain the 'SCS' field
        # so we add it manually.
        if not ('SCS' in cvxpy_problem._solver_cache.keys()):
            cvxpy_problem._solver_cache['SCS'] = {'info': {'status_val': 1,
            'solve_time': scs_solution['info']['solve_time'], 
            'setup_time': scs_solution['info']['setup_time'],
            'iter': scs_solution['info']['iter']}}

        # Update the information in SCS. 
        cvxpy_problem._solver_cache['SCS']['x'] = x            
        cvxpy_problem._solver_cache['SCS']['y'] = y 
        cvxpy_problem._solver_cache['SCS']['s'] = s
        cvxpy_problem._solver_cache['SCS']['info']['pobj'] = c @ x 
        cvxpy_problem._solver_cache['SCS']['info']['dobj'] = -b @ y      # Need the minus sign to compensate for earlier change of sign.
        cvxpy_problem._solver_cache['SCS']['info']['res_pri_after_ref'] = info['primal_residual']
        cvxpy_problem._solver_cache['SCS']['info']['res_dual_after_ref'] = info['dual_residual']
        cvxpy_problem._solver_cache['SCS']['info']['gap_after_ref'] = info['duality_gap']
        cvxpy_problem._solver_cache['SCS']['info']['comp_slack_after_ref'] = info['sTy']
        cvxpy_problem._solver_cache['z'] = refined_z
        cvxpy_problem.unpack_results(cvxpy_problem._solver_cache['SCS'], 
                                 solving_chain, inverse_data)
    # If primal infeasible problem.
    elif 'primal_infeasible' in cvxpy_problem._solver_cache:
        cvxpy_problem._solver_cache['primal_infeas_cert'] = {'y': y}
        cvxpy_problem._solver_cache['z'] = refined_z
    elif 'dual_infeasible' in cvxpy_problem._solver_cache:
        cvxpy_problem._solver_cache['dual_infeas_cert'] = {'x': x, 's': s}
    

