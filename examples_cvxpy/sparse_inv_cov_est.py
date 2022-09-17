import numpy as np
import cvxpy as cp
import scipy.sparse as sp
import scipy
from sklearn.datasets import make_sparse_spd_matrix
import coneref

# p = number of samples, ratio = fraction of zeros in S, q = dimension of covariance
# matrix.
def build_sparse_inv_cov_estimation_problem_instance(p, q, ratio):
    alpha_ratio = 0.001

    S_true = sp.csc_matrix(make_sparse_spd_matrix(q, alpha = ratio))
    Sigma = sp.linalg.inv(S_true).todense()
    z_sample = scipy.linalg.sqrtm(Sigma).dot(np.random.randn(q,p))
    Q = np.cov(z_sample)

    mask = np.ones(Q.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    alpha_max = np.max(np.abs(Q)[mask])
    alpha = alpha_ratio*alpha_max   # 0.001 for q = 100, 0.01 for q = 50

    S = cp.Variable((q, q), PSD = True)
    obj = -cp.log_det(S) + cp.trace(S @ Q) + alpha*cp.sum(cp.abs(S))
    problem = cp.Problem(objective = cp.Minimize(obj))

    return problem 

# Define parameters for the problem.
p, q, ratio = 1000, 50, 0.9

# Parameters for the experiment.
num_of_instances = 1
experiment_results = []

for iter in range(0, num_of_instances):
    # Generate problem instance.
    problem = build_sparse_inv_cov_estimation_problem_instance(p, q, ratio)
    coneref.cvxpy_solve(problem, verbose_ref1 = True, scs_opts = {})
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)

    # Check consistency with your parsing system.
    #A, b, c, _cones = coneref.parse_data(problem)
    #z1, x1, y1, s1, info1 = coneref.SCS_solve(A, b, c, _cones, max_iters = 4000, verbose = True)
    #ref_iter, lsqr_iter, verbose = 2, 300, True
    #z2, x2, y2, s2, tau, kappa, info2 = coneref.refine_py(A, b, c, _cones, z1, ref_iter, lsqr_iter, verbose)