# Logistic regression with ell1-regularization.
import sys 
import numpy as np
import cvxpy as cp
sys.path.append('..') # For cpref
import cpref

def build_logistic_regression_problem_instance(p, q):
    # Generate samples z.
    covariance_matrix = np.zeros((p, p))
    for i in range(0, p):
        for j in range(0, p):
            covariance_matrix[i, j] = 2*(0.99**np.abs(i-j))

    Z = np.random.multivariate_normal(np.zeros((p, )), covariance_matrix, size=q)
    w_true = np.random.rand(p, ) - 1/2 
    w_true[0:40] = 0
    w_true[60:] = 0

    # Construct labels.
    y = np.zeros((q, 1))
    for i in range(0, q):
        y[i] = 1 if np.random.random() < 0.5 else -1

    # Compute appropriate value of regularization parameter.
    mu_max = 0.5*np.linalg.norm(np.sum(y*Z, axis = 1), np.inf)
    mu = 0.1*mu_max 

    w = cp.Variable(p)
    obj = sum([cp.log_sum_exp(cp.vstack([0, y[i]*Z[i,:].T @ w])) for i in range(q)])
    obj = obj + mu*cp.norm(w, 1)
    problem = cp.Problem(cp.Minimize(obj))

    return problem 
# Define parameters for the problem.
q, p = 1000, 100

# Parameters for the experiment.
num_of_instances = 1
experiment_results = []

for iter in range(0, num_of_instances):
    # Generate problem instance.
    problem = build_logistic_regression_problem_instance(p, q)
    cpref.cvxpy_solve(problem, verbose_ref1 = True, scs_opts = {'max_iters': 10000})
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)

    # Check consistency with your parsing system. 
    #A, b, c, _cones = cpref.parse_data(problem)
    #z1, x1, y1, s1, info1 = cpref.SCS_solve(A, b, c, _cones, max_iters = 10000, verbose = True)
    #ref_iter, lsqr_iter, verbose = 2, 300, True
    #z2, x2, y2, s2, tau, kappa, info2 = cpref.refine_py(A, b, c, _cones, z1, ref_iter, lsqr_iter, verbose)