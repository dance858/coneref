import sys 
import numpy as np
import cvxpy as cp
sys.path.append('..') # For cpref
import cpref
import scipy.sparse as sp

def build_lasso_problem_instance(p, q):
    F = np.random.normal(0, 1, (q, p))
    z_hat = sp.rand(p, 1, density=0.1)
    noise = np.random.normal(0, 0.1, (q, 1))
    g = np.squeeze((F @ z_hat + noise))

    mu = 0.1*np.linalg.norm(F.T @ g, np.inf)
    # Construct model in cvxpy.
    w = cp.Variable(1)
    t = cp.Variable(p)
    z = cp.Variable(p)

    _objective = 1/2*w + mu*cp.sum(t)
    _constraints = [z <= t, z>= -t, cp.SOC(1+w, cp.hstack([1-w, 2*(F@z -g)]))]
    problem = cp.Problem(objective = cp.Minimize(_objective), constraints = _constraints)

    return problem 
# Define parameters for the problem.
q, p = 200, 5000

# Parameters for the experiment.
num_of_instances = 1
experiment_results = []

for iter in range(0, num_of_instances):
    # Generate problem instance.
    problem = build_lasso_problem_instance(p, q)
    cpref.cvxpy_solve(problem, verbose_ref1=True)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
