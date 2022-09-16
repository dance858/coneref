import sys 
import numpy as np
import cvxpy as cp
sys.path.append('..') # For coneref
import coneref


def build_portfolio_opt_problem_instance(p, q, gamma):
    log_mu = np.random.normal(0, 1, p)
    mu = np.exp(log_mu).reshape(p, 1)
    F = np.random.normal(0, 0.1, (p, q))
    d = 0.1*np.random.rand(p, )
    print("Generation of data finished.")

    # Build model in cvxpy.
    z = cp.Variable(p)
    t = cp.Variable(1)
    s = cp.Variable(1)
    u = cp.Variable(1)
    v = cp.Variable(1)
    _objective =  mu.T @ z - gamma*(t+s)
    _constraints = [sum(z) == 1, z>= 0, cp.SOC(u, cp.multiply(np.sqrt(d), z)),
                cp.SOC(v, F.T @ z), cp.SOC(1+t, cp.vstack((1-t, 2*u))),
                cp.SOC(1+s, cp.vstack((1-s, 2*v)))] 
    problem = cp.Problem(objective = cp.Maximize(_objective), constraints = _constraints)   
    return problem

# Define parameters for the problem.
p = 500
q = 100
gamma = 1

# Parameters for the experiment.
num_of_instances = 1
experiment_results = []

for iter in range(0, num_of_instances):
    # Generate problem instance.
    problem = build_portfolio_opt_problem_instance(p, q, gamma)
    coneref.cvxpy_solve(problem, verbose_ref1=True)
    coneref.cvxpy_solve(problem, verbose_ref1=True)
    coneref.cvxpy_solve(problem, verbose_ref1=True)
    





    #coneref.cvxpy_solve(problem, verbose = True)
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)
    #coneref.cvxpy_solve(problem)