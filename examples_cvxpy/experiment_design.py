import sys 
import numpy as np
import cvxpy as cp
sys.path.append('..') # For cpref
import cpref


def build_experiment_design_problem_instance(n, d):
    HMatrices = []
    for i in range(0, n):
        matrix = np.random.rand(d, d)
        HMatrices.append(matrix.T @ matrix)

    lambdaCVX = cp.Variable((n, 1))
    obj = 0
    for i in range(0, n):
        obj += lambdaCVX[i]*HMatrices[i]

    obj = -cp.log_det(obj)
    constraints = [lambdaCVX >= 0, sum(lambdaCVX) == 1]

    problem = cp.Problem(cp.Minimize(obj), constraints)

    return problem 

# Define parameters for the problem.
n, d = 1000, 12

# Parameters for the experiment.
num_of_instances = 1
experiment_results = []

for iter in range(0, num_of_instances):
    # Generate problem instance.
    problem = build_experiment_design_problem_instance(n, d)
    cpref.cvxpy_solve(problem, verbose_ref1=  True)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
    #cpref.cvxpy_solve(problem)
