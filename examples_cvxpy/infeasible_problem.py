import numpy as np
import cvxpy as cp
import coneref


x2 = cp.Variable()
constraints = [x2 <= -1]
prob2 = cp.Problem(cp.Minimize(x2), constraints)
print("Unbounded problem: ")
coneref.cvxpy_solve(prob2, verbose = False)

x1 = cp.Variable()
constraints = [x1 <= -1, x1>= 0]
prob1 = cp.Problem(cp.Minimize(x1), constraints)
print("Infeasible problem: ")
coneref.cvxpy_solve(prob1, verbose = False)



