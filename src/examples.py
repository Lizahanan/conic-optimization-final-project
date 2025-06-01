"""
examples.py

Concrete examples of conic optimization problems using cvxpy.
This module provides implementations of 3 conic optimization problems presented in the 
final presentation of the course.

These examples include:
1. A simple linear program (LP) formulated as a conic optimization problem.
2. A simple quadratic program (QP) formulated as a second-order cone program (SOCP).
3. A simple semidefinite program (SDP).

"""

import cvxpy as cp
import numpy as np



def solve_lp_example():
    """
    Solve a bounded LP in conic form:
        minimize   -x
        subject to 2x ≥ 1
                   x ≤ 3
                   x ∈ ℝ

    Converted to:
        minimize   -x₁ + x₂
        subject to 2x₁ - 2x₂ - z₁ = 1
                   x₁ - x₂ + z₂ = 3
                   x₁, x₂, z₁, z₂ ≥ 0
    """
    # Variables: x₁, x₂, z₁, z₂ ≥ 0
    y = cp.Variable(4, nonneg=True)  # [x₁, x₂, z₁, z₂]

    # Constraints matrix
    A = np.array([
        [2, -2, -1,  0],  # 2x₁ - 2x₂ - z₁ = 1
        [1, -1,  0,  1],  # x₁ - x₂ + z₂ = 3
    ])
    b = np.array([1.0, 3.0])

    # Objective: -x₁ + x₂ (recall x = x₁ - x₂)
    c = np.array([-1.0, 1.0, 0.0, 0.0])

    # Solve the problem
    objective = cp.Minimize(c @ y)
    constraints = [A @ y == b]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    print("Status:", problem.status)
    if problem.status == "optimal":
        print("Optimal value:", problem.value)
        print("x₁:", y.value[0])
        print("x₂:", y.value[1])
        print("slack z₁:", y.value[2])
        print("slack z₂:", y.value[3])
        print("Recovered x = x₁ - x₂:", y.value[0] - y.value[1])
    else:
        print("Problem was", problem.status, "- no solution.")


def solve_socp_example():
    """
    Solve an example SOCP problem using cvxpy.
    """
    pass  # implement later

def solve_sdp_example():
    """
    Solve a simple SDP problem using cvxpy.
    """
    pass  # implement later



