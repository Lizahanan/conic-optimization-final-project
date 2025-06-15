import unittest

import numpy as np

from src.cones import SOC
from src.solver import InteriorPointSolver
from src.function import Linear, Quadratic


class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        self.Solver = InteriorPointSolver()

    def test_quadratic(self):
        ineq = [Linear(row) for row in -np.eye(3)]
        f = Quadratic(np.eye(3)) + Linear([0, 0, 2]) + 1
        x0 = np.array([0.1, 0.2, 0.7])
        A = [1, 1, 1]
        b = 1
        result = self.Solver.solve(func=f,
                                   ineq_constraints=ineq,
                                   eq_constraints_mat=A,
                                   eq_constraints_rhs=b)
        print(f"x: {result['x']}")

    def test_SOCP(self):
        """
        min  x^2+y^2
        s.t. ||x,y||_2 <= 2
             x >= 0
             y >= 0
             x + y = 1
        """
        f = Quadratic(np.eye(2))
        ineq = [
            Linear([-1,0]),
            Linear([0,-1]),
            SOC(A=np.eye(2), d=2)
        ]
        A = [1,1]
        b = 1
        result = self.Solver.solve(func=f,
                                   ineq_constraints=ineq,
                                   eq_constraints_mat=A,
                                   eq_constraints_rhs=b)
        print(f"x: {result['x']}")

if __name__ == '__main__':
    unittest.main()