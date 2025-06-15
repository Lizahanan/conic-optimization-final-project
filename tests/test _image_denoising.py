import unittest

import numpy as np

from src.cones import SOC
from src.function import Quadratic, Linear
from src.image_denoising import load_image, show_image
from src.solver import InteriorPointSolver


class TestImageDenoising(unittest.TestCase):
    @staticmethod
    def load_image():
        return load_image('noisy_img.jpg')

    def test_image_denoising(self):
        lam = 0.3

        Y = self.load_image()
        n = 8
        offset = 100
        Y = Y[offset:offset+n, offset:offset+n]

        m, n = Y.shape
        num_vars = m*n + (m-1)*(n-1) + 1
        constraints = []

        for i in range(m - 1):
            for j in range(n - 1):
                A = np.zeros((2, num_vars))
                A[0, i*n + j] = 1
                A[0, i*n + j + 1] = -1
                A[1, i * n + j] = 1
                A[1, (i+1) * n + j] = -1
                c = np.zeros(num_vars)
                c[i*n + j] = 1
                constraints.append(SOC(A=A, c=c))

        A = np.zeros((m*n, num_vars))
        for i, j in zip(range(m*n), range(n*n)):
            A[i, j] = 1
        b = Y.ravel()
        c = np.zeros(num_vars)
        c[-1] = 1
        constraints.append(SOC(A=A, b=b, c=c))

        for i in range(m*n):
            loc = np.zeros(num_vars)
            loc[i] = 1
            constraints.append(Linear(loc)) # X >= 0
            constraints.append(Linear(loc) - 1) # X <= 1

        ss = np.zeros((num_vars, num_vars))
        ss[-1,-1] = 1
        tv = np.zeros(num_vars)
        tv[m*n:-1] = 1

        objective = Quadratic(ss) + lam * Linear(tv)

        solver = InteriorPointSolver(mu=5)
        result = solver.solve(objective, ineq_constraints=constraints)
        X = result['x'].reshape(Y.shape)
        show_image(X, "denoised")


if __name__ == '__main__':
    unittest.main()