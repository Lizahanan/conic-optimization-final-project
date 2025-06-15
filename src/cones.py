import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix

from src.function import Function
from src.utils import parse_affine_vars


class SOC(Function):
    def __init__(self, A, b = None, c = None, d = None):
        self.A, self.b, self.c, self.d = parse_affine_vars(A, b, c, d)
        self.m, self.n = A.shape
        self.A_T = self.A.T
        super().__init__(self.A.shape[1])

    def eval(self, x):
        x = np.asarray(x)
        u = self.A @ x + self.b
        norm_u = np.linalg.norm(u)

        if norm_u == 0:
            raise ValueError("Norm is zero; gradient is undefined.")

        y = np.linalg.norm(u) - self.c @ x - self.d
        g = (self.A_T @ u) / norm_u - self.c
        term1 = (self.A_T @ self.A) / norm_u
        term2 = (self.A_T @ np.outer(u, u) @ self.A) / (norm_u ** 3)
        h = term1 - term2

        return y, g, h

    def pad(self, pad_width, constant_values=0):
        A = np.pad(self.A, pad_width=((0,0), pad_width), constant_values=constant_values, mode='constant')
        c = np.pad(self.c, pad_width=pad_width, constant_values=constant_values, mode='constant')
        return SOC(A, self.b.copy(), c, self.d)