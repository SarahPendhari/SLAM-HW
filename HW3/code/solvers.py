from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np


def solve_default(A, b):
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    AtA = csc_matrix(A.T @ A)
    Atb = A.T @ b
    x = inv(AtA) @ Atb
    return x, None


def solve_lu(A, b):
    N = A.shape[1]
    AtA = csc_matrix(A.T @ A)
    Atb = A.T @ b

    lu = splu(AtA, permc_spec="NATURAL")
    x = lu.solve(Atb)
    U = lu.U
    return x, U


def solve_lu_colamd(A, b):
    AtA = csc_matrix(A.T @ A)
    Atb = A.T @ b

    lu = splu(AtA, permc_spec="COLAMD")
    x = lu.solve(Atb)
    U = lu.U
    return x, U


def solve_qr(A, b):
    N = A.shape[1]

    z, R, E, rank = rz(A, b, permc_spec="NATURAL")
    y = spsolve_triangular(R.tocsr(), z, lower=False) 
    y = np.array(y).squeeze()

    x = np.zeros(N)
    x[E] = y
    return x, R


def solve_qr_colamd(A, b):
    N = A.shape[1]

    z, R, E, rank = rz(A, b, permc_spec="COLAMD")
    y = spsolve_triangular(R.tocsr(), z, lower=False)  
    y = np.array(y).squeeze()

    x = np.zeros(N)
    x[E] = y
    return x, R


def solve(A, b, method='default'):
    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }
    return fn_map[method](A, b)