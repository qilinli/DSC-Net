import numpy as np
from numpy.linalg import norm


def tpg(W, K):
    n = W.shape[0]
    I = np.eye(n)
    W = W / (W.sum(axis=1, keepdims=True) + np.finfo(float).eps)
    idx = np.argsort(-np.abs(W), axis=1)
    S = np.zeros((n,n))
    for i in range(n):
        S[i, idx[i,:K]] = W[i,idx[i,:K]]
    S = 0.5 * (S + S.T)

    A = S
    max_iter = 50
    epsilon = 1e-2
    for i in range(max_iter):
        temp = np.matmul(np.matmul(S, A), S.T) + I
        if norm(temp - A, 'fro') < epsilon:
            break
        A = temp

    np.fill_diagonal(A, 0)
    A = 0.5 * (A + A.T)
    return A

if __name__== '__main__':
    W = np.arange(20)
    W = np.tile(W, (20, 1))
    A = tpg(W, 10)
