#!/usr/bin/env python

"""
Optimal design of experiment modules
"""
__author__ = "Branislav Kveton, M.J. Azizi"
__copyright__ = "Copyright 2021, USC"

#@title Imports and defaults
"""Kveton optimal design code"""

import numpy as np
from numpy.linalg import norm, inv, eigh, det, svd
from numpy.random import multinomial
from scipy.optimize import linprog
from scipy.linalg import lu


import matlab.engine
mat_eng = matlab.engine.start_matlab()


# @title Optimal designs
def g_grad(X, p, gamma=1e-6, return_grad=True):
    n, d = X.shape

    Xp = X * np.sqrt(p[:, np.newaxis])
    G = Xp.T.dot(Xp) + gamma * np.eye(d)
    invG = inv(G)

    # xGxs = np.diag(X.dot(invG).dot(X.T))
    xGxs = [X[i, :].T.dot(invG).dot(X[i, :]) for i in range(n)]
    i_xmax = np.argmax(xGxs)
    xmax = X[i_xmax, :]
    obj = xGxs[i_xmax]
    if return_grad:
        dp = np.array([-(xmax.T.dot(invG).dot(X[i, :])) ** 2 for i in range(n)])
        # print("G: ", obj, dp)
        # dp /= norm(dp)
        # print("G norm: ", obj, dp)
    else:
        dp = 0

    return obj, dp


def fw_optimal_alloc(X, design="a", num_iters=1000, num_iter_LS=124, tol=1e-9):
    n, d = X.shape

    # initial allocation weights
    alphas = np.ones(n)
    # print(f"Design {design}\n")

    for iter in range(num_iters):
        # compute the gradient
        alphas0 = np.copy(alphas)
        if design == "a":
            obj0, grad_alphas0 = a_grad(X, alphas0)
        elif design == "g":
            obj0, grad_alphas0 = g_grad(X, alphas0)
        elif design == "d":
            obj0, grad_alphas0 = d_grad(X, alphas0)
        else:
            obj0, grad_alphas0 = e_grad(X, alphas0)

        # print("%.4f" % obj0, end=" ")
        # if iter % 10 == 9:
        #     print("\n")

        # find a feasible LP solution in the direction of the gradient
        result = linprog(grad_alphas0, A_ub=np.ones((1, n)), b_ub=n)
        alphas_lp = result.x

        # line search in the direction of the gradient with step sizes 0.75^i
        best_step = 0.0
        best_obj = obj0
        for iter in range(num_iter_LS):
            step = np.power(0.75, iter)
            alphas_ = step * alphas_lp + (1 - step) * alphas0
            if design == "a":
                obj, _ = a_grad(X, alphas_, return_grad=False)
            elif design == "g":
                obj, _ = g_grad(X, alphas_, return_grad=False)
            elif design == "d":
                obj, _ = d_grad(X, alphas_, return_grad=False)
            else:
                obj, _ = e_grad(X, alphas_, return_grad=False)
            if obj < best_obj:
                best_step = step
                best_obj = obj

        # update solution
        alphas = best_step * alphas_lp + (1 - best_step) * alphas0

        if obj0 - obj < tol:
            break
        iter += 1
    # print()

    alphas = np.maximum(alphas, 0)
    alphas /= alphas.sum()
    return alphas


def minvol_todd(X, budget, num_iters=100000, tol=1e-6):
    X = matlab.double([list(x) for x in list(X.T)])
    # tol = matlab.double(tol)
    alphas = mat_eng.minvol(X, tol, 0, num_iters, 0)  # (X,tol,KKY,maxit,print,u)
    alphas = np.array(alphas).squeeze()
    # return multinomial(budget, alphas)
    return np.ceil(budget*alphas)


def fw_opt_FB(X, budget, design="a", num_iters=100, num_iter_LS=24, tol=1e-6):
    alphas = fw_optimal_alloc(X, design=design, num_iters=num_iters, num_iter_LS=num_iter_LS, tol=tol)
    # return multinomial(budget, alphas)
    return np.ceil(budget*alphas)


def optimal_design(X):
    cur_num_arms, d = X.shape
    """Frank Wolfe"""
    pi = np.ones(cur_num_arms) / cur_num_arms  # pi_0 in Frank Wolfe
    X = [a.reshape(d, 1) for a in X]
    eps = 1e-2
    lambda_ = .001
    gpi_k = float('inf')
    k = 0
    while gpi_k > d + eps:
        k += 1
        Vpi_k = lambda_ * np.eye(d)
        for i, a in enumerate(X):
            Vpi_k += pi[i] * np.dot(a, a.T)
        # Vpi_k = np.matrix.sum([pi[i] * a * a.T for i, a in enumerate(X)])
        Vpi_k = inv(Vpi_k)
        a_Vpi = [np.dot(np.dot(a.T, Vpi_k), a) for a in X]
        a_k_idx = np.argmax(a_Vpi)
        gpi_k = a_Vpi[a_k_idx]
        a_k = X[a_k_idx]
        gamma_ = ((1 / d * gpi_k - 1) / (gpi_k - 1))[0][0]
        pi *= (1 - gamma_)
        pi[a_k_idx] += gamma_
    # print(k)
    pi_sum = np.sum(pi)
    if pi_sum != 1:
        # rnd_idx = np.random.randint(0, num_arms)
        rnd_idx = np.argmax(pi)
        pi[rnd_idx] = 1 - pi_sum + pi[rnd_idx]
    return np.array(pi)
