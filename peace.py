#!/usr/bin/env python

"""
https://proceedings.neurips.cc//paper_files/paper/
2020/hash/75800f73fa80f935216b8cfbedf77bfa-Abstract.html

    PEACE algorithm
    An Empirical Process Approach to the Union Bound:
    Practical Algorithms for Combinatorial and Linear Bandits
    Julian Katz-Samuels, Lalit Jain, Zohar Karnin, Kevin Jamieson
    2020 NIPS
"""

__author__ = "M.J. Azizi"
__copyright__ = "Copyright 2021, USC"


import numpy as np, math
from scipy.optimize import minimize
from itertools import repeat
from numpy.random import normal as npNormal
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.sparse import kronsum
from scipy.sparse.linalg import inv as scisparinv


class PEACE():
    def __init__(self, X: np.ndarray, z0, b, theta0, delta):
        self.X: list = X  # X and Z
        self.Z = self.X
        self.delta = delta
        self.K, self.d = X.shape
        self.b = b
        self.z0: np.ndarray = z0.reshape((self.d, 1))
        self.theta0: np.ndarray = theta0.reshape((self.d, 1))

    def _median_of_means(self, seq, n_blocks):
        if n_blocks > len(seq):  # preventing the n_blocks > n_observations
            n_blocks = int(np.ceil(len(seq) / 2))
        # dividing seq in k random blocks
        indic = np.array(list(range(n_blocks)) * int(len(seq) / n_blocks))
        np.random.shuffle(indic)
        # computing and saving mean per block
        means = [np.mean(seq[list(np.where(indic == block)[0])]) for block in range(n_blocks)]
        # return median
        return np.median(means)

    def evalAlloc(self, lam, Arootinv):
        d = self.d
        delta = self.delta
        b = self.b

        T = int(864*d**2/b**2*np.log(1/delta))
        T = 10
        print('evalAlloc T', T)
        # eta = npNormal(0, 1, size=(T, d))
        ys = [self.computeMax(lam, npNormal(0, 1, size=(d, 1)), tol=1/2, Arootinv=Arootinv) for s in range(T)]
        tau = self._median_of_means(np.array(ys).squeeze(), 10)
        return (tau+1)**2

    def dirc_denom(self, z):
        z = z.reshape((self.d, 1))
        dirc = self.z0 - z
        denom = self.b + self.theta0.T.dot(dirc)  # denom of g function
        return dirc, denom

    def g2(self, lam, eta, z, Arootinv):
        dirc, denom = self.dirc_denom(z)
        eta = eta.reshape((self.d, 1))
        return dirc.T.dot(Arootinv).dot(eta)/denom

    def g3(self, lam, eta, r, Arootinv):
        gs = [self.g4(lam, eta, r, z, Arootinv) for z in self.Z]
        # gs = np.array(gs).squeeze()
        return np.max(gs), np.argmax(gs)

    def g4(self, lam: np.ndarray, eta: np.ndarray, r, z: np.ndarray, Arootinv):
        # 4th g in Appendix D of the paper
        z0 = self.z0
        theta0 = self.theta0
        b = self.b
        # X = self.X
        z = z.reshape((self.d, 1))
        eta = eta.reshape((self.d, 1))
        Arootinv = Arootinv.reshape((self.d, self.d))

        Ainv2eta = Arootinv.dot(eta)
        ret = z.T.dot(Ainv2eta + r*theta0) - r*(b+theta0.T.dot(z0))-z0.T.dot(Ainv2eta)
        return ret

    def computeMax(self, lam, eta, tol, Arootinv):
        # print('computeMax')
        Low, High = 0, 2
        jj = 0
        while jj < 1e3 and self.g3(lam, eta, High, Arootinv)[0] >= 0:
            High *= 2
            # print('High*2')
            jj += 1
        kk = 0
        while kk < 1e3 and (self.g3(lam, eta, Low, Arootinv)[0] != 0 or (High+Low)/2 > tol):
            if self.g3(lam, eta, (High+Low)/2, Arootinv)[0] < 0:
                Low = (High+Low)/2
            else:
                High = (High + Low)/2
            tmp = self.g3(lam, eta, Low, Arootinv)
            Low = self.g2(lam, eta, self.Z[tmp[1]], Arootinv)
            kk += 1
            # print('kk')
        return Low

    def calA(self, X, lam):
        # np.diagonal(np.dot(np.dot(X, lam), X.T)).reshape(num_arms, 1)
        return sum([lam[i] * X[i].reshape(self.d, 1).dot(X[i].reshape(1, self.d)) for i in range(self.K)])

    def estimateGradient(self, lam, Arootinv, Ainv, sumlamX):
        eta = npNormal(0, 1, size=(self.d, 1))

        max_val = self.computeMax(lam, eta, 0, Arootinv)
        zbaridx = self.g3(lam, eta, max_val, Arootinv)[1]
        zbar = self.Z[zbaridx][:, np.newaxis]

        """
        grad g2= (z0-z) gradArootinv eta/ ( b+theta0(z0-z) )
        vec(gradArootinv) = (ArootT .Kronecker sum. Aroot)^-1 vec(gradA)
        gradAinv = -A^-1 x_jx_jT A^-1
        """
        dirc, denom = self.dirc_denom(zbar)
        gradg2 = np.empty((self.K, 1))
        ksum_inv = scisparinv(kronsum(Arootinv, Arootinv.T))
        for j in range(self.K):
            xj = self.X[j][:, np.newaxis]
            gradAinv = -Ainv.dot(xj).dot(xj.T).dot(Ainv)
            gradg2j = ksum_inv.dot(np.matrix(gradAinv).flatten(order='F').T)
            gradg2j = np.array(gradg2j).reshape(self.d, self.d)
            gradg2[j, 0] = np.real(dirc.T.dot(gradg2j).dot(eta) / denom)[0, 0]

        return gradg2

    def getArootinv(self, lam):
        A = self.calA(self.X, lam)
        Ainv = inv(A+1e-8*np.eye(self.d))
        sumlamX = np.sum(lam[:, np.newaxis] * self.X, axis=0, keepdims=True).T
        Arootinv = sqrtm(Ainv)
        return Arootinv, Ainv, sumlamX

    def rs_Dphi(self, kappa, rs, lam, d, lams):
        return kappa * rs.T.dot(lam / 2 + 1 / 2 / d) + self.dphi(lam, lams)

    def dphi(self, x, y):
        grad = np.log(y) + 1
        bregman = self.calPhi(x) - self.calPhi(y) - grad.dot(x - y)
        return bregman

    def phi_DeltaTilde(self, lam: np.ndarray):
        x = lam / 2 + 1 / 2 / self.d
        return np.sum(x * np.log(x))

    def calPhi(self, lam: np.ndarray):
        return np.sum(lam * np.log(lam))

    def getAlloc(self):
        cThm8 = 2
        T = cThm8 * np.log(self.d)**2 * self.d ** 3 / self.b ** 2 / self.delta ** 2
        # print('getAlloc T', T)
        T = max(10, min(T, 1e2))  # to curb the runtime
        # print('getAlloc T minmax', T)
        cpThm8 = 2
        kappa = cpThm8 / self.d ** 3 * self.b ** 2 * np.sqrt(2 / T)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bnds = list(repeat((0, None), self.K))
        lams = minimize(self.phi_DeltaTilde, np.ones(self.K)/self.K, bounds=bnds,
                        constraints=cons, method='SLSQP').x  # lambda_s
        sumlam = 1*lams
        for s in range(int(T)):
            # print(s)
            Arootinv, Ainv, sumlamX = self.getArootinv(lams)
            rs = self.estimateGradient(lams, Arootinv, Ainv, sumlamX)
            fun = lambda lam: self.rs_Dphi(kappa, rs, lam, self.d, lams)
            lams = minimize(fun, np.ones(self.K)/self.K, bounds=bnds, constraints=cons).x
            sumlam += lams
        lamfinal = sumlam / T
        Arootinv = self.getArootinv(lamfinal)[0]
        return lamfinal, Arootinv

    def computeAlloc(self, B):
        lam, Arootinv = self.getAlloc()
        # tau = self.evalAlloc(lam, Arootinv)
        # return lam, tau
        n_samples = B * lam
        return lam, n_samples.astype(int)










