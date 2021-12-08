#!/usr/bin/env python

"""Main framework for running BAI algorithms"""

__author__ = "M.J. Azizi"
__copyright__ = "Copyright 2021, USC"

import time

import numpy as np
from numpy.linalg import inv, norm, eig, svd, qr
import pandas as pd
from itertools import product, repeat
import math
import os
import sys
import platform
from collections import Counter
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler, minmax_scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans
from utils import intersection, write_csv, get_pst_time, sigmoid, inv_wrap, find_two_closest, make_dir
import multiprocessing as mp
from numpy.random import normal as npNormal
from numpy.random import binomial as npBinomial, multinomial
from optDesign import fw_opt_FB, minvol_todd
from scipy import optimize
from scipy.optimize.nonlin import NoConvergence
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, \
    ExpSineSquared, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from peace import PEACE


mns_ = MinMaxScaler()

np.random.seed(110)


class super_class():
    def __init__(self, seed_, N_1, num_arms, dim, reward_dist, max_depth_dtr, budget_per_arm, sigma2,
                 fb, multithr, rewards_base='auto', BG_par={},
                 algos=None, GPUCB_par=None, params={}, BperArm=1):
        self.params = params
        self.fb = fb
        self.deltas = ""
        self.rewards_base = rewards_base
        self.df_train_fix: pd.DataFrame = None
        if self.params['synt_num'] in [1, 5] or self.rewards_base == 'aml':
            self.params['random_exp'] = False
        self.params['synt1_omega'] = 1/10
        self.N_1 = N_1
        self.log_step = max(int(N_1/10), 1)
        self.trgt_feat = None
        self.df_train = None
        self.arms = None
        self.theta = None
        self.dim = dim
        self.num_arms = num_arms
        if self.params['synt_num']==1:
            self.num_arms = self.dim + 1
        elif self.params['synt_num']==5:
            self.num_arms = self.dim
        self.sigma2 = sigma2
        self.prior_sigma = params['prior_sigma']
        self.seed_ = seed_
        self._set_seeds()
        # self.rs = np.random.RandomState(seed=self.seed_)
        # self.seeds = self.rs.randint(100, 1000, N_1)
        self.sample_id = 0
        self.X_train, self.y_train = None, None
        self.max_depth_dtr=max_depth_dtr
        self.multithr = multithr
        self.BG_par = BG_par
        self.algos = algos
        self.alpha_linucb = 1/np.sqrt(np.log(2/.001)/2)
        self.GPUCB_par = GPUCB_par
        self.reward_dist = reward_dist

        self.read_data()
        self.budget_per_arm = budget_per_arm
        self.BperArm = BperArm
        self.budget = budget_per_arm * self.num_arms if BperArm else budget_per_arm
        self.GSE_on = 0
        if any(['GSE' in algo for algo in algos]):
            self.GSE_on = 1
            self.n_stage_SE()
        if reward_dist == 'Gaus':
            self.reward = self.gauss_reward
        else:
            self.reward = self.bern_reward
        if self.rewards_base == 'aml':
            self.reward = self.aml._sample_eval_size

        print("K", self.num_arms, self.algos, "B/K" if BperArm else "B", self.budget_per_arm, "d", self.dim, self.params)

    def _synt_data(self):
        synt_num = self.params['synt_num']

        if not self.params['random_exp']:
            self.rs = np.random.RandomState(seed=110)

        if synt_num == 1:
            self.num_arms = self.dim+1
            self.arms = np.eye(self.dim)
            v = np.zeros((1, self.dim))
            v[0, 0], v[0, 1] = math.cos(self.params['synt1_omega']), math.sin(self.params['synt1_omega'])
            self.arms = np.vstack([self.arms, v])
            self.theta = np.eye(1, self.dim)
        elif synt_num == 3:
            self.arms = self.rs.random(size=(self.num_arms, self.dim)) - .5  # unif(-.5, .5)
            # self.theta = np.random.normal(0, self.prior_sigma, size=(1, self.dim))
            self.theta = self.rs.normal(0, 3 * self.prior_sigma / self.dim, size=(1, self.dim))  # normal
        elif synt_num == 5:
            # LinGapE paper example
            self.arms = np.eye(self.dim)
            self.theta = np.zeros(self.dim)
            self.theta[0] = self.params['delta_s5']
        elif synt_num == 6:
            # ALBA paper, OD-LinBAI 5.2 too
            self.arms = self.rs.random(size=(self.num_arms, self.dim))*2-1  # unif(-1, 1)
            self.arms /= norm(self.arms, axis=1)[:, np.newaxis]
            x, y, _, _ = find_two_closest(self.arms)
            self.theta = (1 - self.params['alpha_s6']) * x + self.params['alpha_s6']*y
        elif synt_num == 7:
            # Section 5.1 from OD-LinBAI paper
            self.dim = 2
            pi4 = np.pi/4
            tmp_arms = [[np.cos(pi4+phi), np.sin(pi4+phi)] for phi in npNormal(0, 0.09, size=self.num_arms-2)]
            tmp_arms += [[np.cos(3*pi4), np.sin(3*pi4)]]
            tmp_arms = [[1, 0]] + tmp_arms

            self.arms = np.array(tmp_arms)
            self.theta = np.array([1, 0])

        self.df_train = pd.DataFrame(self.arms)
        #####
        self.trgt_feat = "trgt"
        if self.params['synt_num'] == 3:
            tmp = np.dot(self.df_train.to_numpy(), self.theta.T)
            self.df_train[self.trgt_feat] = sigmoid(tmp)
            # self.df_train[self.trgt_feat] = np.maximum(np.sign(1 / (1 + np.exp(-tmp))-0.5), 0).astype(int)
        else:
            self.df_train[self.trgt_feat] = np.dot(self.df_train.to_numpy(), self.theta.T)

        return self.df_train

    def _set_seeds(self):
        self.rs = np.random.RandomState(seed=self.seed_)
        self.seeds = self.rs.randint(10, 1e8, self.N_1)

    def _set_replic_seed(self, sim_id):
        _seed = self.seeds[sim_id]
        np.random.seed(_seed)
        self.rs = np.random.RandomState(seed=_seed)

    def nan_replace_mean(self, feat_name, dtype_):
        df_temp = self.df_train[self.df_train[feat_name] != '?'].copy()
        normalised_mean = df_temp[feat_name].astype(dtype_).mean()
        self.df_train[feat_name] = self.df_train[feat_name].replace('?', normalised_mean).astype(dtype_)

    def get_deltas(self, y):
        return np.sort(np.max(y) - y)

    def scale_data(self):
        for i in self.df_train:
            self.df_train.loc[:, i] = mns_.fit_transform(self.df_train.loc[:, i].values.reshape(-1, 1))

    def read_data(self):
        if self.rewards_base == 'auto':
            self.trgt_feat = 'price'
            self.df_train = pd.read_csv("./Auto/autodata/Automobile_data.csv")
            df_data = self.df_train.replace("?", np.nan)
            tmpList = [(self.trgt_feat, int)]
            for feat_name, dtype_ in tmpList:
                self.nan_replace_mean(feat_name, dtype_)
            subset_ = ['curb-weight', 'width', 'engine-size', 'city-mpg', 'highway-mpg', self.trgt_feat]
            self.df_train = self.df_train[subset_]
            # Standard Scalar
            self.df_train = self.df_train.astype(float)
            self.scale_data()
            self.dim = self.df_train.shape[1]-1
        elif self.rewards_base == 'pmsm':
            cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            self.df_train = pd.read_csv('./PMSM-data/pmsm_temperature_data.csv', usecols=cols)
            self.df_train = self.df_train.astype(float)
            self.scale_data()
            self.trgt_feat = 'motor_speed'
            self.dim = self.df_train.shape[1]-1

        if self.rewards_base == 'synt':
            self.df_train_fix = self._synt_data()
        else:
            self.df_train_fix = self.bootstrap(ss=self.num_arms)

        if self.rewards_base != 'aml':
            self.deltas = self.get_deltas(self.df_train_fix[self.trgt_feat])
        self.SE_dim = 1 * self.dim

    def gauss_reward(self, mu, sigma2, size, rs=None, config=None):
        tmp = self.rs.normal(mu, np.sqrt(sigma2), size=size)
        # print(tmp)
        return tmp

    def bern_reward(self, mu, sigma2, size, rs=None, config=None):
        return npBinomial(1, mu, size)

    def optimal_design(self, X):
        cur_num_arms, d = X.shape
        """Frank Wolfe G-opt, Lattimore 2019"""
        pi = np.ones(cur_num_arms)/cur_num_arms  # pi_0 in Frank Wolfe
        X = [a.reshape(d, 1) for a in X]
        eps = self.params['optimal_design_eps']
        lambda_ = self.params['SE_lambda']
        gpi_k = float('inf')
        k = 0
        while gpi_k > d+eps:
            k+=1
            Vpi_k = lambda_*np.eye(d)
            for i, a in enumerate(X):
                Vpi_k += pi[i] * np.dot(a, a.T)
            # Vpi_k = np.matrix.sum([pi[i] * a * a.T for i, a in enumerate(X)])
            Vpi_k = inv(Vpi_k)
            a_Vpi = [np.dot(np.dot(a.T, Vpi_k), a) for a in X]
            a_k_idx = np.argmax(a_Vpi)
            gpi_k = a_Vpi[a_k_idx]
            a_k = X[a_k_idx]
            gamma_ = ((1/d*gpi_k-1)/(gpi_k-1))[0][0]
            pi *= (1-gamma_)
            pi[a_k_idx] += gamma_

        return pi

    def g_opt_gready(self, X, budget, type):
        """G greedy and XY greedy from Soare et al. 2014
        i.e. sequential optimal design"""
        cur_num_arms, dim = X.shape
        X = np.hstack([np.ones((cur_num_arms, 1)), X])
        dim += 1
        X = [a.reshape(dim, 1) for a in X]
        A = np.eye(dim)
        n_samples = np.zeros(cur_num_arms)
        sample_path = []
        if type == 'G-greedy':
            xpmax_f = lambda tmp: max([xp.T.dot(tmp).dot(xp) for xp in X])
        elif type == 'XY-greedy':
            xpmax_f = lambda tmp: max([(xp - xpp).T.dot(tmp).dot(xp - xpp) for xp, xpp in product(X, X)])
        for i in range(int(budget)):
            xargmin, xmin = None, float('inf')
            for xc, x in enumerate(X):
                tmp = inv(A + np.dot(x, x.T))
                xpmax = xpmax_f(tmp)
                if xpmax < xmin:
                    xargmin, xmin = xc, xpmax

            A = A+np.dot(X[xargmin], X[xargmin].T)
            n_samples[xargmin] += 1
            sample_path.append(xargmin)

        return n_samples.astype(int)

    def se_train_explore(self, sim_id, df_train, budget, num_plays_ind,
                         km=False, not_elim_arms=None, _memory=1, pre_reward=[],
                         pre_X_train=None, prev_y_train=None, optimal_design=0,
                         params={}, thetahat=0, algoname=None, stage_cntr=None):

        X_test, y_test = df_train.drop(self.trgt_feat, axis=1).to_numpy(), df_train[self.trgt_feat]

        # cur_num_arms = self.X_test.shape[0]
        cur_num_arms = X_test.shape[0]
        trgt = df_train[self.trgt_feat].to_numpy()
        tmp_trgt_feat = []
        tmp_df_train = pd.DataFrame()

        """Exploration"""
        n_samples = None
        if optimal_design:
                if cur_num_arms > 2:
                    pi = self.optimal_design(1*X_test)
                else:
                    pi = np.ones(cur_num_arms)/cur_num_arms
                n_samples = multinomial(budget, pi)
                # n_samples = np.floor(pi*budget).astype(int)
        elif 'Greedy' in params:
            n_samples = self.g_opt_gready(1 * X_test, budget, type='G-greedy')
        elif 'G-opt' in params:
            # n_samples = minvol_todd(X=1 * X_test, budget=budget, tol=self.params['SE_proj_tol'])
            if (algoname == 'OD-LinBAI' and self.params['OC_cons']) \
                    or (algoname in ['GSE-Lin-Todd', 'GSE-Lin-Todd-1']):  # and stage_cntr == 0:
                # n_samples = Epeeling(X=1 * X_test, budget=budget, design='g', tol=self.params['SE_proj_tol'])
                n_samples = minvol_todd(X=1 * X_test, budget=budget, tol=self.params['SE_proj_tol'])
            else:
                n_samples = fw_opt_FB(X=1 * X_test, budget=budget, design='g')
        elif 'Peace' in params:
            X_test_tmp = X_test
            z0 = X_test_tmp[np.argmax(X_test_tmp.dot(thetahat))]
            peace = PEACE(X_test_tmp, z0=z0, b=1, theta0=thetahat, delta=.8)
            lam, n_samples = peace.computeAlloc(B=budget)
            bdiffn = budget - np.sum(n_samples)
            if bdiffn > 0:
                # n_samples += multinomial(bdiffn, lam)
                n_samples += multinomial(bdiffn, np.ones(cur_num_arms)/cur_num_arms)
            elif bdiffn < 0:
                n_samples -= multinomial(-bdiffn, np.ones(cur_num_arms)/cur_num_arms)
        else:
            """uniform exploring"""
            budget_per_arm, exes = int(budget / cur_num_arms), int(budget % cur_num_arms)
            n_samples = np.ones(cur_num_arms)*budget_per_arm
            n_samples[self.rs.choice(cur_num_arms, size=exes, replace=False)] += 1

        n_samples = np.array(n_samples).astype(int)

        for i in range(cur_num_arms):
            tmp_trgt_feat += list(self.reward(trgt[i], self.sigma2, size=n_samples[i], rs=self.rs, config=X_test[i]))
            num_plays_ind[not_elim_arms[i]] += n_samples[i]
            tmp_df_train = tmp_df_train.append([df_train.iloc[i]] * n_samples[i])
        tmp_trgt_feat = np.array(tmp_trgt_feat).reshape(sum(n_samples), 1)

        df_train = tmp_df_train
        df_train[self.trgt_feat] = tmp_trgt_feat

        if _memory:
            """With memory"""
            X_train = pd.concat([pre_X_train, df_train.drop(self.trgt_feat, axis=1)], ignore_index=True)
            y_train = pd.concat([prev_y_train, df_train[self.trgt_feat]], ignore_index=True)
        else:
            """No memory"""
            X_train, y_train = \
                df_train.drop(self.trgt_feat, axis=1), df_train[self.trgt_feat]

        return X_train, y_train, X_test, y_test

    def _linear_reg(self, sim_id, X_train, X_test, y_train, y_test, _lambda):

        """Linear regression from scratch"""
        X_train = X_train.to_numpy()
        V_t = np.dot(X_train.T, X_train)
        V_t = inv_wrap(V_t, _lambda)
        b_t = np.dot(X_train.T, y_train)
        theta = np.dot(V_t, b_t)

        pred = np.array([np.dot(x.T, theta) for x in X_test])

        if self.rewards_base != 'aml':
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            r2s = r2_score(y_test, pred)
            pred_max = np.where(pred==pred.max())[0]
            y_test = y_test.to_numpy()
            y_test_max = np.where(y_test==y_test.max())[0]
            crct = len(intersection(pred_max, y_test_max)) > 0
        else:
            # we don't have the y_test (ground truth for AutoML)
            mae, mse, r2s, crct = 0, 0, 0, 0

        ret = {}
        ret['acc'], ret['mae'], ret['mse'], \
        ret['r2s'], ret['pred'], ret['_lambda'] = crct, mae, mse, r2s, pred, _lambda
        ret['thetahat'] = theta
        return ret

    def _Logistic_reg(self, sim_id, X_train, X_test, y_train, y_test, _lambda):

        """GLM for Logistic regression"""
        cur_num_arms, dim = X_train.shape
        df = lambda theta: np.dot((sigmoid(np.dot(X_train, theta))-y_train), X_train)
        try:
            theta = optimize.newton_krylov(df, np.zeros(dim))
            converged = True
        except NoConvergence as e:
            theta = e.args[0]
            converged = False
        except ValueError:
            theta = npNormal(loc=0, scale=1, size=dim)

        pred = np.dot(theta, X_test.T)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        r2s = r2_score(y_test, pred)

        pred_max = np.where(pred == pred.max())[0]
        y_test_max = np.where(y_test == y_test.max())[0]
        crct = len(intersection(pred_max, y_test_max)) > 0
        ret = {}
        ret['acc'] = crct

        ret['mae'], ret['mse'], \
        ret['r2s'], ret['pred'], ret['_lambda'] = mae, mse, r2s, pred, _lambda

        return ret

    def bootstrap(self, ss, sim_id=None):
        if not sim_id:
            sim_id = self.sample_id
            self.sample_id += 1

        # assert self.params['synt_num'] != 1, "synt 1 called bootstrap"

        # if self.rewards_base == "synt":
        #     sample_df = self._synt_data()
        # else:
        #     sample_df = self.df_train.sample(n=ss, replace=True, random_state=self.seeds[sim_id])
        #     sample_df.reset_index(drop=True, inplace=True)

        sample_df = self.df_train.sample(n=ss, replace=True, random_state=self.seeds[sim_id])
        sample_df.reset_index(drop=True, inplace=True)

        return sample_df #, sample_df[self.trgt_feat].to_numpy()

    def simple_regret(self, y, recoms):
        if self.rewards_base == 'aml':
            return -1, -1
        else:
            crt_idx = np.argmax(y)
            if crt_idx in recoms:
                crct = True
                simplereg = 0
            else:
                crct = False
                simplereg = y[crt_idx] - y[int(recoms[0])]
            return crct, simplereg

    def linUCB_disjoint(self, sample_df, sim_id):
        """
        A Contextual-Bandit Approach to Personalized News Article Recommendation, 2010
        :param sample_df: the sampled dataframe
        :param sim_id: the id of the simulation run
        :return:
        """
        n_trial = self.budget
        X, y = np.array(sample_df.drop(self.trgt_feat, axis=1)), np.array(sample_df[self.trgt_feat])
        num_plays_ind = np.zeros(self.num_arms)
        n_feature = X.shape[1]
        arm_choice, r_payoff = [np.empty(n_trial) for _ in range(2)]
        theta = np.empty(shape=(self.num_arms, n_feature))
        p = np.empty(shape=(self.num_arms))

        # 1.2.intermediate object
        A = np.array([np.diag(np.ones(shape=n_feature)) for _ in np.arange(self.num_arms)])
        b = np.array([np.zeros(shape=n_feature) for _ in np.arange(self.num_arms)])

        # 2. Algo
        for t in np.arange(n_trial):

            # Compute estimates (theta) and prediction (p) for all arms
            for a in np.arange(self.num_arms):
                inv_A = np.linalg.inv(A[a])  # caching matrix inversion result because used twice
                theta[a] = inv_A.dot(b[a])
                p[a] = theta[a].dot(X[a]) + self.alpha_linucb * np.sqrt(X[a].dot(inv_A).dot(X[a]))

            # chosing best arms
            chosen_arm = int(np.argmax(p))
            x_chosen_arm = X[chosen_arm]
            r_payoff[t] = self.gauss_reward(y[chosen_arm], self.sigma2, size=1)
            # r_payoff[t]   = self.reward_lin(theta_true, x_chosen_arm)
            num_plays_ind[chosen_arm] += 1
            arm_choice[t] = chosen_arm

            # update intermediate objects (A and b)
            A[chosen_arm] += np.outer(x_chosen_arm, x_chosen_arm.T)
            b[chosen_arm] += r_payoff[t] * x_chosen_arm

        arm_choice_dic = Counter(arm_choice)
        if 0:
            # take the average
            X_best_est = np.average([cnt*X[int(arm)] for arm, cnt in arm_choice_dic.items()], axis=0)
            best = np.argmin([norm(X[arm]-X_best_est)/norm(X[arm]+X_best_est) for arm in np.arange(self.num_arms)])
        else:
            """Most Freq played (mf)"""
            best = arm_choice_dic.most_common(1)[0][0]  # take the most used

        crct, simplereg = self.simple_regret(y, [best])
        if sim_id % self.log_step == 0:
            print('linUCB disjoint sample id {}, budget_per_arm'
                  ' {}: correct {}'.format(sim_id, self.budget_per_arm, self.alpha_linucb, crct))

        ret = {}
        ret['acc'] = crct
        ret['recoms'] = best
        ret['simple_regret'] = simplereg
        ret['num_plays_ind'] = num_plays_ind
        return ret

    def n_stage_SE(self):
        if 'SE_eta' in self.params:
            self.SE_n_stagess = (np.log(self.num_arms) / np.log(self.params['SE_eta']))
            # assert self.SE_n_stagess.is_integer(), "Number of arms must be a power of eta"
            # self.SE_n_stagess = int(self.SE_n_stagess)
            self.SE_n_stagess = np.ceil(self.SE_n_stagess).astype(int)

    def proj_se(self, sample_df, OC_proj):
        X = sample_df.drop(self.trgt_feat, axis=1).to_numpy()
        rankX = np.linalg.matrix_rank(X)
        if rankX != X.shape[1]:
            Xsvd = svd(np.dot(X.T, X))
            X = X.dot(Xsvd[0][:, :rankX])
            sample_df.iloc[:, :rankX] = X
            sample_df.drop([i for i in range(3, sample_df.shape[1] - 1)], axis=1, inplace=True)
            self.dim = sample_df.shape[1]
        return sample_df

    def suces_elim(self, sample_df, sim_id, Lin=0, km=False,
                   _memory=1, pre_reward=[], optimal_design=0, params={}, algoname='GSE'):

        self.SE_dim = 1*self.dim
        y = sample_df[self.trgt_feat].copy()
        X = sample_df.drop(self.trgt_feat, axis=1)
        d_tilde = np.linalg.matrix_rank(X)
        eta = self.params['SE_eta']
        # print(sim_id, "is", sample_df)

        num_stages = int(np.log(self.num_arms)/np.log(eta))
        if self.GSE_on:
            num_stages = self.SE_n_stagess

            if algoname in ['GSE-Lin-FWG-1', 'GSE-Lin-Todd-1']:
                num_stages = 1
                eta = self.num_arms

        if algoname == 'OD-LinBAI':
            num_stages = np.ceil(np.log(self.dim) / np.log(2))
            num_stages = int(num_stages)

            tmp = int(np.ceil(np.log2(d_tilde)) - 1)
            tmp = np.sum([np.ceil(d_tilde/2**r) for r in range(1, tmp)]) if tmp>1 else 0
            m_OD_LinBAI = (self.budget - min([self.num_arms, d_tilde * (d_tilde + 1) / 2]) - tmp)/\
                          np.ceil(np.log2(d_tilde))
            # stage_budgets = np.ones(num_stages) * m_OD_LinBAI
            stage_b = int(self.budget / num_stages)
            stage_budgets = np.ones(num_stages) * min(m_OD_LinBAI, stage_b) #
            stage_b_exes = 0
        else:
            stage_b, stage_b_exes = int(self.budget / num_stages), int(self.budget % num_stages)
            stage_budgets = np.ones(num_stages) * stage_b

        if stage_b_exes > 0:
            stage_budgets[-stage_b_exes:] += 1

        not_elim_arms = sample_df.index.to_list()
        _mae, _mse, _r2s, stage_cntr = 0, 0, 0, 0
        f1, recall, precision = 0, 0, 0
        num_plays_ind = {i: 0 for i in not_elim_arms}
        # for stage in range(num_stages):

        X_train, y_train = None, None
        _lambda = self.params['SE_lambda']
        thetahat = np.zeros(self.dim)
        while len(not_elim_arms) > 1:
            if (self.params['SE_proj'] and algoname[:3]=="GSE") or (algoname == 'OD-LinBAI' and self.params['OC_proj']):
                sample_df = self.proj_se(sample_df, self.params['OC_proj'])
                if stage_cntr==0:
                    thetahat = np.zeros(self.dim)

            X_train, y_train, X_test, y_test = \
                self.se_train_explore(sim_id, sample_df, stage_budgets[stage_cntr], num_plays_ind,
                                      km, not_elim_arms, _memory=_memory, pre_reward=pre_reward,
                                      pre_X_train=X_train, prev_y_train=y_train, optimal_design=optimal_design,
                                      params=params, thetahat=thetahat, algoname=algoname, stage_cntr=stage_cntr)
            if Lin:
                # ret = self.Lin(sim_id, self.X_train, self.X_test, self.y_train, self.y_test)
                ret = self._linear_reg(sim_id, X_train, X_test, y_train, y_test, _lambda)
                thetahat = ret['thetahat']
            elif 'Log' in params:
                ret = self._Logistic_reg(sim_id, X_train, X_test, y_train, y_test, _lambda)
            else:
                raise

            _mae, _mse, _r2s = _mae+ret['mae'], _mse+ret['mse'], _r2s+ret['r2s']

            pred_argsort = ret['pred'].argsort()
            middle = len(not_elim_arms)//eta
            # not_elim_arms = [i for c,i in enumerate(not_elim_arms)
            #                  if pred_argsort[c] >= middle] # eliminate with median
            # not_elim_arms = pred_argsort[middle:]
            if 'SD' in params:
                sample_df = sample_df.drop(not_elim_arms[pred_argsort[0]])
                del not_elim_arms[pred_argsort[0]]
            else:
                if algoname == 'OD-LinBAI':
                    not_elim_arms = [not_elim_arms[i] for i in pred_argsort[-int(np.ceil(self.dim/2**(stage_cntr+1))):]]
                else:
                    if middle > 0:
                        not_elim_arms = [not_elim_arms[i] for i in pred_argsort[-middle:]]
                    else:
                        not_elim_arms = [not_elim_arms[pred_argsort[-1]]]
                sample_df = sample_df.copy().loc[[i for i in not_elim_arms]]
            # sample_df.reset_index(inplace=True)
            stage_cntr += 1

        crct, simplereg = self.simple_regret(y, not_elim_arms)
        if sim_id % self.log_step == 0:
            print('{} sample id {}, km {}, memory {}, optimal_design {}, '
                  'eta {}, budget_per_arm {}, params: {}, correct {}'.format(algoname,
                sim_id, km, _memory, optimal_design, eta, self.budget_per_arm, params, crct))

        res = {}
        res['simple_regret'] = simplereg
        res['recoms'] = not_elim_arms[0]

        res['eta'] = eta
        if 0 and 'Log' in params:
            res['f1'], res['recall'], res['precision'] = f1/stage_cntr, recall/stage_cntr, precision/stage_cntr
        else:
            res['mae'], res['mse'], res['r2s'] = _mae/stage_cntr, _mse/stage_cntr, _r2s/stage_cntr
        res['acc'], res['num_plays_ind'], res['_lambda'] = crct, np.array(list(num_plays_ind.values())), _lambda
        # ret['mae'], ret['mse'], ret['r2s'], ret['acc'] = _mae / cntr, _mse / cntr, _r2s / cntr, crct
        return res

    def beta_bayesGap(self, beta_numerator, mu_hat, sigma2_hat):

        upbs, lowbs = mu_hat + 3 * sigma2_hat, mu_hat - 3 * sigma2_hat
        upbs_argmax = np.argmax(upbs)
        second_best = np.sort(upbs, axis=0)[-2]
        delta_hats = [upbs[upbs_argmax]-lowbs[i] for i in range(self.num_arms)]
        delta_hats[upbs_argmax] = second_best-lowbs[upbs_argmax]  # delta for the maximizer of upbs

        Heps = np.sum([max(1/2*(delta_hats[i]+self.BG_par['eps']),
                           self.BG_par['eps']) for i in range(self.num_arms)])
        beta = beta_numerator/Heps

        return beta

    def bayesGap(self, sample_df, sim_id):
        """
        On correlation and budget constraints in model-based bandit
        optimization with application to automatic machine learning, Hoffman, 2014

        :param sample_df:
        :param sim_id:
        :return:
        """

        tmpX = sample_df.drop([self.trgt_feat], axis=1).to_numpy()

        """Design matrix"""
        BG_kernel = self.params['BG_kernel']
        el = self.params['BG_kernel_l']
        if BG_kernel in ['exp', 'empirical']:
            if BG_kernel == 'exp':
                G = np.array([[np.exp(-norm(x-xp)**2/el) for x in tmpX] for xp in tmpX])
            else:
                G = np.cov(tmpX)
            try:
                D, V = eig(G)
                X = np.dot(V, np.diag(np.sqrt(D)))
            except np.linalg.LinAlgError:
                print('eig no convergence')
                X = tmpX
            # else:
            #     print('eig went wrong')
            #     X = tmpX
        elif BG_kernel == 'Matern':
            kernel = 1.0 * Matern(length_scale=el, nu=1.5)
            X = kernel(tmpX)
        elif BG_kernel == 'RBF':
            kernel = 1.0 * RBF(1.0)
            X = kernel(tmpX)
        elif BG_kernel == 'ExpSineSquared':
            kernel = ExpSineSquared(length_scale=1, periodicity=1)
            X = kernel(tmpX)
        elif BG_kernel == 'DotProduct':
            kernel = DotProduct() + WhiteKernel()
            X = kernel(tmpX)
        elif BG_kernel == 'RationalQuadratic':
            kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
            X = kernel(tmpX)
        else:
            X = tmpX

        Y = sample_df[self.trgt_feat].copy().to_numpy()  # Y or X^t thetha
        # crt_idx = sample_df[self.trgt_feat].idxmax()
        dim = X.shape[1]
        kappa = np.sum(1/norm(X, axis=1)**2)
        beta_numerator = (self.budget-self.num_arms)/self.sigma2+kappa/self.BG_par['eta']**2
        beta_numerator /= 4

        num_plays_ind = np.zeros(self.num_arms)
        # init
        Yt = []
        for a in range(self.num_arms):
            Yt.append(self.reward(mu=Y[a], sigma2=self.sigma2, size=1, rs=self.rs, config=X[a]))
            num_plays_ind[a] += 1
        init_sz = len(Yt)
        Yt = np.array(Yt).reshape(self.num_arms, 1)
        Xt = 1 * X

        if self.params['BG_GPtune']:
            tune_phase_size = int(self.budget/3)-self.num_arms
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=.4 ** 2, random_state=0)
            for _ in range(tune_phase_size):
                a = np.random.randint(low=0, high=self.num_arms)
                Yt = np.append(Yt, self.reward(mu=Y[a], sigma2=self.sigma2, size=1, rs=self.rs, config=X[a]))
                Xt = np.append(Xt, [X[a]], axis=0)
                num_plays_ind[a] += 1
            gpr.fit(Xt, Yt)
            l = gpr.kernel_.k2.get_params()['length_scale']
            sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

        histJt, histjt = [], []
        recom, recom_BJt = None, float('inf')

        for t in range(self.budget-self.num_arms):
            # print(f't {t}')
            Sigma_inv = np.dot(Xt.T, Xt)/self.sigma2 + np.eye(dim)/self.BG_par['eta']**2
            # Sigma = inv(Sigma_inv)
            # if np.isnan(Sigma).all():
            #     Sigma = inv(Sigma_inv+self.params['SE_lambda']*np.eye(Sigma_inv.shape[0]))
            Sigma = inv_wrap(Sigma_inv, self.params['SE_lambda'])
            theta_hat = np.dot(Sigma, np.dot(Xt.T, Yt))/self.sigma2
            mu_hat = np.dot(X, theta_hat).reshape(self.num_arms, 1)
            sigma2_hat = np.diagonal(np.dot(np.dot(X, Sigma), X.T)).reshape(self.num_arms, 1)
            Betat = self.beta_bayesGap(beta_numerator, mu_hat, sigma2_hat)
            U, L = mu_hat+Betat*sigma2_hat, mu_hat-Betat*sigma2_hat
            argmax_U = np.argmax(U)
            second_best = np.sort(U, axis=0)[-2]
            Bt = [U[argmax_U]-L[k] for k in range(self.num_arms)]
            Bt[argmax_U] = second_best-L[argmax_U]
            Jt = np.argmin(Bt)
            Utmp = 1*U
            Utmp[Jt]=np.min(U)-1
            jt = np.argmax(Utmp)
            histJt.append(Jt); histjt.append(jt)
            if Bt[Jt] < recom_BJt:
                recom, recom_BJt = Jt, Bt[Jt]

            at = Jt if sigma2_hat[Jt] >= sigma2_hat[jt] else jt
            Yt = np.append(Yt, self.reward(mu=Y[at], sigma2=self.sigma2, size=1, rs=self.rs, config=X[at]))
            num_plays_ind[at] += 1
            Xt = np.append(Xt, [X[at]], axis=0)

        if recom==None:
            print("recom is none")
            recom=0

        crct, simplereg = self.simple_regret(Y, [recom])
        if sim_id % 100 == 0:
            print('BayesGap sample id {} num arms {}, '
                  'budget_per_arm {}: corrct {}'.format(sim_id, self.num_arms, self.budget_per_arm, crct))

        ret = {}
        ret['recoms'] = recom
        ret['eta']=self.BG_par['eta']
        ret['simple_regret'] = simplereg
        ret['acc'], ret['num_plays_ind'] = crct, num_plays_ind
        return ret

    def ucbglm(self, sample_df, sim_id):
        """
        UCB-GLM from ''Provably Optimal Algorithms for Generalized Linear Contextual Bandits''
        Lihong Li

        :param sample_df:
        :param sim_id:
        :return:
        """
        alp = self.params['UCB_GLM_alpha']
        lamb = self.params['SE_lambda']
        UCB_GLM_mu_f = self.params['UCB_GLM_mu_f']
        X = sample_df.drop([self.trgt_feat], axis=1).to_numpy()
        y = sample_df[self.trgt_feat].copy().to_numpy()  # Y
        dim = X.shape[1]
        num_plays_ind = np.zeros(self.num_arms)
        assert self.num_arms == X.shape[0], "error in num arms vs X.shape"

        Yt = []
        for a in range(self.num_arms):
            Yt.append(self.reward(mu=y[a], sigma2=self.sigma2, size=1, rs=self.rs, config=X[a]))
            num_plays_ind[a] += 1
        Yt = np.array(Yt)

        V = np.dot(X.T, X)
        Xt = np.copy(X)
        for t in range(self.budget-self.num_arms):
            df = lambda theta: np.squeeze(np.dot((np.expand_dims(UCB_GLM_mu_f(np.dot(Xt, theta)), 1)-Yt).T, Xt), 0)
            try:
                theta = optimize.newton_krylov(df, np.zeros(dim))
                converged = True
            except NoConvergence as e:
                theta = e.args[0]
                converged = False
            except ValueError:
                try:
                    theta = optimize.fsolve(df, np.zeros(dim))
                    # print("fsolve")
                except ValueError:
                    raise
            # theta = np.ones(dim)

            Vinv = inv_wrap(V, lamb)
            Xv = np.array([np.dot(np.dot(xp.T, Vinv), xp) for xp in X])
            at = np.argmax(np.dot(X, theta)+alp*Xv)
            Yt = np.append(Yt, self.reward(mu=y[at], sigma2=self.sigma2, size=1, rs=self.rs, config=X[at]))
            Yt = np.expand_dims(Yt, 1)
            num_plays_ind[at] += 1
            x_at = X[at].reshape(dim, 1)
            Xt = np.append(Xt, x_at.T, axis=0)
            V += np.dot(x_at, x_at.T)

        recom = np.argmax(num_plays_ind)
        crct, simplereg = self.simple_regret(y, [recom])
        if sim_id%self.log_step ==0:
            print('UCB-GLM sample id {} num arms {}, '
                  'budget_per_arm {}: corrct {}'.format(sim_id, self.num_arms, self.budget_per_arm, crct))

        ret = {}
        ret['recoms'] = recom
        ret['simple_regret'] = simplereg
        ret['acc'], ret['num_plays_ind'] = crct, num_plays_ind
        return ret

    def linGapE_confidence_bound(self, x, A, t, reg, delta):
        L = 1
        tmp = np.sqrt(x.dot(np.linalg.inv(A)).dot(x))
        res = tmp * (self.sigma2 * np.sqrt(self.dim * np.log(self.num_arms**2 *
                                                             (1 + t * L**2) / reg / delta)) + np.sqrt(reg) * 2)
        return res

    def linGapE_decide_arm(self, y, A, X, it, jt, K, d, arm_selections, greedy):
        if greedy:
            tmp = [y.dot(np.linalg.inv(A + self.matrix_dot(x))).dot(y)
                   for x in X]
        else:
            # (12) in LinGapE paper
            fun = lambda w: np.linalg.norm(w, ord=1)
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w[:, np.newaxis] * X, axis=0) - y})
            bnds = list(repeat((0, None), K))
            # bnds = Bounds(0,None)
            min_res11 = minimize(fun, np.ones(K) / K, method='SLSQP', bounds=bnds, constraints=cons)
            wstar = min_res11.x
            pstar = np.abs(wstar) / np.sum(np.abs(wstar))
            tmp = [(arm_selections[i] / pstar[i] if pstar[i] > 0 else float('inf')) for i in range(K)]
        return np.argmin(tmp)

    def matrix_dot(self, a):
        return np.expand_dims(a, axis=1).dot(np.expand_dims(a, axis=0))

    def linGapE(self, sample_df, sim_id, greedy=0):
        """
        LinGapE algorithm from https://github.com/liyuan9988/LinGapE, author
        :return:
        """
        X = sample_df.drop([self.trgt_feat], axis=1).to_numpy()
        y = sample_df[self.trgt_feat].copy().to_numpy()  # Y
        d = X.shape[1]
        self.dim = d
        K = self.num_arms
        assert self.num_arms == X.shape[0], "error in num arms vs X.shape"

        reg = self.params['LinGapE']['reg']  # 1
        delta = self.params['LinGapE']['delta']  # 0.1
        # greedy = self.params['LinGapE']['greedy']
        epsilon = 2 * (1 - np.cos(self.params['synt1_omega']))
        sigma2 = self.sigma2

        A = np.eye(d) * reg
        b = np.zeros(d)
        # arm_selections = np.ones(K)
        num_plays_ind = np.ones(K)
        t = K
        for i in range(K):
            A += self.matrix_dot(X[i])
            # r = (self.theta.dot(X[i]) + np.random.randn() * sigma2)
            r = self.reward(y[i], sigma2, 1, rs=self.rs, config=X[i])
            b += X[i] * r

        theta_hat = np.linalg.solve(A, b)
        est_reward = X.dot(theta_hat)
        it = np.argmax(est_reward)
        jt = np.argmax(est_reward - est_reward[it] +
                       np.array([self.linGapE_confidence_bound(x - X[it], A, t, reg, delta) for x in X]))
        B = est_reward[jt] - est_reward[it] + self.linGapE_confidence_bound(X[it] - X[jt], A, t, reg, delta)

        while (self.fb > 0 and t < self.budget) or (self.fb == 0 and B > epsilon):
            a = self.linGapE_decide_arm(X[it] - X[jt], A, X, it, jt, K, d, num_plays_ind, greedy)
            A += self.matrix_dot(X[a])
            # b += X[a] * (self.theta.dot(X[a]) + np.random.randn() * sigma2)
            b += X[a] * self.reward(y[a], sigma2, 1, rs=self.rs, config=X[a])
            num_plays_ind[a] += 1
            t += 1
            theta_hat = np.linalg.solve(A, b)
            est_reward = X.dot(theta_hat)
            if (t % 5000 == 0):
                print(num_plays_ind)
                print(B)
                print(it, jt)
            it = np.argmax(est_reward)
            jt = np.argmax(est_reward - np.max(est_reward) +
                           np.array([self.linGapE_confidence_bound(x - X[it], A, t, reg, delta) for x in X]))
            B = est_reward[jt] - est_reward[it] + self.linGapE_confidence_bound(X[it] - X[jt], A, t, reg, delta)

        crct, simplereg = self.simple_regret(y, [it])
        if sim_id%self.log_step ==0:
            print(f"LinGapE {greedy} sample id {sim_id} num arms {self.num_arms}, "
                  f"budget_per_arm {self.budget_per_arm}: corrct {crct}")

        ret = {}
        ret['recoms'] = it
        ret['simple_regret'] = simplereg
        ret['acc'], ret['num_plays_ind'] = crct, num_plays_ind
        return ret

    def __call__(self, sim_id):
        # sim_id = X
        self._set_replic_seed(sim_id)
        ret = {}

        if self.params['random_exp']:
            # sample_df, _ = self.bootstrap(ss=self.num_arms, sim_id=sim_id)
            if self.rewards_base == "synt":
                sample_df = self._synt_data()
            else:
                sample_df = self.bootstrap(ss=self.num_arms, sim_id=sim_id)
        else:
            sample_df = self.df_train_fix

        """Well maintained algorithms"""
        # Lin
        if 'GSE-Lin' in self.algos:
            ret['GSE-Lin'] = self.suces_elim(sample_df, sim_id, Lin=1, _memory=0)
        if 'GSE-Lin-FWG' in self.algos:
            ret['GSE-Lin-FWG'] = self.suces_elim(sample_df, sim_id, _memory=0, Lin=1, params={'G-opt': 1})
        if 'GSE-Lin-Wynn' in self.algos:
            ret['GSE-Lin-Wynn'] = self.suces_elim(sample_df, sim_id, Lin=1, _memory=0, optimal_design=1)
        if 'GSE-Lin-Greedy' in self.algos:
            ret['GSE-Lin-Greedy'] = self.suces_elim(sample_df, sim_id, Lin=1, _memory=0, params={'Greedy': 1})
        # Log
        if 'GSE-Log' in self.algos:
            ret['GSE-Log'] = self.suces_elim(sample_df, sim_id, _memory=0, params={'Log': 1})
        if 'GSE-Log-FWG' in self.algos:
            ret['GSE-Log-FWG'] = self.suces_elim(sample_df, sim_id, _memory=0, params={'Log': 1, 'G-opt': 1})
        if 'GSE-Log-Wynn' in self.algos:
            ret['GSE-Log-Wynn'] = self.suces_elim(sample_df, sim_id, _memory=0, optimal_design=1, params={'Log': 1})
        if 'GSE-Log-Greedy' in self.algos:
            ret['GSE-Log-Greedy'] = self.suces_elim(sample_df, sim_id, _memory=0, params={'Log': 1, 'Greedy': 1})
        # Else
        if 'UCB-GLM' in self.algos:
            ret['UCB-GLM'] = self.ucbglm(sample_df, sim_id)
        if 'LinUCB' in self.algos:
            ret['LinUCB'] = self.linUCB_disjoint(sample_df, sim_id)
        if 'BayesGap' in self.algos:
            ret['BayesGap'] = self.bayesGap(sample_df, sim_id)
        if 'LinGapE' in self.algos:
            ret['LinGapE'] = self.linGapE(sample_df, sim_id)
        if 'LinGapE-Greedy' in self.algos:
            ret['LinGapE-Greedy'] = self.linGapE(sample_df, sim_id, greedy=1)
        if 'Peace' in self.algos:
            ret['Peace'] = self.suces_elim(sample_df, sim_id, Lin=1, _memory=0,
                                           params={'Peace': 1}, algoname='Peace')
        if 'OD-LinBAI' in self.algos:
            ret['OD-LinBAI'] = self.suces_elim(sample_df, sim_id, Lin=1, _memory=0,
                                               algoname='OD-LinBAI', params={'G-opt': 1})
        if 'GSE-Lin-FWG-1' in self.algos:
            ret['GSE-Lin-FWG-1'] = self.suces_elim(sample_df, sim_id, _memory=0, Lin=1,
                                                   params={'G-opt': 1}, algoname='GSE-Lin-FWG-1')
        if 'GSE-Lin-Todd' in self.algos:
            ret['GSE-Lin-Todd'] = self.suces_elim(sample_df, sim_id, _memory=0, Lin=1,
                                                  params={'G-opt': 1}, algoname='GSE-Lin-Todd')
        if 'GSE-Lin-Todd-1' in self.algos:
            ret['GSE-Lin-Todd-1'] = self.suces_elim(sample_df, sim_id, _memory=0, Lin=1,
                                                    params={'G-opt': 1}, algoname='GSE-Lin-Todd-1')

        # cal Deltas
        if self.params['random_exp']:
            for algo in self.algos:
                ret[algo]['deltas'] = self.get_deltas(sample_df[self.trgt_feat])

        return ret

    def get_results(self, poolobjs):
        print('Collecting the results ...')

        for algo in self.algos:
            res = {}
            res[algo] = {}
            algo_res = [i[algo] for i in poolobjs]
            # keys = algo_res[0].keys()
            for indic in ['acc', 'simple_regret', 'mae', 'mse', 'r2s', 'f1', 'recall', 'precision',
                          'eta', 'num_plays_ind']:
                # for indic in keys:
                if indic in algo_res[0]:
                    res[algo][indic] = sum([i[indic] for i in algo_res])/self.N_1

            res[algo]['sd acc'] = np.sqrt(res[algo]['acc']*(1-res[algo]['acc'])/self.N_1)
            res[algo]['sd simple_regret'] = np.std([i["simple_regret"] for i in algo_res])
            res[algo]['num_plays'] = sum(res[algo]['num_plays_ind'])
            tmp_recoms = Counter([i['recoms'] for i in algo_res])
            res[algo]['recoms'] = [tmp_recoms[i] for i in range(self.num_arms)]


            # Specific results
            if algo == 'LinUCB':
                res[algo]['alpha'] = self.alpha_linucb
            if 'BayesGap' in algo:
                res[algo]['kernel'] = self.params['BG_kernel']
                res[algo]['BG_kernel_l'] = self.params['BG_kernel_l']
            if 'embed' in algo:
                res[algo]['model'] = algo_res[0]['model']
            if 'GSE' in algo or 'SD' in algo:
                res[algo]['_lambda'] = algo_res[0]['_lambda']
            if algo == 'UCB-GLM':
                res[algo]['UCB_GLM_alpha'] = self.params['UCB_GLM_alpha']
                res[algo]['UCB_GLM_mu_f'] = self.params['UCB_GLM_mu_f']
            if algo in ['LinGapE', 'LinGapE-Greedy']:
                res[algo]['LinGapE_reg'] = self.params['LinGapE']['reg']
                res[algo]['LinGapE_delta'] = self.params['LinGapE']['delta']
                # res[algo]['LinGapE_greedy'] = self.params['LinGapE']['greedy']

            if self.rewards_base == 'aml':
                res[algo]['aml_model'] = self.params['aml_model']
                res[algo]['RMSE'] = 0
                for config, counts in res[algo]['recoms'].items():
                    res[algo]['RMSE'] += counts * self.aml.final_eval(rs=self.rs, config=self.df_train_fix.iloc[config])
                res[algo]['RMSE'] /= self.N_1

            row = {'algorithm': algo, 'N_1': self.N_1, 'num_arms': self.num_arms, 'dim': self.dim,
                   'max_depth': self.max_depth_dtr, 'sigma2': self.sigma2,
                   'budget_per_arm' if self.BperArm else "Budget": self.budget_per_arm}
            row.update(res[algo])
            if self.params['random_exp']:
                arms = 'random'
                theta = 'random'
                deltas_ = np.average([i["deltas"] for i in algo_res], axis=0)
            else:
                arms = self.arms
                theta = self.theta
                deltas_ = self.deltas

            if self.rewards_base=="synt":
                rewards_base_ = self.rewards_base + str(self.params['synt_num'])
                prior_sigma_ = self.params['prior_sigma']
            else:
                rewards_base_ = self.rewards_base
                prior_sigma_ = ""
            row.update({'rewards_base': rewards_base_, 'theta': theta, 'arms': arms,
                        'synt1_omega': self.params['synt1_omega'],
                        'time': get_pst_time(), 'prior_sigma': prior_sigma_,
                        'optimal_design_eps': self.params['optimal_design_eps'],
                        'deltas': deltas_,
                        'SE_proj': self.params['SE_proj'], 'SE_proj_tol': self.params['SE_proj_tol'],
                        'SE_lambda': self.params['SE_lambda'], 'SE_dim': self.SE_dim, 'params': self.params,
                        'BperArm':self.BperArm})
            file_name = 'dtr_res'
            print(row)
            write_csv(list(row.keys()), list(row.values()), file_name)

        return res

    def exp_dtr(self):

        if self.multithr:
            if platform.system() == 'Linux':
                # for solving the freeze in linux
                try:
                    mp.set_start_method('spawn')
                except RuntimeError:
                    pass
                # mp.set_start_method("spawn", force=True)
            rr = [(sim_id) for sim_id in range(self.N_1)]
            num_cpu = mp.cpu_count()
            # num_cpu = 6
            with mp.Pool(num_cpu) as pool:
                poolobjs = pool.map(self, rr)
                # print(poolobjs)
                poolobjs = np.array(poolobjs)
        else:
            poolobjs = []
            for sim_id in range(self.N_1):
                poolobjs.append(self((sim_id)))

        self.get_results(poolobjs)

        # print(self.SE_dim)


def BAImain():
    print(os.getcwd())

    N_1 = 1000

    max_depth_dtr = 20
    multithr = 1
    multithr = 0

    reward_dist = 'Gaus'

    rewards_base = 'auto'
    rewards_base = 'pmsm'
    rewards_base = 'synt'
    # rewards_base = 'aml'

    aml_model = 'random_forest'
    """synt data"""
    random_exp = True
    # random_exp = False

    prior_sigma = 1
    BperArm = 0  # budget per arm or total budget  ######
    BperArm = 1  # budget per arm or total budget  ######
    budget_per_arm = [10, 20, 50]
    budget_per_arm = [20, 50]
    # budget_per_arm = [20]
    # budget_per_arm = [2*2**6]
    # budget_per_arm = [5**3]
    budget_per_arm = [50]
    if rewards_base in ['auto', 'pmsm']:
        sigma2 = .1
    else:
        sigma2 = 10
        # sigma2 = 5
        # sigma2 = 1

    synt_num, dims, num_arms = 1, [7, 15, 31, 63, 127], [8]  # synt1, hard instance of BAI, k is set =d+1
    # synt_num, dims, num_arms = 3, [5, 7, 10, 12], [8]  # synt3 uniform covar, normal theta, logistic
    # synt_num, dims, num_arms = None, [None], [8, 16]  # auto, pmsm
    # synt_num, dims, num_arms = 5, [16], [16]  # synt 5, LinGapE 2018, Delta example, d=k, k is set =d
    delta_s5 = [.01, .03, .05, .1, .3]
    # delta_s5 = [.1]
    synt_num, dims, num_arms,  = 6, [4], [8, 16, 32, 64]  # synt 6, ALBA 2018 k=100 in paper,
    alpha_s6 = 0.01
    OD_LinBAIpaper = 0  # use to get the OD-LinBAI setting
    # synt_num, dims, num_arms = 7, [2], [8, 16, 64]  # synt 7, OD-LinBAI 2021 paper
    if synt_num != 5:
        delta_s5 = [-1]

    if rewards_base == 'synt' and synt_num == 3:
        algos = ['GSE-Log', 'GSE-Log-FWG',  # 'GSE-Log-Greedy', 'GSE-Log-Wynn',
                 'BayesGap', 'GSE-Lin', 'GSE-Lin-FWG',  # 'GSE-Lin-Greedy', 'GSE-Lin-Wynn',
                 'UCB-GLM']  # synt 3
        # algos = ['BayesGap']
        # algos = ['GSE-Log-FWG', 'UCB-GLM']
        # algos = ['UCB-GLM']
    else:
        algos = ['GSE-Lin', 'GSE-Lin-FWG', 'GSE-Lin-Greedy', 'GSE-Lin-Wynn',
                 'LinUCB', 'BayesGap', 'LinGapE', 'Peace']  # , 'LinGapE-Greedy']  # synt 1 2 4 5 6 auto, pmsm
        # algos = ['GSE-Lin', 'GSE-Lin-FWG']
        # algos = ['GSE-Lin', 'LinUCB', 'BayesGap']
        # algos = ['BayesGap']
        # algos = ['LinGapE', 'LinGapE-Greedy']
        # algos = ['LinGapE']
        algos = ['Peace']
        # algos = ['OD-LinBAI', 'GSE-Lin-FWG']
        # algos = ['OD-LinBAI']
        # algos = ['GSE-Lin-FWG']
        # algos = ['GSE-Lin-FWG-1']
        # algos = ['GSE-Lin-FWG', 'GSE-Lin-FWG-1']
        # algos = ['GSE-Lin-FWG', 'GSE-Lin-FWG-1', 'OD-LinBAI', 'GSE-Lin-Todd', 'GSE-Lin-Todd-1']
        # algos = ['GSE-Lin-FWG', 'OD-LinBAI', 'GSE-Lin-Todd']
        # algos = ['GSE-Lin-FWG', 'LinUCB', 'BayesGap', 'LinGapE-Greedy', 'OD-LinBAI']#, 'Peace']
        # algos = ['LinGapE-Greedy', 'OD-LinBAI']

    # SE pars
    eta_SE = 2
    # eta_SE = 125
    SE_lambda = 1e-8
    SE_proj = 1
    # SE_proj = 0
    SE_proj_tol = 1e-7

    # G-opt Wynn
    optimal_design_eps = 1e-2
    optimal_design_eps = 1e-3
    # optimal_design_eps = 1e-4
    # optimal_design_eps = 1e-5

    # GPUCB
    gpucb_kernel = 'Sq_Exp'  # 'lin'
    gpucb_delta = .01

    # UCB-GLM
    UCB_GLM_alpha = 1/2
    UCB_GLM_mu_f = sigmoid

    # LinGapE
    LinGapE_reg = SE_lambda
    LinGapE_delta = .5
    LinGapE_delta = .9
    LinGapE_delta = .4
    # LinGapE_delta = .1
    # LinGapE_delta = .01
    # LinGapE_greedy = 1
    # LinGapE_greedy = 0


    # BayesGap_pars
    BG_eta = 50
    BG_eps = 0
    BG_kernel = ''
    # BG_kernel = 'empirical'
    BG_kernel = 'exp'
    # BG_kernel = 'Matern'
    # BG_kernel = 'RBF'
    # BG_kernel = 'ExpSineSquared'
    # BG_kernel = 'DotProduct'
    # BG_kernel = 'RationalQuadratic'
    BG_kernel_l = [10]
    # BG_kernel_l = [1e-6, 1e-3, .1, 1, 10, 1e2]
    # BG_kernel_l = [1e3, 5e3, 1e4, 1e5, 1e6]
    # BG_kernel_l = [1e-7, 1e-8, 1e-9]
    # BG_kernel_l = [200,500,700]
    if BG_kernel not in ['exp', 'Matern']:
        BG_kernel_l = [-1]
    BG_GPtune = 0

    OC_proj = 1
    OC_cons = 1

    if len(sys.argv) > 1:
        # For cluster runs
        cntr_1 = iter(range(4, 1000))
        cntr = iter(range(1, len(sys.argv)))
        multithr = 1
        multithr = 0
        N_1 = int(sys.argv[next(cntr)])
        num_arms = [int(sys.argv[next(cntr)])]
        budget_per_arm = [int(sys.argv[next(cntr)])]
        if len(sys.argv) > next(cntr_1):
            sigma2 = float(sys.argv[next(cntr)])
        if len(sys.argv) > next(cntr_1):
            rewards_base = sys.argv[next(cntr)]
        if len(sys.argv) > next(cntr_1):
            dims = [int(sys.argv[next(cntr)])]
        if len(sys.argv) > next(cntr_1):
            prior_sigma = float(sys.argv[next(cntr)])
        if len(sys.argv) > next(cntr_1):
            synt_num = int(sys.argv[next(cntr)])
        if len(sys.argv) > next(cntr_1):
            random_exp = int(sys.argv[next(cntr)])
        if len(sys.argv) > next(cntr_1):
            delta_s5 = [float(sys.argv[next(cntr)])]
        if len(sys.argv) > next(cntr_1):
            BperArm = [int(sys.argv[next(cntr)])]
        algos = []
        while len(sys.argv) > next(cntr_1):
            algos += [sys.argv[next(cntr)]]

        # grid search optimal kernels I found
        if rewards_base == 'synt':
            if synt_num in [1]:
                BG_kernel = 'exp'
                BG_kernel_l = [1e-3]
            elif synt_num in [3]:
                # BG_kernel = 'exp'
                # BG_kernel_l = [1e6]
                BG_kernel = 'Matern'
                BG_kernel_l = [10]
            elif synt_num in [6]:
                BG_kernel = 'Matern'
                BG_kernel_l = [10]
            elif synt_num in [5]:
                BG_kernel = 'exp'
                BG_kernel_l = [1000]
        elif rewards_base == 'auto':
            BG_kernel = 'exp'
            BG_kernel_l = [1e-6]
        elif rewards_base == 'pmsm':
            BG_kernel = 'empirical'
            BG_kernel_l = [-1]

    if synt_num == 3:
        reward_dist = 'Bern'
    if rewards_base in ['auto', 'pmsm']:
        OC_proj = 0
        SE_proj = 0

    make_dir('./result')

    for algo, num_arm, b, d, BG_l, dls5 in product(algos, num_arms, budget_per_arm, dims, BG_kernel_l, delta_s5):
        if synt_num == 6 and OD_LinBAIpaper:
            num_arm = 2**d
            b = 2 * num_arm
        st = time.time()
        exp_class1 = super_class(seed_=110, N_1=N_1, num_arms=num_arm, dim=d, reward_dist=reward_dist,
                                 max_depth_dtr=max_depth_dtr, budget_per_arm=b, sigma2=sigma2, fb=1,
                                 multithr=multithr, rewards_base=rewards_base,
                                 BG_par={'eta': BG_eta, 'eps': BG_eps}, algos=[algo],  # algos=algos
                                 GPUCB_par={'kernel': gpucb_kernel, 'delta': gpucb_delta},
                                 params={'SE_eta': eta_SE, 'SE_lambda': SE_lambda, 'synt_num': synt_num,
                                         'prior_sigma': prior_sigma, 'random_exp': random_exp,
                                         'optimal_design_eps': optimal_design_eps, 'delta_s5': dls5,
                                         'alpha_s6': alpha_s6, 'BG_kernel': BG_kernel, 'BG_kernel_l':BG_l, 'BG_GPtune':BG_GPtune,
                                         'SE_proj': SE_proj, 'SE_proj_tol':SE_proj_tol, 'UCB_GLM_alpha':UCB_GLM_alpha,
                                         'UCB_GLM_mu_f':UCB_GLM_mu_f,
                                         'LinGapE': {'reg': LinGapE_reg, 'delta': LinGapE_delta},
                                         'aml_model': aml_model, 'OC_proj': OC_proj, 'OC_cons': OC_cons}, BperArm=BperArm)
        # exp_class1.read_data()
        # tmp = auto_class1.bootstrap()
        exp_class1.exp_dtr()


if __name__ == "__main__":
    BAImain()

