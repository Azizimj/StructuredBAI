#!/usr/bin/env python

"""
Job creator
"""
__author__ = "M.J. Azizi"
__copyright__ = "Copyright 2021, USC"

import os, time, sys
from itertools import product
from utils import make_dir


def main_StBAI():

    jobs_, cmds = [], []
    app = 'StructBAI.py'
    pyhandle = 'python'
    n_1 = 1000
    bperArm = 1
    OD_LinBAIpaper = 0

    params = {}
    params['rewards_base'] = 'synt'
    # params['rewards_base'] = 'auto'
    # params['rewards_base'] = 'pmsm'

    params['synt_num'] = 1

    if len(sys.argv) > 1:
        params['rewards_base'] = sys.argv[1]
        params['synt_num'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        OD_LinBAIpaper = int(sys.argv[3])

    if params['rewards_base'] == 'synt':
        params['sigma2s'] = [10]
    else:
        params['sigma2s'] = [.1]  # auto, pmsm
    params['prior_sigmas'] = [1]

    if params['rewards_base'] in ['auto', 'pmsm']:
        params['num_arms'] = [8, 16]
        params['dims'] = [-1]  # auto, pmsm
        params['synt_num'] = -1
    elif params['synt_num'] in [1, 2, 6]:
        params['num_arms'] = [8, 16, 32, 64, 128]  # synt 1, 2, 6
        params['dims'] = [10]
    elif params['synt_num'] in [3]:
        params['num_arms'] = [8]  # synt 3
        params['dims'] = [5, 7, 10, 12]  # synt 4, 3
    elif params['synt_num'] in [5]:
        params['num_arms'] = [3]  # synt 5
        params['dims'] = [16]  # synt 5

    if (params['rewards_base'] in ['auto', 'pmsm']) or (params['synt_num'] in [1, 2, 3]):
        params['bs'] = [10, 20, 50]  # synt 1, 2, 3, auto, pmsm
    elif params['synt_num'] in [5]:
        params['bs'] = [20]  # synt 5
    elif params['synt_num'] in [6]:
        params['bs'] = [20, 50]  # synt 6
        params['dims'] = [10]  # synt 6

    params['delta_s5'] = [.01, .03, .05, .1, .3]
    if 5 != params['synt_num']:
        params['delta_s5'] = [1]

    random_exp = 1

    # algos
    if params['synt_num'] == 3:
        params['algos'] = ['GSE-Log-FWG',
                           'BayesGap', 'UCB-GLM',
                           'GSE-Lin-FWG']
    else:
        params['algos'] = ['GSE-Lin-FWG', 'GSE-Lin-Greedy', 'GSE-Lin-Wynn',
                           'LinUCB', 'BayesGap', 'LinGapE', 'Peace', 'OD-LinBAI']

    if params['synt_num'] == 6 and OD_LinBAIpaper:
        n_1 = 1024
        params['num_arms'] = [8, 16, 32, 64]
        params['dims'] = [10]
        params['sigma2s'] = [1]
        params['bs'] = [100, 200, 500]
        bperArm = 0
        params['algos'] = ['GSE-Lin-FWG', 'OD-LinBAI']

    if params['synt_num'] == 7:
        params['num_arms'] = range(3, 25)
        params['dims'] = [2]
        params['bs'] = [25]
        bperArm = 0
        params['algos'] = ['GSE-Lin-FWG', 'OD-LinBAI']
        params['sigma2s'] = [1]

    print(params)

    for num_arm, d, algo, sig, psig, b in \
            product(params['num_arms'],
                    params['dims'], params['algos'], params['sigma2s'], params['prior_sigmas'],
                    params['bs']):
        reward = params['rewards_base']
        synt_num = params['synt_num']
        for ds5 in params['delta_s5']:
            cmd_ = ' '.join([str(jj) for jj in [pyhandle, app, n_1, num_arm, b, sig, reward,
                                                d if synt_num != 1 else num_arm - 1, psig, synt_num,
                                                random_exp, ds5, bperArm, algo]])
            jobs_.append(cmd_)

    make_dir("log_")
    f_jobs = open("log_/jobs.txt", "a")
    f_jobs.write("#############################\n")
    f_jobs.write(str(params))
    f_jobs.write("\n")
    print(f"Num jobs {len(jobs_)}")
    for cmd_ in jobs_:
        print(cmd_)
        os.system(cmd_)
    f_jobs.close()


if __name__ == '__main__':
    main_StBAI()
