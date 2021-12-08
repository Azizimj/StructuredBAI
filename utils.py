#!/usr/bin/env python

"""
Utilities
"""
__author__ = "M.J. Azizi"
__copyright__ = "Copyright 2021, USC"

import os, csv, numpy as np
from datetime import datetime
from pytz import timezone, utc
from numpy.linalg import norm


def make_dir(dir):
    if not os.path.exists(dir):
        print("dir ( {} ) is made under {}".format(dir, os.getcwd()))
        os.mkdir(dir)


list2str = lambda s: ', '.join(map(str, s))


def write_csv(rowTitle, row, file_name='res'):
    # print('csv Title {}\n row {}'.format(rowTitle, row))
    make_dir('result')
    with open('result/{}.csv'.format(file_name), 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(rowTitle)
        writer.writerow(row)
    csvFile.close()


intersection = lambda lst1, lst2: [value for value in lst1 if value in lst2]


def get_pst_time():
    date_format='%m-%d-%Y--%H-%M-%S-%Z'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)
    # pstDateTime = pd.Timestamp.today()
    return pstDateTime


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def inv_wrap(X, lamb):
    dim = X.shape[0]
    try:
        tmp = np.linalg.inv(X)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            tmp = np.linalg.inv(X + lamb*np.eye(dim))
        else:
            raise
    return tmp


def find_two_closest(arms):
    # most distant points:
    x, y, delta, idx1, idx2 = arms[0], arms[1], norm(arms[0]-arms[1]), 0, 1
    for i, element in enumerate(arms):
        for j, sec_element in enumerate(arms):
            if i == j:
                continue
            tmp_delta = norm(sec_element - element)
            if tmp_delta < delta:
                x, y, delta, idx1, idx2 = sec_element, element, tmp_delta, i, j
    return x, y, idx1, idx2