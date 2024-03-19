import sys
sys.path.append("/home/ycm9079/code_py/upload")

import torch
import numpy as np
from temptrainSigArm_Gau2 import *
import pickle
import pandas as pd
import argparse

import matplotlib.pyplot as plt

#simulation
def RGdata(n, d, e, rho, c, args, xdis, cl, seed=1):
    # noise
    np.random.seed(seed)
    u = e*np.random.normal(0, 1, size=n)

    # distribution of X
    if xdis == 1:
        np.random.seed(seed * 200)
        s = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                s[i][j] = rho ** abs(i - j)
        X = np.random.multivariate_normal(np.zeros(d), s, size=n)
    elif xdis == 2:
        np.random.seed(seed * 200)
        X = np.random.uniform(0, 1, size=(n, d))
    elif xdis == 3:
        np.random.seed(seed * 200)
        X = np.random.choice([0, 1], size=(n, d), p=[1 - rho, rho])

    # distribution of y
    if args == 1:
        y0 = X[:, 0] + X[:, 1] + X[:, 2]
    elif args == 2:
        y0 = X[:, 0] * X[:, 1]
    elif args == 3:
        y0 = 0.1 * (X[:, 0] + X[:, 1] + X[:, 2])**3 + np.tanh(X[:, 0] + X[:, 2] + X[:, 4])
    elif args == 4:
        y0 = 2 * X[:, 0] + 2 * X[:, 1] + (X[:, 1] + X[:, 2])**2 + (X[:, 3] - 0.5)**3
    elif args == 5:
        y0 = 0
    y = y0 + u + c
    if cl == 1:
        y = (y > 0.5).astype(int)
    yy = torch.from_numpy(y.reshape(n, 1))#.float()
    if cl == 2:
        yy = np.zeros((len(y),3))
        yy[y<-1] = [1,0,0]
        yy[(y>=-1) & (y<=1)] = [0,1,0]
        yy[y>1] = [0,0,1]
        yy = torch.from_numpy(yy)#.float()
    return torch.from_numpy(X).float(), yy


n = 300
d = 50
e = 0.1
rho = 0.
c = 0.
args = 1
xdis = 1
cl = 2

X_tr, y_tr = RGdata(n, d, e, rho, c, args, xdis, cl)

#M = np.eye(d, dtype='float32') / d#M = torch.eye(d) / d#
M = np.ones(d, dtype='float32') / d
d0 = d
iters = 5
batch_size = n
reg = n*1
lr0 = 0.1
alpha = 0.001
beta = 0.5
tol = 0.001
y_tr = y_tr.float()
terr, M, obj_seq, tval_seq, val_seq, rank_seq = trainSig_diag(X_tr, y_tr, X_tr, y_tr, M, d0, iters=iters, batch_size=batch_size, reg=reg, lr0=lr0,
                                                         alpha=alpha, beta=beta, tol=tol, X_val=X_tr, y_val=y_tr, classification=1)

X_val = X_test = X_train = X_tr
y_val = y_test = y_train = y_tr#.float()
classification=0; unfold=0