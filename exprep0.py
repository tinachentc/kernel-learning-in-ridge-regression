'''Main function'''

import torch
import numpy as np
from trainSigArm_Gau import *
from trainSigArm_transinv import *
from trainSigArm_innerprod import *
import pickle
import pandas as pd
import argparse

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
    return torch.from_numpy(X).float(), torch.from_numpy(y.reshape(n, 1)).float()

# function for each repetition
def main(argwrap, data=None):
    if data is not None:
        lam_seq0, ker, iters, lr, alpha, beta, tol, cl = argwrap

        lam_seq = lam_seq0.split(',')
        lam_seq = [float(val) for val in lam_seq]
        data_seq = data.split(',')
        if len(data_seq)<=1:
            print('Missing data path!')
            return
        else:
            df_tr = pd.read_csv(data_seq[0], header=0)
            df_tr = torch.tensor(df_tr.values, dtype=torch.float32)
            X_tr = df_tr[:, :-1]
            y_tr = df_tr[:, -1].view(-1,1)
            df_te = pd.read_csv(data_seq[1], header=0)
            df_te = torch.tensor(df_te.values, dtype=torch.float32)
            X_te = df_te[:, :-1]
            y_te = df_te[:, -1].view(-1,1)
            if len(data_seq)>=3:
                df_val = pd.read_csv(data_seq[2], header=0)
                df_val = torch.tensor(df_val.values, dtype=torch.float32)
                X_val = df_val[:, :-1]
                y_val = df_val[:, -1].view(-1, 1)
                n, d = X_tr.shape
                name = data_seq[0][:-4] + '_TrTeVal' + '_' + str(ker) + '_n' + str(n) + '_d' +str(d) + '_iter' +str(iters) + '_alpha' +str(alpha) + '_beta' +str(beta) + '_tol' +str(tol) + '_lr' +str(lr) + '_l' + str(lam_seq[0]) + '_' + str(lam_seq[-1])
            else:
                X_val = None
                y_val = None
                n, d = X_tr.shape
                name = data_seq[0][:-4] + '_TrTe' + '_' + str(ker) + '_n' + str(n) + '_d' +str(d) + '_iter' +str(iters) + '_alpha' +str(alpha) + '_beta' +str(beta) + '_tol' +str(tol) + '_lr' +str(lr) + '_l' + str(lam_seq[0]) + '_' + str(lam_seq[-1])
            if cl == 1:
                name = name + '_cl'

    else:
        seed, lam_seq0, ker, args, xdis, n, d, e, rho, c, iters, lr, alpha, beta, tol, cl = argwrap

        lam_seq = lam_seq0.split(',')
        lam_seq = [float(val) for val in lam_seq]
        name = './' + 'sp_simulation' + str(args) + str(xdis) + '_' + str(ker) + '_seed' + str(seed) + '_n' + str(n) + '_d' +str(d) + '_e' +str(e) + '_r' +str(rho) + '_c' +str(c) + '_iter' +str(iters) + '_alpha' +str(alpha) + '_beta' +str(beta) + '_tol' +str(tol) + '_lr' +str(lr) + '_l' + str(lam_seq[0]) + '_' + str(lam_seq[-1])
        if cl == 1:
            name = name + '_cl'

        #simulation parameter
        X_tr, y_tr = RGdata(n, d, e, rho, c, args, xdis, cl, seed)
        val = 0.5
        n_val = int(n * val)
        X_te, y_te = RGdata(n_val, d, e, rho, c, args, xdis, cl, seed * 10000)
        X_val = None
        y_val = None

    #optimization parameters
    bs = len(y_tr)

    sav = pd.DataFrame(columns=['testerr', 'lambda', 'rank', 'rank_seq', 'obj', 'tval', 'val', 'M'])

    d0 = d
    if ker == 'Gaudiag':
        M = np.ones(d, dtype='float32') / d
    else:
        M = np.eye(d, dtype='float32') / d
    for i in range(len(lam_seq)):
        lam = lam_seq[i]
        reg = lam * bs

        #kernel option
        if ker == 'Gau':
            terr, M, obj_seq, tval_seq, val_seq, rank_seq = trainSig(X_tr, y_tr, X_te, y_te, M, d0,
                                                               iters=iters, batch_size=bs, reg=reg, lr0=lr,
                                                               alpha=alpha, beta=beta, tol=tol, X_val=X_val, y_val=y_val, classification=cl)

        elif ker == 'Gaudiag':
            terr, M, obj_seq, tval_seq, val_seq, rank_seq = trainSig_diag(X_tr, y_tr, X_te, y_te, M, d0,
                                                               iters=iters, batch_size=bs, reg=reg, lr0=lr,
                                                               alpha=alpha, beta=beta, tol=tol, X_val=X_val, y_val=y_val, classification=cl)

        elif ker == 'IMQ':
            terr, M, obj_seq, tval_seq, val_seq, rank_seq = trainSig_IMQ(X_tr, y_tr, X_te, y_te, M, d0,
                                                               iters=iters, batch_size=bs, reg=reg, lr0=lr,
                                                               alpha=alpha, beta=beta, tol=tol, X_val=X_val, y_val=y_val, classification=cl)

        elif ker == 'MT':
            terr, M, obj_seq, tval_seq, val_seq, rank_seq = trainSig_MT(X_tr, y_tr, X_te, y_te, M, d0,
                                                               iters=iters, batch_size=bs, reg=reg, lr0=lr,
                                                               alpha=alpha, beta=beta, tol=tol, X_val=X_val, y_val=y_val, classification=cl)

        elif ker == 'linear':
            terr, M, obj_seq, tval_seq, val_seq, rank_seq = trainSig_linear(X_tr, y_tr, X_te, y_te, M, d0,
                                                               iters=iters, batch_size=bs, reg=reg, lr0=lr,
                                                               alpha=alpha, beta=beta, tol=tol, X_val=X_val, y_val=y_val, classification=cl)

        elif ker == 'cubic':
            terr, M, obj_seq, tval_seq, val_seq, rank_seq = trainSig_cubic(X_tr, y_tr, X_te, y_te, M, d0,
                                                               iters=iters, batch_size=bs, reg=reg, lr0=lr,
                                                               alpha=alpha, beta=beta, tol=tol, X_val=X_val, y_val=y_val, classification=cl)

        ran = rank_seq[-1]
        obj = obj_seq[-1]
        tval = tval_seq[-1]
        val = val_seq[-1]
        sav = pd.concat([sav, pd.DataFrame([{'testerr': terr, 'lambda': lam, 'rank': ran, 'rank_seq': rank_seq, 'obj': obj, 'tval': tval, 'val': val, 'M': M}])], ignore_index=True)#sav = pd.concat([sav, pd.DataFrame([{'lambda': lam, 'rank': ran, 'M': M, 'obj_seq': obj_seq, 'tval_seq': tval_seq, 'val_seq': val_seq, 'rank_seq': rank_seq}])], ignore_index=True)
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(sav, f)
        d0 = ran
    print(sav[['lambda','rank']])

#interactive arguments
def parse_commandline():
    """Parse the arguments given on the command-line.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s", help="seed", default=1)
    # model arguments
    parser.add_argument("--l", help="lambda sequence using comma to separate (e.g 0.1,0.2,0.3)", type=str)
    parser.add_argument("--ker", help="kernel (Gau/IMQ/MT/linear/cubic, default='Gau')", type=str, default='Gau')
    # data arguments
    parser.add_argument("--data", help="csv data path with last column y, using comma to separate train and test path (train,test/train,test,validate, default=None)", type=str, default=None)
    parser.add_argument("--args", help="simulation model number (1sum/2prod/3tanh/4uptocubic/5emp, default=3)", type=int, default=3)
    parser.add_argument("--xdis", help="X distribution number (1Gau/2Unif/3Ber, default=1)", type=int, default=1)
    parser.add_argument("--n", help="sample size (type=int, default=300)", type=int, default=300)
    parser.add_argument("--d", help="sample dimension (type=int, default=50)", type=int, default=50)
    parser.add_argument("--e", help="noise size (type=float, default=0.1)", type=float, default=0.1)
    parser.add_argument("--rho", help="parameter for X distribution: correlation for 1Gau and probability equaling one for 3Ber (type=float, default=0.)", type=float, default=0.)
    parser.add_argument("--c", help="sample intercept (type=float, default=0.)", type=float, default=0.)
    # optimization arguments
    parser.add_argument("--iter", help="number of epoches (type=int, default=2000)", type=int, default=2000)
    parser.add_argument("--lr", help="initial learning rate (type=float, default=0.1)", type=float, default=0.1)
    parser.add_argument("--alpha", help="Armijo alpha (type=float, default=0.001)", type=float, default=0.001)#0.01
    parser.add_argument("--beta", help="Armijo beta (type=float, default=0.5)", type=float, default=0.5)
    parser.add_argument("--tol", help="tolerance (type=float, default=0.001)", type=float, default=0.001)
    # classification argument
    parser.add_argument("--cl", help="whether a classification problem (1/0, default=0)", type=int, default=0)
    args = parser.parse_args()
    return args

cmd = parse_commandline()
s = cmd.s
# model arguments
lam_seq0 = cmd.l
ker = cmd.ker
# data arguments
data = cmd.data
if data is None:
    args = cmd.args
    xdis = cmd.xdis
    n = cmd.n
    d = cmd.d
    e = cmd.e
    rho = cmd.rho
    c = cmd.c
# optimization arguments
iters = cmd.iter
lr = cmd.lr
alpha = cmd.alpha
beta = cmd.beta
tol = cmd.tol
# classification argument
cl = cmd.cl

s = int(s)
if data is None:
    main((s, lam_seq0, ker, args, xdis, n, d, e, rho, c, iters, lr, alpha, beta, tol, cl))
else:
    main((lam_seq0, ker, iters, lr, alpha, beta, tol, cl), data=data)