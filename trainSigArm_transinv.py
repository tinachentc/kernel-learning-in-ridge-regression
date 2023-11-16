'''Gradient function and PGD algorithm (other translation invariant kernels: inverse multiquadric and Matern)'''

import numpy as np
import torch
from scipy.linalg import eigh
from numpy.linalg import solve, inv, norm
import kernellab
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# inverse multiquadric kernel
def get_grads_IMQ(reg, X, Y, L, P, batch_size, unfold=0):
    d, _ = P.shape
    n, c = Y.shape
    batches_X = torch.split(X, batch_size)
    batches_Y = torch.split(Y, batch_size)
    G = torch.zeros(d, d)
    for i in tqdm(range(len(batches_X))):
        x = batches_X[i]
        y = batches_Y[i]
        m, _ = x.shape
        K = kernellab.IMQ_M(x, x, L, P)

        # calculate alpha and diff
        proj = torch.eye(m)-torch.ones((m,m))/m
        a = solve(proj@K + reg * np.eye(m), proj@y)
        a = torch.from_numpy(a).float()
        diff = x.unsqueeze(1) - x.unsqueeze(0)

        # for IMQ
        K.pow_(3)

        # separate unfold cases
        if unfold == 0:
            res = torch.einsum('ijk,ijl->ijkl', diff, diff)
            C = torch.mul(res, K.view(m,m,1,1))
            del res
            G0 = torch.tensordot(a, C, dims=([0], [0]))
            del C
            G += torch.sum(torch.tensordot(torch.sum(G0, dim=0), a, dims=([0], [0])), dim=2)
            del G0

        elif unfold == 1:
            for j in range(d):
                resj = torch.einsum('ijk,ijl->ijkl', diff, diff[:, :, j:(j + 1)]).squeeze()
                Cj = torch.mul(resj, K.view(m,m,1))
                del resj
                Gj = torch.tensordot(a, Cj, dims=([0], [0]))
                del Cj
                G[j] += torch.sum(torch.tensordot(torch.sum(Gj, dim=0), a, dims=([0], [0])), dim=1)
                del Gj

        else:
            for j in range(d):
                for k in range(d):
                    resjk = torch.einsum('ijk,ijl->ijkl', diff[:, :, j:(j + 1)], diff[:, :, k:(k + 1)]).squeeze()
                    Cjk = torch.mul(resjk, K)
                    del resjk
                    Gjk = torch.tensordot(a, Cjk, dims=([0], [0]))
                    G[j,k] += torch.sum(torch.tensordot(torch.sum(Gjk, dim=0), a, dims=([0], [0])), dim=0)
                    del Gjk

    G *= reg/n/2/L
    G = G.numpy()
    return G

def trainSig_IMQ(X_train, y_train, X_test, y_test, M, d0,
        iters, batch_size, reg, lr0,
        alpha, beta, X_val=None, y_val=None, classification=0, unfold=0):
    L = 1
    n, d = X_train.shape
    mm, d = X_test.shape
    if X_val is not None and y_val is not None:
        m, d = X_val.shape

    U = np.eye(d, dtype='float32')
    obj_seq = np.zeros(iters+1)
    err_seq = np.zeros(iters+1); terr_seq = np.zeros(iters+1)
    rank_seq = np.zeros(iters+1)

    proj = torch.eye(n) - torch.ones((n, n)) / n


    i = 0
    # rank
    rank_seq[i] = d0

    # training err
    K_train = kernellab.IMQ_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
    preds0 = (sol @ K_train).T
    csol = np.mean(y_train.numpy() - preds0)
    preds = preds0 + np.ones((n, 1)) * csol
    terr_seq[i] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
    print("Round " + str(i) + " Train RMSE: ", terr_seq[i])

    # validation err
    if X_val is not None and y_val is not None:
        K_val = kernellab.IMQ_M(X_train, X_val, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_val).T + np.ones((m, 1)) * csol
        if classification:
            err_seq[i] = 1. - accuracy_score(preds > 0.5, y_val.numpy())
            print("Round " + str(i) + " Val Acc: ", 1. - err_seq[i])
        else:
            err_seq[i] = np.sqrt(np.mean(np.square(preds - y_val.numpy())))
            print("Round " + str(i) + " Val RMSE: ", err_seq[i])

    # objective function value
    obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
    obj_seq[i] = obj[0, 0] * reg / n / 2
    print("Round " + str(i) + " Obj: ", obj_seq[i])


    for i in range(iters):
        # shuffle and calculate gradient
        indices = torch.randperm(X_train.shape[0])
        G = get_grads_IMQ(reg, X_train[indices], y_train[indices], L, torch.from_numpy(M), batch_size=batch_size, unfold=unfold).astype('float32')

        # PGD with Armijo rule
        lr = lr0
        M0 = M - lr * G
        D, V = eigh(M0)
        if sum(D < 0) > 0:
            D[D < 0] = 0
            M0 = V@np.diag(D)@inv(V)
        K_train_upd = kernellab.IMQ_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
        sol_upd = solve(proj@K_train_upd + reg * np.eye(n), proj@y_train).T
        obj_upd = sol_upd @ (K_train_upd + reg * np.eye(n)) @ sol_upd.T
        f_upd = obj_upd[0,0]*reg/n/2
        Gp = U@U.T@G@U@U.T
        while f_upd > obj_seq[i] - alpha * lr * np.trace(Gp.T@Gp):
           lr *= beta
           M0 = M - lr * G
           D, V = eigh(M0)
           if sum(D < 0) > 0:
               D[D < 0] = 0
               M0 = V @ np.diag(D) @ inv(V)
           K_train_upd = kernellab.IMQ_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
           sol_upd = solve(proj@K_train_upd + reg * np.eye(n), proj@y_train).T
           obj_upd = sol_upd @ (K_train_upd + reg * np.eye(n)) @ sol_upd.T
           f_upd = obj_upd[0,0]*reg/n/2
           if lr < 1e-15:
               lr = 0.
               break
           print(lr)

        print(norm(M-M0, 'fro')/lr)
        if lr == 0 or norm(M-M0, 'fro')/lr < 1e-3:
            obj_seq[i+1] = obj_seq[i]
            terr_seq[i+1] = terr_seq[i]
            err_seq[i+1] = err_seq[i]
            rank_seq[i+1] = rank_seq[i]
            break
        else:
            M = M0
        U = V@np.diag(np.sqrt(D))
        U = U[:,D>0]
        print(D)

        # rank
        rank_seq[i+1] = sum(D > 0)

        # training err
        K_train = kernellab.IMQ_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
        preds0 = (sol @ K_train).T
        csol = np.mean(y_train.numpy() - preds0)
        preds = preds0 + np.ones((n, 1)) * csol
        terr_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
        print("Round " + str(i+1) + " Train RMSE: ", terr_seq[i+1])

        # validation err
        if X_val is not None and y_val is not None:
            K_val = kernellab.IMQ_M(X_train, X_val, L, torch.from_numpy(M)).numpy()
            preds = (sol @ K_val).T + np.ones((m, 1)) * csol
            if classification:
                err_seq[i+1] = 1. - accuracy_score(preds>0.5, y_val.numpy())
                print("Round " + str(i+1) + " Val Acc: ", 1. - err_seq[i+1])
            else:
                err_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_val.numpy())))
                print("Round " + str(i+1) + " Val RMSE: ", err_seq[i+1])

        # objective function value
        obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
        obj_seq[i+1] = obj[0, 0] * reg / n / 2
        print("Round " + str(i+1) + " Obj: ", obj_seq[i+1])

    # test err
    K_test = kernellab.IMQ_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T + np.ones((mm, 1)) * csol
    if classification:
        test_err = 1. - accuracy_score(preds > 0.5, y_test.numpy())
        print("Test Acc: ", 1. - test_err)
    else:
        test_err = np.sqrt(np.mean(np.square(preds - y_test.numpy())))
        print("Test RMSE: ", test_err)

    return test_err, M, obj_seq[0:i+2], terr_seq[0:i+2], err_seq[0:i+2], rank_seq[0:i+2]

# Matern kernel
def get_grads_MT(reg, X, Y, L, P, batch_size, unfold=0):
    d, _ = P.shape
    n, c = Y.shape
    batches_X = torch.split(X, batch_size)
    batches_Y = torch.split(Y, batch_size)
    G = torch.zeros(d, d)
    for i in tqdm(range(len(batches_X))):
        x = batches_X[i]
        y = batches_Y[i]
        m, _ = x.shape
        K = kernellab.Matern_M(x, x, L, P)

        # calculate alpha and diff
        proj = torch.eye(m)-torch.ones((m,m))/m
        a = solve(proj@K + reg * np.eye(m), proj@y)
        a = torch.from_numpy(a).float()
        diff = x.unsqueeze(1) - x.unsqueeze(0)

        # for MT
        K = kernellab.laplacian_M(x, x, L, P)

        # separate unfold cases
        if unfold == 0:
            res = torch.einsum('ijk,ijl->ijkl', diff, diff)
            C = torch.mul(res, K.view(m,m,1,1))
            del res
            G0 = torch.tensordot(a, C, dims=([0], [0]))
            del C
            G += torch.sum(torch.tensordot(torch.sum(G0, dim=0), a, dims=([0], [0])), dim=2)
            del G0

        elif unfold == 1:
            for j in range(d):
                resj = torch.einsum('ijk,ijl->ijkl', diff, diff[:, :, j:(j + 1)]).squeeze()
                Cj = torch.mul(resj, K.view(m,m,1))
                del resj
                Gj = torch.tensordot(a, Cj, dims=([0], [0]))
                del Cj
                G[j] += torch.sum(torch.tensordot(torch.sum(Gj, dim=0), a, dims=([0], [0])), dim=1)
                del Gj

        else:
            for j in range(d):
                for k in range(d):
                    resjk = torch.einsum('ijk,ijl->ijkl', diff[:, :, j:(j + 1)], diff[:, :, k:(k + 1)]).squeeze()
                    Cjk = torch.mul(resjk, K)
                    del resjk
                    Gjk = torch.tensordot(a, Cjk, dims=([0], [0]))
                    G[j,k] += torch.sum(torch.tensordot(torch.sum(Gjk, dim=0), a, dims=([0], [0])), dim=0)
                    del Gjk

    G *= reg/n/2/L
    G = G.numpy()
    return G

def trainSig_MT(X_train, y_train, X_test, y_test, M, d0,
        iters, batch_size, reg, lr0,
        alpha, beta, X_val=None, y_val=None, classification=0, unfold=0):
    L = 1
    n, d = X_train.shape
    mm, d = X_test.shape
    if X_val is not None and y_val is not None:
        m, d = X_val.shape

    U = np.eye(d, dtype='float32')
    obj_seq = np.zeros(iters+1)
    err_seq = np.zeros(iters+1); terr_seq = np.zeros(iters+1)
    rank_seq = np.zeros(iters+1)

    proj = torch.eye(n) - torch.ones((n, n)) / n


    i = 0
    # rank
    rank_seq[i] = d0

    # training err
    K_train = kernellab.Matern_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
    preds0 = (sol @ K_train).T
    csol = np.mean(y_train.numpy() - preds0)
    preds = preds0 + np.ones((n, 1)) * csol
    terr_seq[i] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
    print("Round " + str(i) + " Train RMSE: ", terr_seq[i])

    # validation err
    if X_val is not None and y_val is not None:
        K_val = kernellab.Matern_M(X_train, X_val, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_val).T + np.ones((m, 1)) * csol
        if classification:
            err_seq[i] = 1. - accuracy_score(preds > 0.5, y_val.numpy())
            print("Round " + str(i) + " Val Acc: ", 1. - err_seq[i])
        else:
            err_seq[i] = np.sqrt(np.mean(np.square(preds - y_val.numpy())))
            print("Round " + str(i) + " Val RMSE: ", err_seq[i])

    # objective function value
    obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
    obj_seq[i] = obj[0, 0] * reg / n / 2
    print("Round " + str(i) + " Obj: ", obj_seq[i])


    for i in range(iters):
        # shuffle and calculate gradient
        indices = torch.randperm(X_train.shape[0])
        G = get_grads_MT(reg, X_train[indices], y_train[indices], L, torch.from_numpy(M), batch_size=batch_size, unfold=unfold).astype('float32')

        # PGD with Armijo rule
        lr = lr0
        M0 = M - lr * G
        D, V = eigh(M0)
        if sum(D < 0) > 0:
            D[D < 0] = 0
            M0 = V@np.diag(D)@inv(V)
        K_train_upd = kernellab.Matern_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
        sol_upd = solve(proj@K_train_upd + reg * np.eye(n), proj@y_train).T
        obj_upd = sol_upd @ (K_train_upd + reg * np.eye(n)) @ sol_upd.T
        f_upd = obj_upd[0,0]*reg/n/2
        Gp = U@U.T@G@U@U.T
        while f_upd > obj_seq[i] - alpha * lr * np.trace(Gp.T@Gp):
           lr *= beta
           M0 = M - lr * G
           D, V = eigh(M0)
           if sum(D < 0) > 0:
               D[D < 0] = 0
               M0 = V @ np.diag(D) @ inv(V)
           K_train_upd = kernellab.Matern_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
           sol_upd = solve(proj@K_train_upd + reg * np.eye(n), proj@y_train).T
           obj_upd = sol_upd @ (K_train_upd + reg * np.eye(n)) @ sol_upd.T
           f_upd = obj_upd[0,0]*reg/n/2
           if lr < 1e-15:
               lr = 0.
               break
           print(lr)

        print(norm(M-M0, 'fro')/lr)
        if lr == 0 or norm(M-M0, 'fro')/lr < 1e-3:
            obj_seq[i+1] = obj_seq[i]
            terr_seq[i+1] = terr_seq[i]
            err_seq[i+1] = err_seq[i]
            rank_seq[i+1] = rank_seq[i]
            break
        else:
            M = M0
        U = V@np.diag(np.sqrt(D))
        U = U[:,D>0]
        print(D)

        # rank
        rank_seq[i+1] = sum(D > 0)

        # training err
        K_train = kernellab.Matern_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
        preds0 = (sol @ K_train).T
        csol = np.mean(y_train.numpy() - preds0)
        preds = preds0 + np.ones((n, 1)) * csol
        terr_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
        print("Round " + str(i+1) + " Train RMSE: ", terr_seq[i+1])

        # validation err
        if X_val is not None and y_val is not None:
            K_val = kernellab.Matern_M(X_train, X_val, L, torch.from_numpy(M)).numpy()
            preds = (sol @ K_val).T + np.ones((m, 1)) * csol
            if classification:
                err_seq[i+1] = 1. - accuracy_score(preds>0.5, y_val.numpy())
                print("Round " + str(i+1) + " Val Acc: ", 1. - err_seq[i+1])
            else:
                err_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_val.numpy())))
                print("Round " + str(i+1) + " Val RMSE: ", err_seq[i+1])

        # objective function value
        obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
        obj_seq[i+1] = obj[0, 0] * reg / n / 2
        print("Round " + str(i+1) + " Obj: ", obj_seq[i+1])

    # test err
    K_test = kernellab.Matern_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T + np.ones((mm, 1)) * csol
    if classification:
        test_err = 1. - accuracy_score(preds > 0.5, y_test.numpy())
        print("Test Acc: ", 1. - test_err)
    else:
        test_err = np.sqrt(np.mean(np.square(preds - y_test.numpy())))
        print("Test RMSE: ", test_err)

    return test_err, M, obj_seq[0:i+2], terr_seq[0:i+2], err_seq[0:i+2], rank_seq[0:i+2]
