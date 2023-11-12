'''Gradient function and PGD algorithm (other inner product kernels: linear and cubic)'''

import numpy as np
import torch
from scipy.linalg import eigh
from numpy.linalg import solve, inv, norm
import kernellab
from tqdm import tqdm

# linear kernel
def get_grads_linear(reg, X, Y, L, P, batch_size, unfold=0):
    d, _ = P.shape
    n, c = Y.shape
    batches_X = torch.split(X, batch_size)
    batches_Y = torch.split(Y, batch_size)
    G = torch.zeros(d, d)
    for i in tqdm(range(len(batches_X))):
        x = batches_X[i]
        y = batches_Y[i]
        m, _ = x.shape
        K = kernellab.linear_M(x, x, L, P)

        # calculate alpha and diff
        proj = torch.eye(m)-torch.ones((m,m))/m
        a = solve(proj@K + reg * np.eye(m), proj@y)
        a = torch.from_numpy(a).float()

        # for linear
        del K

        C = torch.einsum('ij,kl->ikjl', x, x)+torch.einsum('ij,kl->kijl', x, x)
        G0 = torch.tensordot(a, C, dims=([0], [0]))
        del C
        G += torch.sum(torch.tensordot(torch.sum(G0, dim=0), a, dims=([0], [0])), dim=2)
        del G0

    G *= -reg/n/2/L
    G = G.numpy()
    return G

def trainSig_linear(X_train, y_train, X_test, y_test, M, d0,
        iters, batch_size, reg, lr0,
        alpha, beta, unfold=0):
    L = 1
    n, d = X_train.shape
    m, d = X_test.shape

    U = np.eye(d, dtype='float32')
    obj_seq = np.zeros(iters+1)
    err_seq = np.zeros(iters+1); terr_seq = np.zeros(iters+1)
    rank_seq = np.zeros(iters+1)

    proj = torch.eye(n) - torch.ones((n, n)) / n


    i = 0
    # rank
    rank_seq[i] = d0

    # training err
    K_train = kernellab.linear_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
    preds0 = (sol @ K_train).T
    csol = np.mean(y_train.numpy() - preds0)
    preds = preds0 + np.ones((n, 1)) * csol
    terr_seq[i] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
    print("Round " + str(i) + " Train RMSE: ", terr_seq[i])

    # testing err
    K_test = kernellab.linear_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T + np.ones((m, 1)) * csol
    err_seq[i] = np.sqrt(np.mean(np.square(preds - y_test.numpy())))
    print("Round " + str(i) + " RMSE: ", err_seq[i])

    # objective function value
    obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
    obj_seq[i] = obj[0, 0] * reg / n / 2
    print("Round " + str(i) + " Obj: ", obj_seq[i])


    for i in range(iters):
        # shuffle and calculate gradient
        indices = torch.randperm(X_train.shape[0])
        G = get_grads_linear(reg, X_train[indices], y_train[indices], L, torch.from_numpy(M), batch_size=batch_size, unfold=unfold).astype('float32')

        # PGD with Armijo rule
        lr = lr0
        M0 = M - lr * G
        D, V = eigh(M0)
        if sum(D < 0) > 0:
            D[D < 0] = 0
            M0 = V@np.diag(D)@inv(V)
        K_train_upd = kernellab.linear_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
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
           K_train_upd = kernellab.linear_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
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
        K_train = kernellab.linear_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
        preds0 = (sol @ K_train).T
        csol = np.mean(y_train.numpy() - preds0)
        preds = preds0 + np.ones((n, 1)) * csol
        terr_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
        print("Round " + str(i+1) + " Train RMSE: ", terr_seq[i+1])

        # testing err
        K_test = kernellab.linear_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T + np.ones((m, 1)) * csol
        err_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_test.numpy())))
        print("Round " + str(i+1) + " RMSE: ", err_seq[i+1])

        # objective function value
        obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
        obj_seq[i+1] = obj[0, 0] * reg / n / 2
        print("Round " + str(i+1) + " Obj: ", obj_seq[i+1])

    return M, obj_seq[0:i+2], terr_seq[0:i+2], err_seq[0:i+2], rank_seq[0:i+2]

# cubic kernel
def get_grads_cubic(reg, X, Y, L, P, batch_size, unfold=0):
    d, _ = P.shape
    n, c = Y.shape
    batches_X = torch.split(X, batch_size)
    batches_Y = torch.split(Y, batch_size)
    G = torch.zeros(d, d)
    for i in tqdm(range(len(batches_X))):
        x = batches_X[i]
        y = batches_Y[i]
        m, _ = x.shape
        K = kernellab.cubic_M(x, x, L, P)

        # calculate alpha and diff
        proj = torch.eye(m)-torch.ones((m,m))/m
        a = solve(proj@K + reg * np.eye(m), proj@y)
        a = torch.from_numpy(a).float()

        del K
        K = kernellab.square_M(x, x, L, P)
        res = torch.einsum('ij,kl->ikjl', x, x)+torch.einsum('ij,kl->kijl', x, x)
        C = torch.mul(res, K.view(m, m, 1, 1))
        del res
        G0 = torch.tensordot(a, C, dims=([0], [0]))
        del C
        G += torch.sum(torch.tensordot(torch.sum(G0, dim=0), a, dims=([0], [0])), dim=2)
        del G0

    G *= -reg/n/L*3/2
    G = G.numpy()
    return G

def trainSig_cubic(X_train, y_train, X_test, y_test, M, d0,
        iters, batch_size, reg, lr0,
        alpha, beta, unfold=0):
    L = 1
    n, d = X_train.shape
    m, d = X_test.shape

    U = np.eye(d, dtype='float32')
    obj_seq = np.zeros(iters+1)
    err_seq = np.zeros(iters+1); terr_seq = np.zeros(iters+1)
    rank_seq = np.zeros(iters+1)

    proj = torch.eye(n) - torch.ones((n, n)) / n


    i = 0
    # rank
    rank_seq[i] = d0

    # training err
    K_train = kernellab.cubic_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
    preds0 = (sol @ K_train).T
    csol = np.mean(y_train.numpy() - preds0)
    preds = preds0 + np.ones((n, 1)) * csol
    terr_seq[i] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
    print("Round " + str(i) + " Train RMSE: ", terr_seq[i])

    # testing err
    K_test = kernellab.cubic_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T + np.ones((m, 1)) * csol
    err_seq[i] = np.sqrt(np.mean(np.square(preds - y_test.numpy())))
    print("Round " + str(i) + " RMSE: ", err_seq[i])

    # objective function value
    obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
    obj_seq[i] = obj[0, 0] * reg / n / 2
    print("Round " + str(i) + " Obj: ", obj_seq[i])


    for i in range(iters):
        # shuffle and calculate gradient
        indices = torch.randperm(X_train.shape[0])
        G = get_grads_cubic(reg, X_train[indices], y_train[indices], L, torch.from_numpy(M), batch_size=batch_size, unfold=unfold).astype('float32')

        # PGD with Armijo rule
        lr = lr0
        M0 = M - lr * G
        D, V = eigh(M0)
        if sum(D < 0) > 0:
            D[D < 0] = 0
            M0 = V@np.diag(D)@inv(V)
        K_train_upd = kernellab.cubic_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
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
           K_train_upd = kernellab.cubic_M(X_train, X_train, L, torch.from_numpy(M0)).numpy()
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
        K_train = kernellab.cubic_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(proj @ K_train + reg * np.eye(n), proj @ y_train).T
        preds0 = (sol @ K_train).T
        csol = np.mean(y_train.numpy() - preds0)
        preds = preds0 + np.ones((n, 1)) * csol
        terr_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_train.numpy())))
        print("Round " + str(i+1) + " Train RMSE: ", terr_seq[i+1])

        # testing err
        K_test = kernellab.cubic_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T + np.ones((m, 1)) * csol
        err_seq[i+1] = np.sqrt(np.mean(np.square(preds - y_test.numpy())))
        print("Round " + str(i+1) + " RMSE: ", err_seq[i+1])

        # objective function value
        obj = sol @ (K_train + reg * np.eye(n)) @ sol.T
        obj_seq[i+1] = obj[0, 0] * reg / n / 2
        print("Round " + str(i+1) + " Obj: ", obj_seq[i+1])

    return M, obj_seq[0:i+2], terr_seq[0:i+2], err_seq[0:i+2], rank_seq[0:i+2]
