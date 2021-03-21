import numpy as np
import numpy.linalg as la
from scipy.stats import gamma


def rbf_dot2(p1, p2, deg):
    if p1.ndim == 1:
        p1 = np.expand_dims(p1, axis=1)
        p2 = np.expand_dims(p2, axis=1)

    size1 = p1.shape
    size2 = p2.shape

    G = np.expand_dims(np.sum(p1 * p1, axis=1), axis=1)
    H = np.expand_dims(np.sum(p2 * p2, axis=1), axis=1)
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    H = Q + R - 2.0 * np.dot(p1, p2.T)
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


def rbf_dot(X, deg):
    # Set kernel size to median distance between points, if no kernel specified
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
    m = X.shape[0]
    G = np.expand_dims(np.sum(X * X, axis=1), axis=1)
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    H = np.exp(-H / 2.0 / (deg ** 2))

    return H


# https://github.com/lawrennd/mlopy/blob/master/netlab/netlab.py


def eigdec(x, N):
    """EIGDEC	Sorted eigendecomposition
        Description
        EVALS, EVEC = EIGDEC(X, N) computes the largest N eigenvalues of the
        matrix X in descending order.
        See also
        PCA, PPCA
        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

    # x = np.asmatrix(x)
    # Would be true if you are returning only eigenvectors, can't make
    # that decision in python
    evals_only = False

    if not N == round(N) or N < 1 or N > x.shape[1]:
        raise Exception('Number of eigenvalues must be integer, >0, < dim')

    # Find the eigenvalues of the data covariance matrix
    if evals_only:
        # This isn't called in python version.
        # Use eig function as always more efficient than eigs here
        temp_evals = np.eig(x)
    else:
        # Use eig function unless fraction of eigenvalues required is tiny
        if (N / x.shape[1]) > 0.04:
            temp_evals, temp_evec = la.eig(x)
        else:
            # Want to use eigs here, but it doesn't exist for python yet.
            # options.disp = 0
            # temp_evec, temp_evals = eigs(x, N, 'LM', options)
            temp_evals, temp_evec = la.eig(x)

    # Sort eigenvalues into descending order
    perm = np.argsort(-temp_evals)
    evals = temp_evals[perm[0:N]]

    if not evals_only:
        # should always come through here.
        evec = temp_evec[:, perm[0:N]]
    return evals, evec


def kernel(X, Y, theta):
    if X.ndim == 1:
        X = np.expand_dims(X, axis=1)
        Y = np.expand_dims(Y, axis=1)
    size1 = X.shape
    size2 = Y.shape

    G = np.expand_dims(np.sum(X * X, axis=1), axis=1)
    H = np.expand_dims(np.sum(Y * Y, axis=1), axis=1)
    Q = np.tile(G, (1, size2[0]))
    R = np.tile(H.T, (size1[0], 1))
    H = Q + R - 2.0 * np.dot(X, Y.T)
    wi2 = theta / 2
    H = np.exp(-H * wi2)
    return H


def stack(M):
    # n, t = M.shape
    # v = np.zeros(n * t, dtype=float)
    # for i in range(t):
    #     v[i * n:(i + 1) * n] = M[:, i]
    # return v
    return M.transpose().reshape(M.shape[0] * M.shape[1])


def uind_kci_test(x, y, width=0):
    T = x.shape[0]

    if T > 1000:
        approximate = True
        bootstrap = False
    else:
        bootstrap = True
        approximate = False

    Method_kernel_width = 1

    if T > 1000:
        eig_num = T // 2
    else:
        eig_num = T

    T_BS = 1000
    Thresh = 1e-6

    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    if width == 0:
        # print("width == 0")
        if T < 200:
            width = .8
        elif T < 1200:
            width = .5
        else:
            width = .3
    if Method_kernel_width == 1:
        theta = 1 / (width ** 2)
    else:
        theta = 0

    H = np.eye(T) - np.ones((T, T)) / T
    # Kx = rbf_dot(x, -1)
    # Ky = rbf_dot(y, -1)
    # Kx = rbf_dot2(x,x,-1)
    # Ky = rbf_dot2(y,y,-1)
    Kx = kernel(x, x, theta)
    Ky = kernel(y, y, theta)
    Kx = np.dot(H, np.dot(Kx, H))
    Ky = np.dot(H, np.dot(Ky, H))
    # print(Ky)
    stat = np.trace(np.dot(Kx, Ky))
    pval = -1
    if bootstrap:
        eig_Kx, eivx = eigdec(.5 * (Kx + Kx.T), eig_num)
        eig_Ky, eivy = eigdec(.5 * (Ky + Ky.T), eig_num)

        if eig_Kx.ndim == 1:
            eig_Kx = np.expand_dims(eig_Kx, axis=1)
            eig_Ky = np.expand_dims(eig_Ky, axis=1)
        eig_prod = stack(np.tile(eig_Kx, (1, eig_num)) *
                         np.tile(eig_Ky.T, (eig_num, 1)))
        II = np.where(eig_prod > np.max(eig_prod) * Thresh)
        eig_prod = eig_prod[II]
        eig_prod = np.expand_dims(eig_prod, axis=1)

        if len(eig_prod) * T < 1e6:
            f_rand1 = np.random.chisquare(1, (len(eig_prod), T_BS))
            Null_dstr = np.dot(eig_prod.T / T, f_rand1)
        else:
            Null_dstr = np.asmatrix(np.zeros(T_BS))
            Length = max(100, np.floor(1e6 / T))
            Itmax = np.floor(len(eig_prod) / Length)
            for i in range(Itmax):
                f_rand1 = np.random.chisquare(1, (Length, T_BS))
                temp_eig_prod = eig_prod[i * Length:(i + 1) * Length]
                Null_dstr = Null_dstr + np.dot(temp_eig_prod.T / T, f_rand1)
            temp_eig_prod = eig_prod[Itmax * Length:]
            Null_dstr = Null_dstr + \
                np.dot(temp_eig_prod.T / T, np.random.chisquare(1,
                                                                (len(eig_prod) - Itmax * Length, T_BS)))
        # sort_Null_dstr = np.sort(Null_dstr)
        pval = np.sum(Null_dstr > stat) / T_BS

    if approximate:
        mean_appr = np.trace(Kx) * np.trace(Ky) / T
        var_appr = 2 * np.trace(np.dot(Kx, Kx)) * \
            np.trace(np.dot(Ky, Ky)) / T ** 2
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr
        p_appr = 1 - gamma.cdf(stat, k_appr, scale=theta_appr)
        pval = p_appr

    return stat, pval
