import numpy as np
from math import sqrt, log1p
from scipy.stats import norm, distributions


def gauss_ci_test_generator(df, alpha):
    # df: a DataFrame
    # alpha: significant
    cov = df.cov()
    n = df.shape[0]

    def gauss_ci_test(x, y, s):
        nonlocal cov
        nonlocal n
        if len(s) == 0:
            r = cov[x, y]
        elif len(s) == 1:
            r = (cov[x, y] - cov[x, s] * cov[s, y]) / (
                (sqrt(1 - cov[x, s] ** 2) * sqrt(1 - cov[s, y] ** 2)))
        else:
            li = [x] + [y] + s
            m = cov[li, li]
            pm = np.linalg.pinv(m)
            r = -1.0 * pm[0, 1] / sqrt(abs(pm[0, 0] * pm[1, 1]))
        cut_at = 0.9999999
        r = min(cut_at, max(-cut_at, r.item()))
        res = sqrt(n - len(s) - 3) * .5 * log1p((2 * r) / (1 - r))
        return 2 * (1 - norm.cdf(abs(res))) >= alpha

    return gauss_ci_test


def chi_square(obs):
    # print(rows)
    # print(np.transpose(cols))
    # alls = np.sum(rows)
    # print(np.dot(rows, cols))
    rows = np.asmatrix(np.sum(obs, axis=0))
    cols = np.asmatrix(np.sum(obs, axis=1))
    alls = np.sum(rows)
    exp = np.multiply(rows, np.transpose(cols)) / alls
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    terms = np.power(obs - exp, 2) / exp
    stat = np.sum(terms)
    p = distributions.chi2.sf(stat, dof)
    print(p)
    return stat, p

def gtest(obs):
    rows = np.asmatrix(np.sum(obs, axis=0))
    cols = np.asmatrix(np.sum(obs, axis=1))
    alls = np.sum(rows)
    exp = np.multiply(rows, np.transpose(cols)) / alls
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    terms = 2 * np.multiply(obs, np.log(obs/exp))
    g = np.sum(terms)
    p = distributions.chi2.sf(g, dof)
    return g, p