"""对比方法。

MANTIS 按其公开代码（github.com/yuhaotuo/MANTIS，v0.55，tools/_spc.py 与 _sampler.py）的流程复现，
做了两处为速度的改动，统计量和零分布的构造不变：
- 保持空间自相关的置换：原代码用 networkx 图上的模拟退火逐对交换；这里用稠密权重矩阵向量化实现同一个目标
  （让置换后 x' W x 等于原值），贪心接受使目标差变小的交换。
- 每个代谢物生成 1 个置换（与原代码默认一致），零分布按代谢物空间自相关分 10 箱汇总。
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import Lasso

from .core import standardize


def _zscore(A):
    return standardize(np.asarray(A, dtype=float))


def gaussian_W(coords, l2, row_norm=True, zero_diag=True):
    D2 = cdist(coords, coords, "sqeuclidean")
    W = np.exp(-D2 / (2 * l2))
    if zero_diag:
        np.fill_diagonal(W, 0.0)
    if row_norm:
        W /= W.sum(1, keepdims=True)
    return W


def sampler_W(coords, l):
    """MANTIS 采样器的权重：距离 <= l 权重为 1，之外为高斯衰减。"""
    D = cdist(coords, coords)
    W = np.where(D <= l, 1.0, np.exp(-D ** 2 / (2 * l * l)))
    np.fill_diagonal(W, 0.0)
    return W


def autocorr_preserving_shuffle(x, W, rng, max_steps=60000, rel_tol=1e-3):
    """返回 x 的一个置换，使 x' W x 接近原值（MANTIS 零分布的思路）。"""
    xc = x - x.mean()
    target = xc @ W @ xc
    y = xc[rng.permutation(len(xc))]
    S = W @ y
    cur = y @ S
    tol = rel_tol * abs(target) + 1e-12
    n = len(y)
    for _ in range(max_steps):
        if abs(cur - target) < tol:
            break
        u, v = rng.integers(n, size=2)
        if u == v:
            continue
        a, b = y[u], y[v]
        # 交换 u, v 后 y'Wy 的变化
        delta = 2 * (b - a) * (S[u] - S[v] - W[u, v] * (b - a))
        if abs(cur + delta - target) < abs(cur - target):
            y[u], y[v] = b, a
            S += (b - a) * W[:, u] + (a - b) * W[:, v]
            cur += delta
    return y + x.mean()


def _lasso_resid(X, P, alpha):
    R = np.empty_like(X)
    mdl = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000)
    for j in range(X.shape[1]):
        y = X[:, j]
        if P is None:
            R[:, j] = y
            continue
        mdl.fit(P, y)
        R[:, j] = y - P @ mdl.coef_
    return R


def _pooled_empirical_p(real, null):
    z = np.sort(null.ravel())
    N0 = z.size
    left = np.searchsorted(z, real, side="right")
    right = N0 - np.searchsorted(z, real, side="left")
    p = 2 * np.minimum((left + 1) / (N0 + 1), (right + 1) / (N0 + 1))
    return np.minimum(p, 1.0)


def mantis(Yg, Ym, coords, CT=None, seed=0, n_bins=10, sampler_l=4 * np.sqrt(2), spc_l2=10.0):
    """MANTIS 的 SCI（CT=None）或 SPC-CT（给出细胞类型比例 CT）。返回 p 值矩阵（基因 × 代谢物）。"""
    rng = np.random.default_rng(seed)
    X0 = _zscore(Yg)
    Y0 = _zscore(Ym)
    P = None
    if CT is not None:
        P = _zscore(CT)
    Ws = sampler_W(coords, sampler_l)
    W = gaussian_W(coords, spc_l2)
    n = len(coords)

    Xs = _zscore(_lasso_resid(X0, P, 0.1))
    Ys = _zscore(_lasso_resid(Y0, P, 0.1))
    real = Xs.T @ (W @ Ys) / n

    Ynull = np.column_stack([autocorr_preserving_shuffle(Ym[:, k], Ws, rng) for k in range(Ym.shape[1])])
    Yn = _zscore(_lasso_resid(_zscore(Ynull), P, 0.06))
    null = Xs.T @ (W @ Yn) / n

    # 按代谢物的空间自相关分箱，箱内汇总零分布
    Yc = Y0 - Y0.mean(0)
    M0 = np.einsum("ik,ij,jk->k", Yc, Ws, Yc)
    edges = np.unique(np.quantile(M0, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, M0, side="right") - 1, 0, len(edges) - 2)
    p = np.ones_like(real)
    for b in np.unique(bins):
        cols = bins == b
        p[:, cols] = _pooled_empirical_p(real[:, cols], null[:, cols])
    return p
