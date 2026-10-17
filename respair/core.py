"""简化版：低秩混杂 + 稀疏成对联系的空间转录组–代谢组联合检验。

流程
1. 归一化：基因用解析 Pearson 残差（近似负二项），代谢物取 log 后标准化。
2. 扣除已知结构 Z 和 K 个隐藏模式 F（对拼接后的两块数据做加权 PCA）。
3. 残差投影到空间频率基上，T_jk = sum_l g_jl m_kl。
4. 零假设方差 Var0 = sum_l s_g,jl s_m,kl（按频带平滑的功率谱），z = T / sqrt(Var0)。
5. K 用负对照配对选：取负对照 z 的膨胀系数 lambda_GC 首次低于阈值的最小 K。
6. 经验校准：按负对照配对 |z| 的 95% 分位数把所有 z 缩放到名义水平。
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import laplacian
from scipy.spatial import cKDTree
from scipy.stats import norm


# ---------- 归一化 ----------

def pearson_residuals(counts, theta=100.0):
    """解析 Pearson 残差（Lause et al. 2021）。counts: n × p。"""
    counts = np.asarray(counts, dtype=float)
    n = counts.shape[0]
    total = counts.sum()
    mu = np.outer(counts.sum(1), counts.sum(0)) / total
    r = (counts - mu) / np.sqrt(mu + mu ** 2 / theta)
    clip = np.sqrt(n)
    return np.clip(r, -clip, clip)


def log_standardize(x, eps=None):
    x = np.asarray(x, dtype=float)
    if eps is None:
        pos = x[x > 0]
        eps = np.percentile(pos, 1) if pos.size else 1.0
    return np.log(x + eps)


def standardize(X):
    X = X - X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return X / sd


# ---------- 空间频率基 ----------

def spatial_basis(coords, k=8):
    """kNN 图拉普拉斯的特征向量，按频率（特征值）从低到高排列。去掉常数分量。"""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    A = sparse.coo_matrix((np.ones_like(rows, dtype=float), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    L = laplacian(A, normed=True).toarray()
    evals, U = np.linalg.eigh(L)
    return evals[1:], U[:, 1:]


# ---------- 扣除混杂 ----------

def _residualize(X, C):
    if C is None or C.shape[1] == 0:
        return X - X.mean(0)
    C1 = np.column_stack([np.ones(len(C)), C])
    beta, *_ = np.linalg.lstsq(C1, X, rcond=None)
    return X - C1 @ beta


def hidden_factors(Rg, Rm, K):
    """对拼接矩阵做加权 PCA，两块数据各自总方差相同。返回 n × K 的因子得分。"""
    if K == 0:
        return np.zeros((Rg.shape[0], 0))
    Xg = standardize(Rg) / np.sqrt(Rg.shape[1])
    Xm = standardize(Rm) / np.sqrt(Rm.shape[1])
    X = np.hstack([Xg, Xm])
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :K] * s[:K]


# ---------- 检验 ----------

def band_power(Y_hat, band_size):
    """Y_hat: L × p（频率分量 × 特征）。返回每个分量的局部平均功率，形状同 Y_hat。"""
    L = Y_hat.shape[0]
    P = np.empty_like(Y_hat)
    for start in range(0, L, band_size):
        sl = slice(start, min(start + band_size, L))
        P[sl] = (Y_hat[sl] ** 2).mean(0, keepdims=True)
    return P


def spectral_z(Eg, Em, U, band_size=10, scales=None):
    """返回 z（p × q）。scales: 频率分量的分段边界列表，若给出则同时返回各尺度的 z。"""
    G_hat = U.T @ standardize(Eg)
    M_hat = U.T @ standardize(Em)
    T = G_hat.T @ M_hat
    Pg = band_power(G_hat, band_size)
    Pm = band_power(M_hat, band_size)
    V = Pg.T @ Pm
    z = T / np.sqrt(V)
    out = {"z": z}
    if scales is not None:
        out["z_scale"] = []
        for a, b in scales:
            Ts = G_hat[a:b].T @ M_hat[a:b]
            Vs = Pg[a:b].T @ Pm[a:b]
            out["z_scale"].append(Ts / np.sqrt(Vs))
    return out


def naive_z(Eg, Em):
    """普通 Pearson 相关的 z（Fisher 变换），不做空间校正。"""
    n = Eg.shape[0]
    r = standardize(Eg).T @ standardize(Em) / n
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r) * np.sqrt(n - 3)


def pvals(z):
    return 2 * norm.sf(np.abs(z))


def calibration_factor(z_neg, q=0.95):
    """负对照 |z| 的 q 分位数与标准正态对应分位数之比。"""
    return np.quantile(np.abs(z_neg), q) / norm.isf((1 - q) / 2)


def lambda_gc(z):
    """膨胀系数：z² 的中位数 / χ²₁ 的中位数。"""
    return np.median(z ** 2) / 0.4549364


def bh(p, alpha=0.1):
    """Benjamini–Hochberg，返回布尔数组。"""
    flat = p.ravel()
    m = flat.size
    order = np.argsort(flat)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = flat[order] <= thresh
    k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej.reshape(p.shape)


# ---------- 主流程 ----------

def fit(G_counts, M_int, coords, Z=None, K=None, K_grid=range(0, 31, 2),
        neg_genes=None, gc_target=1.1, band_size=10, knn=8, gene_resid=None, met_resid=None,
        calibrate=True):
    """
    G_counts: n × p 基因计数；M_int: n × q 代谢物强度；coords: n × 2；Z: n × c 已知结构（可选）。
    K: 隐藏模式个数；为 None 时用负对照基因 neg_genes（索引）按 lambda_GC 自动选。
    calibrate: 有负对照时按负对照经验分布缩放 z（见 calibration_factor）。
    返回 dict：z, p, K, lambda_gc 曲线, 校准系数 c, 基, 因子。
    """
    Yg = pearson_residuals(G_counts) if gene_resid is None else gene_resid
    Ym = standardize(log_standardize(M_int)) if met_resid is None else met_resid
    Rg = _residualize(Yg, Z)
    Rm = _residualize(Ym, Z)
    evals, U = spatial_basis(coords, k=knn)

    # 预先算出最大 K 的因子，较小的 K 取前几列
    Kmax = K if K is not None else max(K_grid)
    F_all = hidden_factors(Rg, Rm, Kmax)

    curve = {}
    if K is None:
        if neg_genes is None:
            raise ValueError("自动选 K 需要负对照基因 neg_genes")
        for k in K_grid:
            F = F_all[:, :k]
            Eg = _residualize(Rg[:, neg_genes], F)
            Em = _residualize(Rm, F)
            z = spectral_z(Eg, Em, U, band_size)["z"]
            curve[k] = lambda_gc(z)
            if curve[k] < gc_target:
                K = k
                break
        if K is None:
            K = min(curve, key=curve.get)

    F = F_all[:, :K]
    Eg = _residualize(Rg, F)
    Em = _residualize(Rm, F)
    nL = U.shape[1]
    scales = [(0, nL // 20), (nL // 20, nL // 4), (nL // 4, nL)]
    res = spectral_z(Eg, Em, U, band_size, scales=scales)
    z = res["z"]
    c = calibration_factor(z[neg_genes]) if (calibrate and neg_genes is not None) else 1.0
    z = z / c
    return {"z": z, "p": pvals(z), "z_scale": [zs / c for zs in res["z_scale"]],
            "scales": scales, "K": K, "gc_curve": curve, "c": float(c), "F": F, "evals": evals,
            "Eg": Eg, "Em": Em, "z_uncalibrated": res["z"]}
