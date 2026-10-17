"""Benchmark：在模拟数据上比较各方法的误报率、检出率和 FDR。

用法：python3 scripts/run_sim.py [种子数]
结果写到 results/sim_{种子数}seeds.json。
"""
import sys, os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from respair import core as sj
from respair import benchmarks as bm
from respair.simulate import simulate

# 主比较（已有方法 vs 我们的方法）和消融（我们方法拆开的每一步）
MAIN = ["普通相关", "MANTIS-SCI", "MANTIS-SPC-CT", "简化版"]
ABLATION = ["只扣已知结构+普通相关", "只扣已知结构+空间检验", "扣隐藏模式+普通相关", "简化版(未校准)"]


def evaluate(z=None, truth=None, conf_pair=None, p=None, alpha=0.05, fdr=0.1):
    if p is None:
        p = sj.pvals(z)
    null = ~truth
    conf_null = conf_pair & null
    other_null = null & ~conf_pair
    rej = sj.bh(p, fdr)
    n_rej = rej.sum()
    return {
        "混杂配对误报率": float((p[conf_null] < alpha).mean()),
        "其他零配对误报率": float((p[other_null] < alpha).mean()),
        "检出率(p<0.05)": float((p[truth] < alpha).mean()),
        "BH发现数": int(n_rej),
        "BH实际FDR": float((rej & null).sum() / max(n_rej, 1)),
        "BH检出率": float((rej & truth).sum() / truth.sum()),
    }


def all_pvals(d, with_mantis=True):
    """返回 {方法名: p 值矩阵}，以及简化版的拟合结果。"""
    Yg = sj.pearson_residuals(d["G"])
    Ym = sj.standardize(sj.log_standardize(d["M"]))
    Z = d["Z"]
    _, U = sj.spatial_basis(d["coords"])
    Rg, Rm = sj._residualize(Yg, Z), sj._residualize(Ym, Z)
    fit = sj.fit(d["G"], d["M"], d["coords"], Z=Z, neg_genes=d["neg_genes"], gene_resid=Yg, met_resid=Ym)
    P = {
        "普通相关": sj.pvals(sj.naive_z(Yg, Ym)),
        "只扣已知结构+普通相关": sj.pvals(sj.naive_z(Rg, Rm)),
        "只扣已知结构+空间检验": sj.pvals(sj.spectral_z(Rg, Rm, U)["z"]),
        "扣隐藏模式+普通相关": sj.pvals(sj.naive_z(fit["Eg"], fit["Em"])),
        "简化版(未校准)": sj.pvals(fit["z_uncalibrated"]),
        "简化版": fit["p"],
    }
    if with_mantis:
        P["MANTIS-SCI"] = bm.mantis(Yg, Ym, d["coords"], CT=None, seed=1)
        P["MANTIS-SPC-CT"] = bm.mantis(Yg, Ym, d["coords"], CT=Z, seed=1)
    return P, fit, (Yg, Ym)


def run(seed, **sim_kw):
    d = simulate(seed=seed, **sim_kw)
    P, fit, _ = all_pvals(d)
    out = {m: evaluate(truth=d["truth"], conf_pair=d["conf_pair"], p=P[m]) for m in MAIN + ABLATION}
    out["_K"] = fit["K"]
    out["_c"] = fit["c"]
    return out


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    results = []
    for s in range(n_seeds):
        results.append(run(s))
        print("seed", s, "done", flush=True)
    methods = [k for k in results[0] if not k.startswith("_")]
    summary = {m: {k: float(np.mean([r[m][k] for r in results])) for k in results[0][m]} for m in methods}
    sd = {m: {k: float(np.std([r[m][k] for r in results])) for k in results[0][m]} for m in methods}
    out = {"每个种子": results, "平均": summary, "标准差": sd}
    with open(os.path.join(ROOT, "results", f"sim_{n_seeds}seeds.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for m in methods:
        print(m, {k: round(v, 3) for k, v in summary[m].items()})
