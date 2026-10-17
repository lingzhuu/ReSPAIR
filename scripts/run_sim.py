"""在模拟数据上比较四种做法，输出误报率、检出率、FDR。

用法：python3 scripts/run_sim.py [种子数]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from respair import core as sj
from respair.simulate import simulate


def evaluate(z, truth, conf_pair, alpha=0.05, fdr=0.1):
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


def run(seed):
    d = simulate(seed=seed)
    Yg = sj.pearson_residuals(d["G"])
    Ym = sj.standardize(sj.log_standardize(d["M"]))
    Z = d["Z"]
    _, U = sj.spatial_basis(d["coords"])
    Rg, Rm = sj._residualize(Yg, Z), sj._residualize(Ym, Z)

    out = {}
    out["普通相关"] = evaluate(sj.naive_z(Yg, Ym), d["truth"], d["conf_pair"])
    out["只扣已知结构+普通相关"] = evaluate(sj.naive_z(Rg, Rm), d["truth"], d["conf_pair"])
    out["只扣已知结构+空间检验"] = evaluate(sj.spectral_z(Rg, Rm, U)["z"], d["truth"], d["conf_pair"])
    fit = sj.fit(d["G"], d["M"], d["coords"], Z=Z, neg_genes=d["neg_genes"],
                 gene_resid=Yg, met_resid=Ym)
    out["扣隐藏模式+普通相关"] = evaluate(sj.naive_z(fit["Eg"], fit["Em"]), d["truth"], d["conf_pair"])
    out["简化版(未校准)"] = evaluate(fit["z_uncalibrated"], d["truth"], d["conf_pair"])
    out["简化版(负对照校准)"] = evaluate(fit["z"], d["truth"], d["conf_pair"])
    out["_c"] = fit["c"]
    out["_K"] = fit["K"]
    out["_gc曲线"] = {int(k): round(float(v), 3) for k, v in fit["gc_curve"].items()}
    return out


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    results = [run(s) for s in range(n_seeds)]
    methods = [k for k in results[0] if not k.startswith("_")]
    summary = {m: {k: float(np.mean([r[m][k] for r in results])) for k in results[0][m]} for m in methods}
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", f"sim_{n_seeds}seeds.json")
    with open(out_path, "w") as f:
        json.dump({"每个种子": results, "平均": summary}, f, ensure_ascii=False, indent=1)
    print(json.dumps({"每个种子": results, "平均": summary}, ensure_ascii=False, indent=1))
