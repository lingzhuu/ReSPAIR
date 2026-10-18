"""Benchmark: false-positive rate, power and FDR of every method on simulated data.

Usage: python benchmarks/scripts/run_sim.py [n_seeds]
Writes results/sim_{n_seeds}seeds.json.
"""
import sys, os, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
import respair
from respair import core as sj
import baselines as bm
from simulate import simulate

# Main comparison (existing methods vs ReSPAIR) and ablation (each step of ReSPAIR)
MAIN = ["Pearson", "MANTIS-SCI", "MANTIS-SPC-CT", "ReSPAIR"]
ABLATION = ["Z-only + Pearson", "Z-only + spectral test", "Hidden factors + Pearson", "ReSPAIR (uncalibrated)"]


def evaluate(z=None, truth=None, conf_pair=None, p=None, alpha=0.05, fdr=0.1):
    if p is None:
        p = sj.pvals(z)
    null = ~truth
    conf_null = conf_pair & null
    other_null = null & ~conf_pair
    rej = sj.bh(p, fdr)
    n_rej = rej.sum()
    return {
        "FPR confounded pairs": float((p[conf_null] < alpha).mean()),
        "FPR other null pairs": float((p[other_null] < alpha).mean()),
        "power (p<0.05)": float((p[truth] < alpha).mean()),
        "BH discoveries": int(n_rej),
        "BH empirical FDR": float((rej & null).sum() / max(n_rej, 1)),
        "BH power": float((rej & truth).sum() / truth.sum()),
    }


def stepwise_fit(G_counts, M_int, coords, Z=None, neg_genes=None, gene_resid=None, met_resid=None):
    """Step-wise pipeline used only for the ablations: weighted-PCA hidden factors, residual spectral test, calibration."""
    Yg = sj.pearson_residuals(G_counts) if gene_resid is None else gene_resid
    Ym = sj.standardize(sj.log_standardize(M_int)) if met_resid is None else met_resid
    Rg, Rm = sj._residualize(Yg, Z), sj._residualize(Ym, Z)
    _, U = sj.spatial_basis(coords)
    F = sj.hidden_factors(Rg, Rm, sj.select_K(Rg, Rm))
    Eg, Em = sj._residualize(Rg, F), sj._residualize(Rm, F)
    z0 = sj.spectral_z(Eg, Em, U)["z"]
    c = sj.calibration_factor(z0[neg_genes]) if neg_genes is not None else 1.0
    return {"z": z0 / c, "p": sj.pvals(z0 / c), "K": F.shape[1], "c": float(c), "Eg": Eg, "Em": Em,
            "z_uncalibrated": z0}


def all_pvals(d, with_mantis=True):
    """Return {method: p-value matrix}, the step-wise fit (residuals for the ablations and figure 1) and the
    normalised data."""
    Yg = sj.pearson_residuals(d["G"])
    Ym = sj.standardize(sj.log_standardize(d["M"]))
    Z = d["Z"]
    _, U = sj.spatial_basis(d["coords"])
    Rg, Rm = sj._residualize(Yg, Z), sj._residualize(Ym, Z)
    fit = stepwise_fit(d["G"], d["M"], d["coords"], Z=Z, neg_genes=d["neg_genes"], gene_resid=Yg, met_resid=Ym)
    full = respair.fit(d["G"], d["M"], d["coords"], Z=Z, neg_genes=d["neg_genes"])
    P = {
        "Pearson": sj.pvals(sj.naive_z(Yg, Ym)),
        "Z-only + Pearson": sj.pvals(sj.naive_z(Rg, Rm)),
        "Z-only + spectral test": sj.pvals(sj.spectral_z(Rg, Rm, U)["z"]),
        "Hidden factors + Pearson": sj.pvals(sj.naive_z(fit["Eg"], fit["Em"])),
        "ReSPAIR (uncalibrated)": sj.pvals(full["z_uncalibrated"]),
        "ReSPAIR": full["p"],
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
    out = {"per seed": results, "mean": summary, "sd": sd}
    with open(os.path.join(ROOT, "results", f"sim_{n_seeds}seeds.json"), "w") as f:
        json.dump(out, f, indent=1)
    for m in methods:
        print(m, {k: round(v, 3) for k, v in summary[m].items()})
