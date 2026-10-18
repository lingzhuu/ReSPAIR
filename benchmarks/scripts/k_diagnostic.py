"""Which hidden factor carries the dopamine–gene associations? (Vicari et al. 2024, mouse striatum, FMP-10)

For dopamine (m/z 674.2805) against Drd1, Drd2, Ppp1r1b, Pcp4 and Tac1, ReSPAIR is refitted with
K = 0, 1, ..., K_rule hidden factors and the pair z-score and rank (among all genes, for the dopamine column) are
recorded. The initial factors are the leading components of a weighted PCA of both modalities and are nested in K,
so the association can be decomposed over them: corr(gene, dopamine) = sum_k r(gene, F_k) r(dopamine, F_k) + rest,
where r are correlations with the k-th component; each term is the part of the association carried by factor k.

Usage: python benchmarks/scripts/k_diagnostic.py <aligned_authors dir> <section> [<section> ...]
Writes results/k_diagnostic_<section>.json; plot with --plot after all sections are done.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from respair import core
from respair.estimator import _apply, fit, kernel_map
import real_data as rd

GROUP = "vicari_mouse_striatum_fmp10"
GENES = ["Drd1", "Drd2", "Ppp1r1b", "Pcp4", "Tac1"]
K_SHOWN = [0, 1, 2, 3, 5, 10]


def features(data_dir):
    s = rd.load_vicari(data_dir, GROUP)
    L = list(s.values())
    genes = rd.select_shared(L, "G", "genes", rd.N_GENES, rd.log_cpm, force=rd.DOPAMINE_GENES[GROUP])
    mz = rd.select_shared(L, "M", "mz", rd.N_METS, lambda M: np.log1p(rd.tic_normalise(M)), force=[rd.DOPAMINE_MZ])
    return s, list(genes), list(mz)


def run(data_dir, sec):
    s, genes, mz = features(data_dir)
    d = rd.subset(s[sec], genes, mz)
    k_da, gi = mz.index(rd.DOPAMINE_MZ), [genes.index(g) for g in GENES]

    # factor decomposition on the initial (nested) factors, as inside respair.fit
    Ym = core.standardize(core.log_standardize(d["M"]))
    Rg0 = core.pearson_residuals(d["G"])
    Rm0 = _apply(kernel_map(d["coords"], d["coords"]), Ym)
    K_rule = core.select_K(Rg0, Rm0)
    F = core.hidden_factors(Rg0, Rm0, K_rule)
    zs = lambda X: (X - X.mean(0)) / X.std(0)
    Fz, Gz, mz_ = zs(F), zs(Rg0[:, gi]), zs(Rm0[:, [k_da]])[:, 0]
    n = len(mz_)
    r_g, r_m = Gz.T @ Fz / n, Fz.T @ mz_ / n
    contrib = r_g * r_m[None, :]                      # genes × factors
    total = Gz.T @ mz_ / n

    out = {"section": sec, "spots": n, "K_rule": int(K_rule), "genes": GENES,
           "corr(gene, dopamine) before removing factors": dict(zip(GENES, total.round(4).tolist())),
           "factor contributions r(gene,F_k) r(dopamine,F_k)": {g: contrib[i].round(4).tolist()
                                                                for i, g in enumerate(GENES)},
           "r(dopamine, F_k)": r_m.round(4).tolist(), "by K": {}}
    for K in range(K_rule + 1):
        f = fit(d["G"], d["M"], d["coords"], K=K, seed=0)
        col = f["z"][:, k_da]
        order = np.argsort(-np.abs(col), kind="stable")
        rank = np.empty(len(col), int); rank[order] = np.arange(1, len(col) + 1)
        out["by K"][K] = {g: {"z": float(col[i]), "rank": int(rank[i])} for g, i in zip(GENES, gi)}
        print(sec, "K", K, {g: (round(v["z"], 1), v["rank"]) for g, v in out["by K"][K].items()}, flush=True)
    with open(os.path.join(ROOT, "results", f"k_diagnostic_{sec}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def plot(secs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_sim import GRID, INK, MUTED, SURF
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    R = [json.load(open(os.path.join(ROOT, "results", f"k_diagnostic_{s}.json"))) for s in secs]
    fig, axes = plt.subplots(3, len(R), figsize=(5.4 * len(R), 13), squeeze=False)
    for c, r in enumerate(R):
        Ks = sorted(int(k) for k in r["by K"])
        shown = [k for k in Ks if k in K_SHOWN or k == r["K_rule"]]
        for i, g in enumerate(GENES):
            z = [r["by K"][str(k)][g]["z"] for k in shown]
            rk = [r["by K"][str(k)][g]["rank"] for k in shown]
            axes[0, c].plot(range(len(shown)), z, "o-", color=colors[i], label=g)
            axes[1, c].plot(range(len(shown)), rk, "o-", color=colors[i], label=g)
            axes[2, c].bar(np.arange(r["K_rule"]) + (i - 2) * 0.16, r["factor contributions r(gene,F_k) r(dopamine,F_k)"][g],
                           width=0.16, color=colors[i], label=g)
        labels = [f"{k}\n(rule)" if k == r["K_rule"] else str(k) for k in shown]
        for ax in axes[:2, c]:
            ax.set_xticks(range(len(shown)), labels); ax.set_xlabel("K (hidden factors removed)")
        axes[0, c].axhline(0, color=MUTED, lw=1); axes[0, c].axhspan(-1.96, 1.96, color=GRID, alpha=0.6)
        axes[0, c].set_ylabel("z (dopamine vs gene)")
        axes[1, c].set_yscale("log"); axes[1, c].invert_yaxis(); axes[1, c].set_ylabel(f"rank among {1000}+ genes")
        axes[2, c].axhline(0, color=MUTED, lw=1)
        axes[2, c].set_xticks(range(r["K_rule"]), [str(k + 1) for k in range(r["K_rule"])])
        axes[2, c].set_xlabel("factor k"); axes[2, c].set_ylabel("r(gene, F_k) × r(dopamine, F_k)")
        axes[0, c].set_title(r["section"], loc="left")
        for ax in axes[:, c]:
            ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        axes[0, c].legend(frameon=False, fontsize=8)
    fig.suptitle("Dopamine–gene associations vs number of removed hidden factors (mouse striatum, FMP-10)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig_real_K_diagnostic.png"), dpi=150)


if __name__ == "__main__":
    if sys.argv[1] == "--plot":
        plot(sys.argv[2:])
    else:
        for sec in sys.argv[2:]:
            run(sys.argv[1], sec)
