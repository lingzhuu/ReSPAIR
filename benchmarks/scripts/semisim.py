"""Semi-synthetic benchmark on a real tissue section (SMA mouse brain, Visium + MALDI-MSI).

Genes are the real counts; metabolites keep the large-scale maps of real metabolites, with real residual variance,
real zeros and planted links to the denoised expression of real genes (see `simulate.semi_synthetic`). Run prepare_sma.py first.
Scenarios: links on the gene's whole expression (not ReSPAIR's model) at two strengths, and links on the gene's
fine-scale part only (ReSPAIR's model) as reference.

Usage: python benchmarks/scripts/semisim.py <aligned.npz> [n_seeds, default 5]
Writes results/semisim.json and results/figures/fig_semisim_ROC.png.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from simulate import semi_synthetic
from compare import GRID_FPR, METHODS, roc_stats, scores
from plot_sim import GRID, INK, MUTED, SURF

SCENARIOS = {
    "links on whole gene expression, strength 0.5": {"link_on": "total", "link_strength": 0.5},
    "links on whole gene expression, strength 0.3": {"link_on": "total", "link_strength": 0.3},
    "links on whole gene expression, strength 0.15": {"link_on": "total", "link_strength": 0.15},
    "links on fine-scale gene signal, strength 0.5": {"link_on": "local", "link_strength": 0.5},
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#1f1f1e"]

if __name__ == "__main__":
    src = np.load(sys.argv[1])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out, curves = {}, {}
    for name, kw in SCENARIOS.items():
        rows = {m: {"all pairs": [], "confounded pairs only": []} for m in METHODS}
        for seed in range(n_seeds):
            d = semi_synthetic(src["G"], src["M"], src["coords"], seed=seed, **kw)
            S = scores(seed, kw["link_strength"], METHODS, d, tag=f"_sma_{kw['link_on']}")
            for sname, neg in {"all pairs": ~d["truth"], "confounded pairs only": d["conf_pair"]}.items():
                for m in METHODS:
                    rows[m][sname].append(roc_stats(S[m], d["truth"], neg))
        out[name] = {m: {s: {k: float(np.mean([r[k] for r in v])) for k in ["AUC", "TPR@5%"]} for s, v in r.items()}
                     for m, r in rows.items()}
        curves[name] = {m: {s: np.mean([r["curve"] for r in v], 0) for s, v in r.items()} for m, r in rows.items()}
        print(name, flush=True)
        for s in ["all pairs", "confounded pairs only"]:
            print(" ", s, {m: (round(out[name][m][s]["AUC"], 2), round(out[name][m][s]["TPR@5%"], 2)) for m in METHODS})
    with open(os.path.join(ROOT, "results", "semisim.json"), "w") as f:
        json.dump({"data": "SMA mouse brain V11L12-038_D1 (Vicari et al. 2024)", "seeds": n_seeds,
                   "mean over seeds": out}, f, indent=1)

    fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(5.6 * len(SCENARIOS), 11))
    for c, name in enumerate(SCENARIOS):
        for r, sname in enumerate(["all pairs", "confounded pairs only"]):
            ax = axes[r, c]
            for i, m in enumerate(METHODS):
                v = out[name][m][sname]
                ax.plot(GRID_FPR, curves[name][m][sname], color=COLORS[i], lw=2.6 if m == "ReSPAIR" else 1.6,
                        label=f"{m}  AUC {v['AUC']:.2f}, power {v['TPR@5%']:.0%}")
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
            ax.axvline(0.05, color=INK, lw=1, ls=":")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
            ax.set_xlabel("false-positive rate"); ax.set_ylabel("power")
            vs = "all unlinked pairs" if sname == "all pairs" else "confounded unlinked pairs"
            ax.set_title(f"{name}\nvs {vs}", loc="left", fontsize=10.5)
            ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_facecolor(SURF)
            ax.legend(frameon=False, loc="lower right", fontsize=7.8)
    fig.suptitle(f"Semi-synthetic ROC on a real SMA section (power at 5% FPR; mean of {n_seeds} seeds)",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(ROOT, "results", "figures", "fig_semisim_ROC.png"), dpi=150)
