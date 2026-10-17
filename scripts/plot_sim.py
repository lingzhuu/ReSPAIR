"""模拟结果作图，输出到 results/figures/。

需要先运行 run_sim.py 5 和 sweep_sim.py 2（图 2 用 results/sim_5seeds.json，图 4 用 results/sweep.json）。
用法：python3 scripts/plot_sim.py
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from respair import core as sj
from respair.simulate import simulate

FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# 配色：分类色按固定顺序；发散色 蓝–灰–红
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#5f5e5a", "#8a8984", "#e6e5e0", "#fcfcfb"
DIVERGE = LinearSegmentedColormap.from_list("bgr", ["#104281", "#3987e5", "#f0efec", "#e66767", "#9b1c1c"])

plt.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def spatial(ax, v, side, title):
    v = (v - v.mean()) / v.std()
    ax.imshow(v.reshape(side, side), cmap=DIVERGE, vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scatter(ax, x, y, title, color):
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, s=4, color=color, alpha=0.35, linewidths=0)
    ax.set_title(f"{title}\nr = {r:.2f}", fontsize=10.5)
    ax.set_xlabel("基因"); ax.set_ylabel("代谢物")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


def fig1(d, fit, Yg, Ym):
    """一对受混杂影响但没有联系的配对，和一对真实联系，扣除前后对比。"""
    side = int(np.sqrt(len(d["coords"])))
    conf_null = d["conf_pair"] & ~d["truth"]
    raw_r = np.abs(sj.standardize(Yg).T @ sj.standardize(Ym) / len(Yg))
    j, k = np.unravel_index(np.argmax(np.where(conf_null, raw_r, 0)), raw_r.shape)
    tj, tk = np.argwhere(d["truth"])[np.argmax(np.abs(fit["z"][d["truth"]]))]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    spatial(axes[0, 0], d["H"], side, "未标注的混杂（类似缺氧梯度）")
    spatial(axes[0, 1], Yg[:, j], side, f"基因 {j}（受混杂影响）\n原始")
    spatial(axes[0, 2], Ym[:, k], side, f"代谢物 {k}（受混杂影响）\n原始")
    scatter(axes[0, 3], Yg[:, j], Ym[:, k], "二者没有真实联系，但原始数据高度相关", SERIES[1])

    zjk = fit["z"][j, k]
    axes[1, 0].axis("off")
    axes[1, 0].text(0.02, 0.62, "简化版扣除已知结构和\n隐藏模式之后：", fontsize=11, color=INK, va="center")
    axes[1, 0].text(0.02, 0.30, f"这对配对的 z = {zjk:.2f}\n（不显著，误报消失）", fontsize=11, color=INK2, va="center")
    spatial(axes[1, 1], fit["Eg"][:, j], side, f"基因 {j}\n残差")
    spatial(axes[1, 2], fit["Em"][:, k], side, f"代谢物 {k}\n残差")
    scatter(axes[1, 3], fit["Eg"][:, tj], fit["Em"][:, tk],
            f"对照：一对真实联系的残差\n(基因 {tj} – 代谢物 {tk}, z = {fit['z'][tj, tk]:.1f})", SERIES[0])
    fig.suptitle("扣除隐藏模式前后：假相关消失，真实联系保留", fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "空间图颜色：标准化后的数值，蓝 = 低，红 = 高。", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig1_示意.png"), dpi=160)
    plt.close(fig)


LABEL = {
    "普通相关": "普通相关\n(Desium / SpatialMETA 类)",
    "MANTIS-SCI": "MANTIS-SCI\n(不扣结构)",
    "MANTIS-SPC-CT": "MANTIS-SPC-CT\n(扣细胞类型)",
    "简化版": "简化版（我们）",
    "只扣已知结构+普通相关": "只扣已知结构\n+ 普通相关",
    "只扣已知结构+空间检验": "只扣已知结构\n+ 空间检验",
    "扣隐藏模式+普通相关": "扣隐藏模式\n+ 普通相关",
    "简化版(未校准)": "简化版\n(未做负对照校准)",
}


def fig2():
    from run_sim import MAIN, ABLATION
    res = json.load(open(os.path.join(ROOT, "results", "sim_5seeds.json")))
    avg, sd = res["平均"], res["标准差"]
    methods = MAIN + ABLATION
    gap = 0.8
    y = []
    pos = 0.0
    for i, m in enumerate(methods):
        if i == len(MAIN):
            pos += gap
        y.append(pos); pos += 1
    y = -np.array(y)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})
    h = 0.38
    for i, (key, name) in enumerate([("混杂配对误报率", "受混杂影响的零配对"), ("其他零配对误报率", "其他零配对")]):
        v = np.array([avg[m][key] for m in methods]) * 100
        e = np.array([sd[m][key] for m in methods]) * 100
        yy = y + (h / 2 + 0.02) * (1 - 2 * i)
        axes[0].barh(yy, v, height=h, color=SERIES[i], label=name, xerr=e,
                     error_kw={"ecolor": INK2, "lw": 1, "capsize": 0})
        for a, b in zip(yy, v):
            axes[0].text(b + 2, a, f"{b:.1f}%", va="center", fontsize=8.5, color=INK2)
    axes[0].axvline(5, color=INK, lw=1, ls="--")
    axes[0].set_title("误报率（p < 0.05，虚线 = 5%）", loc="left")
    axes[0].set_xlim(0, 105)
    axes[0].legend(frameon=False, loc="lower right")
    for ax, key, ref, title in [(axes[1], "BH实际FDR", 10, "BH(0.1) 后的实际 FDR（虚线 = 10%）"),
                                (axes[2], "BH检出率", None, "BH(0.1) 后真实联系的检出率")]:
        v = np.array([avg[m][key] for m in methods]) * 100
        e = np.array([sd[m][key] for m in methods]) * 100
        ax.barh(y, v, height=0.55, color=SERIES[0], xerr=e, error_kw={"ecolor": INK2, "lw": 1, "capsize": 0})
        for a, b in zip(y, v):
            ax.text(b + 2, a, f"{b:.1f}%", va="center", fontsize=8.5, color=INK2)
        if ref:
            ax.axvline(ref, color=INK, lw=1, ls="--")
        ax.set_title(title, loc="left"); ax.set_xlim(0, 115)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[m] for m in methods])
    for t, m in zip(axes[0].get_yticklabels(), methods):
        if m == "简化版":
            t.set_fontweight("bold"); t.set_color(INK)
    axes[0].text(-0.02, y[0] + 0.75, "已有方法 vs 我们", transform=axes[0].get_yaxis_transform(), ha="right",
                 fontsize=10.5, fontweight="bold")
    axes[0].text(-0.02, y[len(MAIN)] + 0.75, "消融：我们方法的每一步", transform=axes[0].get_yaxis_transform(),
                 ha="right", fontsize=10.5, fontweight="bold")
    for ax in axes:
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel("%")
    fig.suptitle("Benchmark：各方法在含未标注共享混杂的模拟数据上的表现（5 个种子，误差线 = 标准差）",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "fig2_benchmark.png"), dpi=160)
    plt.close(fig)


def fig3(d, P):
    """零配对 p 值的 QQ 图。"""
    null = ~d["truth"]
    names = ["普通相关", "MANTIS-SCI", "MANTIS-SPC-CT", "只扣已知结构+空间检验", "简化版"]
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    m = int(null.sum())
    lim = -np.log10(0.5 / m)
    ends = []
    for i, name in enumerate(names):
        p = np.sort(np.clip(P[name][null], 1e-300, 1))
        idx = np.unique(np.concatenate([np.arange(2000), np.geomspace(1, m, 3000).astype(int) - 1]))
        exp = -np.log10((idx + 0.5) / m); obs = np.minimum(-np.log10(p[idx]), 30)
        ax.plot(exp, obs, color=SERIES[i], lw=2)
        ends.append([obs[0], name, SERIES[i]])
    # 右侧直接标注，避免重叠
    ends.sort(key=lambda t: t[0])
    last = -10
    for val, name, col in ends:
        yv = max(val, last + 1.6); last = yv
        ax.plot([lim, lim + 0.15], [val, yv], color=col, lw=1)
        ax.text(lim + 0.2, yv, LABEL[name].replace("\n", " "), color=INK, fontsize=9, va="center")
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("期望 −log10(p)"); ax.set_ylabel("实际 −log10(p)（上限截到 30）")
    ax.set_xlim(0, lim + 3.6); ax.set_ylim(0, 32)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("零配对的 p 值（QQ 图，种子 0）：越贴近灰色对角线越好", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_QQ图.png"), dpi=160)
    plt.close(fig)


def fig4():
    sw = json.load(open(os.path.join(ROOT, "results", "sweep.json")))
    methods = ["普通相关", "MANTIS-SCI", "MANTIS-SPC-CT", "只扣已知结构+空间检验", "简化版"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    conf = sw["混杂强度"]; xs = [float(c) for c in conf]
    ends = []
    for i, m in enumerate(methods):
        ys = [conf[c][m]["混杂配对误报率"] * 100 for c in conf]
        axes[0].plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=8, mec=SURF, mew=2)
        ends.append([ys[-1], m])
    ends.sort(); last = -10
    for val, m in ends:
        yv = max(val, last + 5); last = yv
        axes[0].text(xs[-1] + 0.04, yv, LABEL[m].replace("\n", " "), fontsize=8.5, va="center")
    axes[0].axhline(5, color=INK, lw=1, ls="--")
    axes[0].set_xlim(-0.05, 1.75); axes[0].set_ylim(0, 105)
    axes[0].set_xlabel("未标注混杂的强度（0 = 没有混杂）"); axes[0].set_ylabel("受混杂影响的零配对误报率 (%)")
    axes[0].set_title("混杂越强，其他方法误报越多（虚线 = 5%）", loc="left")

    link = sw["联系强度"]; xs = [float(w) for w in link]
    pw = [link[w]["简化版"]["BH检出率"] * 100 for w in link]
    fd = [link[w]["简化版"]["BH实际FDR"] * 100 for w in link]
    axes[1].plot(xs, pw, color=SERIES[0], lw=2, marker="o", ms=8, mec=SURF, mew=2)
    axes[1].plot(xs, fd, color=SERIES[1], lw=2, marker="o", ms=8, mec=SURF, mew=2)
    axes[1].text(xs[-1] + 0.01, pw[-1], "检出率", fontsize=9.5, va="center")
    axes[1].text(xs[-1] + 0.01, fd[-1], "实际 FDR", fontsize=9.5, va="center")
    for x, v in zip(xs, pw):
        axes[1].text(x, v + 4, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK2)
    axes[1].axhline(10, color=INK, lw=1, ls="--")
    axes[1].set_xlim(0.08, 0.6); axes[1].set_ylim(0, 110)
    axes[1].set_xlabel("真实联系强度（基因残差变化 1 个标准差时代谢物 log 强度的变化）")
    axes[1].set_ylabel("%")
    axes[1].set_title("简化版：检出率随联系强度变化（BH 0.1，虚线 = 10%）", loc="left")
    for ax in axes:
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_强度扫描.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run_sim import all_pvals
    d = simulate(seed=0)
    P, fit, (Yg, Ym) = all_pvals(d)
    fig1(d, fit, Yg, Ym)
    fig2()
    fig3(d, P)
    if os.path.exists(os.path.join(ROOT, "results", "sweep.json")):
        fig4()
