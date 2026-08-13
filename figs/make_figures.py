"""figures for the PhysBench benchmark paper."""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not (os.path.isdir(os.path.join(ROOT, "physx"))
           or os.path.isdir(os.path.join(ROOT, "src", "physx"))):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        break
    ROOT = parent
sys.path.insert(0, ROOT)
if os.path.isdir(os.path.join(ROOT, "src", "physx")):
    sys.path.insert(0, os.path.join(ROOT, "src"))   # standalone repo layout
OUT = HERE

plt.rcParams.update({"font.size": 9, "figure.dpi": 200})


def fig1_domains():
    from physx import dataset
    n = 4
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 4.4))
    for ax, dom in zip(axes.flat, ["beam", "cantilever", "projectile",
                                   "pendulum", "spring", "rc", "burgers",
                                   "heat2d"]):
        if dom in ("burgers", "heat2d"):
            ax.axis("off")
            ax.set_title(dom, fontsize=9)
            ax.text(0.5, 0.5, "field domain\n(see Fig. 3/4)", ha="center",
                    va="center", fontsize=8, color="#666666")
            continue
        smp = dataset.generate(dom, n=8, seed=0)
        for i in range(n):
            traj = np.asarray(smp[i]["traj"])  # (T, D)
            ax.plot(traj[:, 0], lw=0.9, alpha=0.75)
        ax.set_title(dom, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Representative trajectories from each benchmark domain "
                 "(shape-normalized)", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_domains.png"), bbox_inches="tight")
    plt.close(fig)


def fig2_matrix():
    files = sorted(glob.glob(os.path.join(ROOT, "physx", "models", "matrix",
                                          "matrix_*.json")))
    rows = {}
    for f in files:
        d = json.load(open(f))
        base = os.path.basename(f).replace(".stats.json", "")
        parts = base.split("_")
        dom, kind, seed = parts[1], parts[2], int(parts[3][1:])
        rows.setdefault((dom, kind), []).append(d["train_rel_mae"])
    doms = ["beam", "cantilever", "burgers", "heat2d", "projectile"]
    kinds = ["phys", "nophys", "mlp"]
    colors = {"phys": "#1a6f8e", "nophys": "#c0552c", "mlp": "#7a7a7a"}
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(len(doms))
    w = 0.26
    for j, k in enumerate(kinds):
        vals = [np.mean(rows[(d, k)]) for d in doms]
        errs = [np.std(rows[(d, k)]) for d in doms]
        ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=2.5,
               color=colors[k], alpha=0.9, label=k, error_kw=dict(lw=0.9))
    ax.set_xticks(x); ax.set_xticklabels(doms)
    ax.set_ylabel("mean training rel-MAE (5 seeds)")
    ax.set_title("Baseline matrix: 75 runs, 5 domains x 3 architectures x 5 seeds")
    ax.legend(frameon=False, fontsize=8.4)
    ax.grid(axis="y", color="#dddddd", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_matrix.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_domains()
    fig2_matrix()
    print("done")
