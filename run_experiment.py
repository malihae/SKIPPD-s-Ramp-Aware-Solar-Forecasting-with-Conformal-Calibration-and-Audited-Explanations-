"""Run the transfer-calibration experiment and save tables + figure to ./results.

Usage:
    python run_experiment.py            # 30 seeds (default)
    python run_experiment.py --seeds 5  # quick check
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tcdemo.experiment import run_experiment, summarise

ALPHA = 0.1


def make_figure(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    methods = ["B recalibrated", "B affine + conformal"]
    colors = {"B recalibrated": "tab:blue", "B affine + conformal": "tab:green"}

    naive = summary[summary.method == "B naive (reuse A quantile)"].iloc[0]
    a_in = summary[summary.method == "A in-distribution"].iloc[0]

    panels = [("coverage_mean", "Marginal coverage on target B"),
              ("coverage_high_noise_mean", "Coverage in high-noise region (x1 > 0.75)"),
              ("width_mean", "Mean interval width")]
    for ax, (col, title) in zip(axes, panels):
        for m in methods:
            d = summary[summary.method == m].sort_values("n_target")
            ax.plot(d.n_target, d[col], marker="o", color=colors[m], label=m)
        ax.axhline(naive[col], color="tab:red", ls="--", label="B naive (no target data)")
        if col == "coverage_mean":
            ax.axhline(a_in[col], color="gray", ls=":", label="A in-distribution")
            ax.axhline(1 - ALPHA, color="black", lw=0.8, alpha=0.5)
            ax.set_ylim(0.0, 1.05)
        elif col == "coverage_high_noise_mean":
            ax.axhline(1 - ALPHA, color="black", lw=0.8, alpha=0.5)
            ax.set_ylim(0.0, 1.05)
        ax.set_xscale("log")
        ax.set_xticks(sorted(summary[summary.n_target > 0].n_target.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Target-system measurements used (n)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.55))
    fig.suptitle(f"Transferring a model from system A to system B (target coverage {1 - ALPHA:.0%}, "
                 f"mean over {int(summary.n_seeds.max())} seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = run_experiment(n_seeds=args.seeds, alpha=ALPHA)
    summary = summarise(df)
    df.to_csv(os.path.join(args.out, "raw_results.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    make_figure(summary, os.path.join(args.out, "coverage_vs_target_samples.png"))

    show = summary.copy()
    for c in ["coverage_mean", "coverage_std", "coverage_high_noise_mean", "width_mean", "width_std"]:
        show[c] = show[c].round(3)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
