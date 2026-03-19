#!/usr/bin/env python3
"""Generate aggregate analysis plots from WiLoR evaluation results.

Reads a results JSON (from run_eval_wilor.py) and produces publication-quality
figures summarizing off-the-shelf WiLoR performance on POV-Surgery.

Usage:
    python visualize_results.py --results results/wilor_crop_results.json
    python visualize_results.py --results results/wilor_crop_results.json --output-dir results/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Paper-quality defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
})


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    detected = [m for m in data["per_frame"] if m.get("detected", False)]
    return data, detected


def plot_metric_bar(detected, summary, output_dir, model_name="WiLoR"):
    """Bar chart of our aggregate metrics."""
    metrics = ["mpjpe", "pa_mpjpe", "pve", "pa_pve"]
    metric_labels = ["MPJPE", "PA-MPJPE", "PVE", "PA-PVE"]
    keys = ["mpjpe_mean", "pa_mpjpe_mean", "pve_mean", "pa_pve_mean"]
    vals = [summary[k] for k in keys]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(metric_labels, vals, color=colors, edgecolor="white", width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Error (mm)")
    ax.set_title(f"{model_name} Off-the-Shelf on POV-Surgery (Crop-Regress)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "metric_bar.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved metric_bar.png")


def plot_pa_mpjpe_histogram(detected, output_dir):
    """Histogram of per-frame PA-MPJPE distribution."""
    vals = np.array([m["pa_mpjpe"] for m in detected])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Mean: {vals.mean():.1f} mm")
    ax.axvline(np.median(vals), color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Median: {np.median(vals):.1f} mm")
    ax.set_xlabel("PA-MPJPE (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"PA-MPJPE Distribution (n={len(vals)} frames)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "pa_mpjpe_histogram.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pa_mpjpe_histogram.png")


def plot_mpjpe_over_frames(detected, output_dir):
    """Line plot of MPJPE over frame index with rolling average."""
    vals = np.array([m["mpjpe"] for m in detected])
    indices = np.arange(len(vals))

    window = min(50, len(vals) // 5) if len(vals) > 10 else 1
    rolling = np.convolve(vals, np.ones(window) / window, mode="valid")
    rolling_x = indices[window - 1:]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(indices, vals, alpha=0.25, color="#3498db", linewidth=0.5, label="Per-frame")
    ax.plot(rolling_x, rolling, color="#e74c3c", linewidth=1.5,
            label=f"Rolling avg (w={window})")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("MPJPE Over Frames")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "mpjpe_over_frames.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mpjpe_over_frames.png")


def plot_per_finger_boxplot(detected, output_dir):
    """Box plot of per-finger MPJPE."""
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    finger_data = {fn: [m["per_finger"][fn] for m in detected] for fn in finger_names}

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [finger_data[fn] for fn in finger_names],
        tick_labels=[n.capitalize() for n in finger_names],
        patch_artist=True,
        showfliers=False,
    )
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    means = [np.mean(finger_data[fn]) for fn in finger_names]
    ax.scatter(range(1, 6), means, marker="D", color="black", s=30, zorder=5, label="Mean")

    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Per-Finger MPJPE (Root-Relative)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_dir / "per_finger_boxplot.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved per_finger_boxplot.png")


def plot_mpjpe_vs_pa_mpjpe(detected, output_dir):
    """Scatter plot: MPJPE vs PA-MPJPE per frame, colored by frame index."""
    mpjpe_vals = np.array([m["mpjpe"] for m in detected])
    pa_mpjpe_vals = np.array([m["pa_mpjpe"] for m in detected])
    indices = np.arange(len(detected))

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pa_mpjpe_vals, mpjpe_vals, c=indices, cmap="viridis",
                    s=8, alpha=0.6, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Frame Index")

    lim = max(mpjpe_vals.max(), pa_mpjpe_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("PA-MPJPE (mm)")
    ax.set_ylabel("MPJPE (mm)")
    ax.set_title("Position Error vs Shape Error per Frame")
    ax.grid(alpha=0.3)
    fig.savefig(output_dir / "error_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved error_scatter.png")


def plot_summary_table(detected, summary, output_dir, model_name="WiLoR"):
    """Render the results summary table as a clean PNG."""
    cols = ["MPJPE", "PA-MPJPE", "PVE", "PA-PVE", "P2D (px)"]
    keys = ["mpjpe_mean", "pa_mpjpe_mean", "pve_mean", "pa_pve_mean", "p2d_mean"]

    # Mean row
    mean_row = [f"{summary[k]:.2f}" for k in keys]

    # Compute median/std from per-frame data
    metric_keys = ["mpjpe", "pa_mpjpe", "pve", "pa_pve", "p2d"]
    median_row = [f"{np.median([m[k] for m in detected]):.2f}" for k in metric_keys]
    std_row = [f"{np.std([m[k] for m in detected]):.2f}" for k in metric_keys]

    rows = [mean_row, median_row, std_row]
    row_labels = ["Mean", "Median", "Std"]

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=[f"{c} (mm)" if "px" not in c else c for c in cols],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Highlight mean row
    for j in range(len(cols)):
        table[1, j].set_facecolor("#d4edff")
    table[1, -1].set_facecolor("#d4edff")

    ax.set_title(f"{model_name} Off-the-Shelf on POV-Surgery — Summary Statistics", pad=20)
    fig.savefig(output_dir / "summary_table.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary_table.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize WiLoR eval results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to wilor_*_results.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/figures/)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    data, detected = load_results(results_path)
    summary = data["metrics_summary"]

    if not detected:
        print("ERROR: No detected frames in results file.")
        return

    # Infer model name from results JSON
    model_label = data.get("model", "")
    if "hamer" in model_label.lower() or "hamer" in results_path.name.lower():
        model_name = "HaMeR"
    else:
        model_name = "WiLoR"

    print(f"Loaded {len(detected)} detected frames from {results_path.name}")
    print(f"Generating figures in {output_dir}/\n")

    plot_metric_bar(detected, summary, output_dir, model_name=model_name)
    plot_pa_mpjpe_histogram(detected, output_dir)
    plot_mpjpe_over_frames(detected, output_dir)
    plot_per_finger_boxplot(detected, output_dir)
    plot_mpjpe_vs_pa_mpjpe(detected, output_dir)
    plot_summary_table(detected, summary, output_dir, model_name=model_name)

    print(f"\nDone! 6 figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
