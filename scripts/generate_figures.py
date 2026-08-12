"""根据已发布的 CSV/JSON 结果文件重新生成论文图表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"static": "#C56F28", "dynasid": "#35658F", "neutral": "#B9C0C8"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#E3E7EB", "grid.linewidth": 0.7, "font.size": 9,
    })


def cross_city(results: Path, out: Path) -> None:
    rows = []
    for city in ("NYC", "TKY"):
        frame = pd.read_csv(results / "cross_city" / city / "extended_summary.csv")
        rows.append(frame[frame.method.isin(["Static-GNPR-SID", "DynaSID-v8"])])
    data = pd.concat(rows, ignore_index=True)
    metrics = ["Acc@1", "Recall@10", "NDCG@10", "Infeasible@10"]
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.8))
    x = np.arange(2); width = 0.34
    for axis, metric in zip(axes, metrics):
        static = [data[(data.city == c) & (data.method == "Static-GNPR-SID")][metric].iloc[0] for c in ("NYC", "TKY")]
        dynamic = [data[(data.city == c) & (data.method == "DynaSID-v8")][metric].iloc[0] for c in ("NYC", "TKY")]
        axis.bar(x - width / 2, static, width, color=COLORS["static"], label="Static SID")
        axis.bar(x + width / 2, dynamic, width, color=COLORS["dynasid"], label="DynaSID-v8")
        axis.set_xticks(x, ["NYC", "TKY"]); axis.set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out / "cross_city_confirmation.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def sid_quality(results: Path, out: Path) -> None:
    nyc = json.loads((results / "sid_quality" / "sid_quality_metrics.json").read_text(encoding="utf-8"))
    tky = json.loads((results / "tky_sid" / "sid_quality_metrics.json").read_text(encoding="utf-8"))
    collision = [nyc["collision_rate"], tky["collision_rate"]]
    fig, axis = plt.subplots(figsize=(4.8, 3.1))
    bars = axis.bar(["NYC", "TKY"], collision, color=[COLORS["dynasid"], COLORS["static"]], width=0.55)
    axis.set_ylabel("SID collision rate")
    for bar, value in zip(bars, collision):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.1%}", ha="center")
    fig.tight_layout(); fig.savefig(out / "sid_collision_rate.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    style(); cross_city(args.results_dir, args.out); sid_quality(args.results_dir, args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
