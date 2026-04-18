"""Generate a publication-ready stacked bar chart of difficulty distribution.

Two-panel design:
  (a) Linear scale — shows the overall mass concentration at 0.2-0.3.
  (b) Log scale    — reveals the tail structure at higher difficulty.

Bars are stacked and colour-coded by dataset source.
"""

import json
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCORED_DIR   = "/data/troy/datasets/guac/scored"
CHARTQA_CKPT = os.path.join(SCORED_DIR, "chartqa_train.jsonl.ckpt")
HERE         = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PDF   = os.path.join(HERE, "..", "difficulty_distribution.pdf")
OUTPUT_PNG   = os.path.join(HERE, "..", "difficulty_distribution.png")

N_BINS     = 10
BIN_EDGES  = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_WIDTH  = BIN_EDGES[1] - BIN_EDGES[0]

DATASET_META = {
    "geometry3k": dict(label="Geometry3K", color="#4e79a7"),
    "scienceqa":  dict(label="ScienceQA",  color="#f28e2b"),
    "mathverse":  dict(label="MathVerse",  color="#e15759"),
    "chartqa":    dict(label="ChartQA",    color="#76b7b2"),
}
DATASET_ORDER = ["geometry3k", "scienceqa", "mathverse", "chartqa"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_records(path):
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_dataset(record_id):
    return record_id.split("_")[0]


def collect_difficulties():
    data = collections.defaultdict(list)
    for split in ("train", "val", "test"):
        path = os.path.join(SCORED_DIR, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipping.", file=sys.stderr)
            continue
        for rec in load_records(path):
            d = rec.get("difficulty")
            if d is not None:
                data[extract_dataset(rec["id"])].append(float(d))

    if os.path.exists(CHARTQA_CKPT):
        for rec in load_records(CHARTQA_CKPT):
            d = rec.get("difficulty")
            if d is not None:
                data["chartqa"].append(float(d))
    else:
        print(f"Warning: {CHARTQA_CKPT} not found — ChartQA omitted.", file=sys.stderr)

    return data


def bin_data(data):
    binned = {}
    for ds, diffs in data.items():
        counts, _ = np.histogram(diffs, bins=BIN_EDGES)
        binned[ds] = counts
    return binned


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------
def draw_bars(ax, binned, log_scale=False):
    bar_width = BIN_WIDTH * 0.82
    bottom = np.zeros(N_BINS)
    for ds in DATASET_ORDER:
        if ds not in binned:
            continue
        meta   = DATASET_META[ds]
        counts = binned[ds].astype(float)
        ax.bar(
            BIN_CENTERS, counts,
            width=bar_width,
            bottom=bottom,
            color=meta["color"],
            linewidth=0,
            zorder=3,
        )
        bottom += counts
    return bottom


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(binned):
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8.5,
        "axes.linewidth":    0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(7.2, 3.0),
        gridspec_kw={"wspace": 0.38},
    )

    # ---- Panel A: linear ----
    total_lin = draw_bars(ax_lin, binned, log_scale=False)
    ax_lin.set_xlabel("Difficulty Score", labelpad=3)
    ax_lin.set_ylabel("Number of Examples", labelpad=3)
    ax_lin.set_title("(a) Linear Scale", pad=4, fontweight="bold")
    ax_lin.set_xticks(np.round(BIN_EDGES, 2))
    ax_lin.set_xticklabels([f"{v:.1f}" for v in BIN_EDGES], rotation=45, ha="right")
    ax_lin.set_xlim(BIN_EDGES[0] - BIN_WIDTH * 0.55, BIN_EDGES[-1] + BIN_WIDTH * 0.55)
    ax_lin.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55, zorder=0)
    ax_lin.set_axisbelow(True)
    # label tallest bar only
    idx_max = int(np.argmax(total_lin))
    ax_lin.text(
        BIN_CENTERS[idx_max], total_lin[idx_max] + total_lin.max() * 0.012,
        f"{int(total_lin[idx_max]):,}",
        ha="center", va="bottom", fontsize=7, color="#222222",
    )

    # ---- Panel B: log ----
    total_log = draw_bars(ax_log, binned, log_scale=True)
    ax_log.set_yscale("log")
    ax_log.set_xlabel("Difficulty Score", labelpad=3)
    ax_log.set_ylabel("Number of Examples (log scale)", labelpad=3)
    ax_log.set_title("(b) Log Scale", pad=4, fontweight="bold")
    ax_log.set_xticks(np.round(BIN_EDGES, 2))
    ax_log.set_xticklabels([f"{v:.1f}" for v in BIN_EDGES], rotation=45, ha="right")
    ax_log.set_xlim(BIN_EDGES[0] - BIN_WIDTH * 0.55, BIN_EDGES[-1] + BIN_WIDTH * 0.55)
    ax_log.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.55, zorder=0)
    ax_log.set_axisbelow(True)
    ax_log.set_ylim(bottom=0.8)
    for x, t in zip(BIN_CENTERS, total_log):
        if t > 0:
            ax_log.text(x, t * 1.25, f"{int(t):,}",
                        ha="center", va="bottom", fontsize=6, color="#222222")

    # ---- Legend (on log panel) ----
    legend_patches = [
        mpatches.Patch(facecolor=DATASET_META[ds]["color"],
                       label=DATASET_META[ds]["label"])
        for ds in reversed(DATASET_ORDER)
        if ds in binned
    ]
    ax_log.legend(
        handles=legend_patches,
        loc="upper right",
        frameon=True, framealpha=0.92, edgecolor="#bbbbbb",
        handlelength=1.1, handleheight=0.85,
        borderpad=0.55, labelspacing=0.30,
    )

    fig.suptitle(
        "Data Difficulty Distribution by Dataset",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading difficulty data …")
    data = collect_difficulties()
    for ds, diffs in sorted(data.items()):
        arr = np.array(diffs)
        print(f"  {ds:>12s}: {len(arr):6,d} examples  "
              f"(mean={arr.mean():.3f}, min={arr.min():.2f}, max={arr.max():.2f})")

    binned = bin_data(data)

    print("\nGenerating figure …")
    fig = make_figure(binned)

    for out_path in (OUTPUT_PDF, OUTPUT_PNG):
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"  → saved: {out_path}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
