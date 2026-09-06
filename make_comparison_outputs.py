"""Builds the RQ3 positive-control comparison outputs: Marlene vs. GENIE3.

Run this AFTER run_genie3_sergio.py has produced
genie3/results/genie3_density_results_summary.csv.

Reads:
  - genie3/results/genie3_density_results_summary.csv (this baseline's own output)
  - marlene/Marlene/findings/marlene_density_results_summary.csv (located via the
    `find` used when this pipeline was built -- see genie3/README.md)

Writes (into genie3/results/):
  - model_comparison_summary.csv: the two summaries concatenated with an added
    `model` column ("Marlene" / "GENIE3"), ready to paste into the paper.
  - model_comparison_degradation_curve.pdf: both models' AUPRC/AUROC vs.
    number-of-timepoints curves overlaid on one set of axes, in the same
    Okabe-Ito style as marlene_density_degradation_curve.pdf, plus the shared
    random-baseline reference lines. Metric is encoded by color (blue=AUPRC,
    vermillion=AUROC, matching the existing chart); model is encoded by
    linestyle + marker shape (solid circle/square = Marlene, dashed
    diamond/triangle = GENIE3) so no new hues are introduced.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_GENIE3_SUMMARY = THIS_DIR / "results" / "genie3_density_results_summary.csv"
DEFAULT_MARLENE_SUMMARY = (
    THIS_DIR.parent / "genie3" / "marlene_density_results_summary.csv"
)
DEFAULT_OUT_DIR = THIS_DIR / "results"

# Same ground truth across all tiers/seeds/models (KNOWN SCHEMA), so these
# mirror the hardcoded constants in marlene/Marlene/aggregate_density_results.py
# (N_GROUND_TRUTH_EDGES=1155, N_TOTAL_LINKS=14800 for the 400-gene/37-TF SERGIO
# dataset) rather than being recomputed here.
N_GROUND_TRUTH_EDGES = 1155
N_TOTAL_LINKS = 14800
BASELINE_AUPRC = N_GROUND_TRUTH_EDGES / N_TOTAL_LINKS
BASELINE_AUROC = 0.5

COLOR_AUPRC = "#0072B2"  # blue, Okabe-Ito
COLOR_AUROC = "#D55E00"  # vermillion, Okabe-Ito


def load_summaries(genie3_path: Path, marlene_path: Path) -> pd.DataFrame:
    genie3_df = pd.read_csv(genie3_path)
    genie3_df["model"] = "GENIE3"
    marlene_df = pd.read_csv(marlene_path)
    marlene_df["model"] = "Marlene"
    return pd.concat([marlene_df, genie3_df], ignore_index=True)


def make_figure(combined_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            plt.style.use("ggplot")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    model_style = {
        "Marlene": dict(linestyle="-", auprc_marker="o", auroc_marker="s"),
        "GENIE3": dict(linestyle="--", auprc_marker="D", auroc_marker="^"),
    }

    for model, style in model_style.items():
        df = combined_df[combined_df["model"] == model].sort_values("n_timepoints")
        if df.empty:
            continue
        ax.errorbar(
            df["n_timepoints"], df["mean_auprc_mean"], yerr=df["mean_auprc_std"],
            marker=style["auprc_marker"], markersize=7, color=COLOR_AUPRC,
            linewidth=2, linestyle=style["linestyle"], capsize=5, capthick=1.5,
            label=f"{model} AUPRC (mean ± std)",
        )
        ax.errorbar(
            df["n_timepoints"], df["mean_auroc_mean"], yerr=df["mean_auroc_std"],
            marker=style["auroc_marker"], markersize=7, color=COLOR_AUROC,
            linewidth=2, linestyle=style["linestyle"], capsize=5, capthick=1.5,
            label=f"{model} AUROC (mean ± std)",
        )

    ax.axhline(BASELINE_AUPRC, color="gray", linestyle="--", linewidth=1.3,
               label="Random baseline (AUPRC)")
    ax.axhline(BASELINE_AUROC, color="gray", linestyle=":", linewidth=1.3,
               label="Random baseline (AUROC)")

    ax.set_xlabel("Number of Timepoints", fontsize=11)
    ax.set_ylabel("Score (AUPRC / AUROC)", fontsize=11)
    ax.set_title(
        "Marlene vs. GENIE3 Under Temporal Sparsity\n"
        "(SERGIO Synthetic Data, Positive-Control Comparison)",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)

    xticks = sorted(combined_df["n_timepoints"].unique())
    tier_by_n = dict(zip(combined_df["n_timepoints"], combined_df["tier"]))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{n} ({tier_by_n[n]})" for n in xticks])

    ax.legend(loc="best", frameon=True, framealpha=0.9, ncol=2)
    fig.tight_layout()

    pdf_path = out_dir / "model_comparison_degradation_curve.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genie3_summary", type=str, default=str(DEFAULT_GENIE3_SUMMARY))
    ap.add_argument("--marlene_summary", type=str, default=str(DEFAULT_MARLENE_SUMMARY))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_df = load_summaries(Path(args.genie3_summary), Path(args.marlene_summary))
    summary_path = out_dir / "model_comparison_summary.csv"
    combined_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    make_figure(combined_df, out_dir)


if __name__ == "__main__":
    main()
