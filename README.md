# genie3/ — GENIE3 positive-control baseline

This folder is **not one of the 4 benchmark models** in the study. It exists
to answer one question: Marlene's SERGIO density-sweep results came back at
chance level (AUROC≈0.50, flat across all 3 temporal-density tiers). Is that
a real model failure, or a bug in the data pipeline / eval harness?

GENIE3 (Huynh-Thu et al., 2010) is a well-established classical GRN-inference
method that reliably beats chance on GRN benchmarks (it's a standard baseline
throughout the BEELINE benchmark literature). Running it through the exact
same data and eval code Marlene was judged by is a **positive control**:

- If GENIE3 also comes back at chance level → the harness (data prep, gt
  edges, or eval code) is suspect, not Marlene.
- If GENIE3 clearly beats chance → the harness is fine, and Marlene's
  chance-level results reflect a real model limitation.

`eval_utils.py`'s `compute_auprc_auroc` is copied **verbatim** from the
Marlene eval code specifically so this comparison is apples-to-apples.

## Isolation

This folder is fully decoupled from the Marlene pipeline's environment, per
the proposal's isolated-environments requirement: it imports nothing from
the `marlene` package, `torch`, or `train_sergio.py`. Dependencies are
limited to `numpy`, `pandas`, `anndata`, `scikit-learn`, `arboreto`, and
`matplotlib`.

## Data location

Prepared SERGIO datasets (`data_tier{N}_seed{S}/SERGIO-Marlene.h5ad` +
`sergio_gt_edges.csv`) live wherever `sergio_prepare_data.py` /
`train_sergio.py` / `run_density_experiment.py` live — confirmed at build
time to be `marlene/Marlene/`, a sibling of this `genie3/` folder. Both
driver scripts here default `--data_root` to `../marlene/Marlene` (resolved
relative to this file, so it works regardless of your current directory).

Ground truth is identical across all tiers/seeds/models — only the amount of
data available differs — so the random-baseline formula
(`n_true_edges / (n_genes * n_tfs)` for AUPRC, `0.5` for AUROC) and the
concrete constants used in the comparison plot (1155 true edges / 14800
total TF×target links, matching
`marlene/Marlene/aggregate_density_results.py`) are the same across models.

## Setup

```bash
cd genie3
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas anndata scikit-learn matplotlib arboreto
# add --break-system-packages only if pip complains about an externally
# managed environment and you are NOT using a venv
```

## Run

```bash
# 1. Run GENIE3 over all 9 (tier, seed) combos (this is the slow step —
#    it's meant for Kaggle/a real machine, not a quick local smoke test).
python run_genie3_sergio.py \
    --data_root ../marlene/Marlene \
    --results_root results \
    --log_file results/genie3_density_experiment_log.txt

# optional: lower n_estimators if a full run proves too slow
python run_genie3_sergio.py --n_estimators 100

# 2. Build the RQ3 comparison table + figure (after step 1 has produced
#    results/genie3_density_results_summary.csv).
python make_comparison_outputs.py
```

## Outputs (`genie3/results/`)

- `rankedEdges_tier{N}_seed{S}.csv` — full ranked GENIE3 edge table per combo
  (supplementary material).
- `genie3_raw_results.csv` — one row per (tier, seed): tier, n_timepoints,
  seed, auprc, auroc.
- `genie3_density_results_summary.csv` — one row per tier, same column
  schema as `marlene_density_results_summary.csv` (tier, n_timepoints,
  mean_auprc_mean, mean_auprc_std, mean_auroc_mean, mean_auroc_std, n_seeds)
  so the two can be concatenated directly.
- `genie3_density_experiment_log.txt` — timestamped run log, same style as
  `marlene_density_experiment_log.txt`.
- `model_comparison_summary.csv` — Marlene + GENIE3 summaries concatenated
  with a `model` column; the RQ3 comparison table for the paper.
- `model_comparison_degradation_curve.pdf` — both models' AUPRC/AUROC vs.
  number-of-timepoints curves overlaid on one set of axes, with the shared
  random-baseline reference lines. The positive-control figure for the paper.

## Dependency

[arboreto](https://arboreto.readthedocs.io/) — the maintained, pip-installable
reference reimplementation of GENIE3 used throughout the BEELINE benchmark
literature. `arboreto.algo.genie3` uses a Dask `LocalCluster` internally, so
`run_genie3_sergio.py` must be run as a script (`python run_genie3_sergio.py`),
never imported.
