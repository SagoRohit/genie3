"""GENIE3 classical-baseline positive control on the SERGIO density-tier data.

Mirrors run_density_experiment.py's tier x seed loop (9 combos: 3 density
tiers x 3 seeds) so the results are a drop-in comparison against Marlene's.
Uses arboreto's GENIE3 implementation (Huynh-Thu et al. 2010), the reference
reimplementation used throughout the BEELINE benchmark literature.

This folder is fully decoupled from the Marlene pipeline: only numpy,
pandas, anndata, scikit-learn, arboreto, and matplotlib are used — nothing
from the `marlene` package, torch, or train_sergio.py.

Usage:
    python run_genie3_sergio.py [--data_root PATH] [--results_root PATH]
                                 [--n_estimators N] [--log_file PATH]

Must be run as a script (not imported) because arboreto's genie3() spins up
a Dask distributed LocalCluster internally.
"""

import argparse
import datetime
import time
from pathlib import Path

import anndata
import pandas as pd

from eval_utils import build_A_full, compute_auprc_auroc

TIERS = {1: 15, 2: 5, 3: 3}
N_SEEDS = 3

THIS_DIR = Path(__file__).resolve().parent
# The prepared SERGIO data (data_tier{N}_seed{S}/) lives wherever
# sergio_prepare_data.py / train_sergio.py / run_density_experiment.py live,
# confirmed via `find` to be marlene/Marlene/, a sibling of this genie3/ dir.
DEFAULT_DATA_ROOT = THIS_DIR.parent / "marlene" / "Marlene"
DEFAULT_RESULTS_ROOT = THIS_DIR / "results"
DEFAULT_LOG_FILE = DEFAULT_RESULTS_ROOT / "genie3_density_experiment_log.txt"


def log(log_file, msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def load_combo(data_root, tier, seed):
    data_dir = Path(data_root) / f"data_tier{tier}_seed{seed}"
    adata = anndata.read_h5ad(data_dir / "SERGIO-Marlene.h5ad")
    gt_df = pd.read_csv(data_dir / "sergio_gt_edges.csv")
    gt_edges = set(zip(gt_df["regulator"], gt_df["target"]))
    return adata, gt_edges


def _create_graph_scatter_fixed(expression_matrix, gene_names, tf_names, regressor_type,
                                 regressor_kwargs, client, target_genes='all', limit=None,
                                 include_meta=False, early_stop_window_length=None, seed=666,
                                 repartition_multiplier=1):
    """Drop-in replacement for arboreto.core.create_graph (arboreto/core.py,
    installed package, NEVER edited in place -- vendored precedent is to
    patch the caller, same as PseudoGRN's _smooth_pandas_compat). Every
    line is identical to the original EXCEPT one: each target gene's
    expression column is scattered to workers ONCE up front
    (client.scatter(target_columns, broadcast=False), the SAME
    scatter-not-embed pattern arboreto's own code already uses for the TF
    matrix a few lines above) instead of embedded directly into a fresh
    delayed(expression_matrix[:, target_gene_index], pure=True) closure
    per target gene.

    WHY: confirmed by reading arboreto's actual source that the original
    per-target delayed() call embeds that column's raw data BY VALUE into
    the Dask task graph, once per target gene (400 times here). At real
    SERGIO cell counts (8,100-40,500 cells) this produces a very large
    graph (Dask's own "Sending large graph of size NN MiB" warning,
    observed directly on Kaggle at both 40,500 and 15,000 cells) --
    serialization/scheduling overhead from this, not RF-fitting compute,
    is the actual bottleneck (RF_KWARGS already sets n_jobs=1 per task,
    ruling out a nested-parallelism/oversubscription explanation).

    This change is PURELY a data-transport optimization: same regressor,
    same target/TF assignment, same seed, same algorithm -- it cannot
    change GENIE3's output, only how the input data reaches workers. Not
    the same class of risk as a GPU rewrite (which would use a different
    numerical implementation) -- this is why it was judged safe to apply
    under today's time pressure where the GPU path was not.
    """
    from dask import delayed
    from dask.dataframe import from_delayed
    from arboreto.core import to_tf_matrix, infer_partial_network, target_gene_indices, _GRN_SCHEMA, _META_SCHEMA, EARLY_STOP_WINDOW_LENGTH as _DEFAULT_ESWL

    if early_stop_window_length is None:
        early_stop_window_length = _DEFAULT_ESWL

    assert expression_matrix.shape[1] == len(gene_names)
    assert client, "client is required"

    tf_matrix, tf_matrix_gene_names = to_tf_matrix(expression_matrix, gene_names, tf_names)
    future_tf_matrix = client.scatter(tf_matrix, broadcast=True)
    [future_tf_matrix_gene_names] = client.scatter([tf_matrix_gene_names], broadcast=True)

    indices = list(target_gene_indices(gene_names, target_genes))
    # THE FIX: scatter every target column ONCE, instead of embedding each
    # one directly in its own delayed() closure.
    target_columns = [expression_matrix[:, i] for i in indices]
    future_target_columns = client.scatter(target_columns, broadcast=False)

    delayed_link_dfs = []
    delayed_meta_dfs = []
    for pos, target_gene_index in enumerate(indices):
        target_gene_name = delayed(gene_names[target_gene_index], pure=True)
        target_gene_expression = future_target_columns[pos]

        if include_meta:
            delayed_link_df, delayed_meta_df = delayed(infer_partial_network, pure=True, nout=2)(
                regressor_type, regressor_kwargs,
                future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed)
            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)
                delayed_meta_dfs.append(delayed_meta_df)
        else:
            delayed_link_df = delayed(infer_partial_network, pure=True)(
                regressor_type, regressor_kwargs,
                future_tf_matrix, future_tf_matrix_gene_names,
                target_gene_name, target_gene_expression, include_meta, early_stop_window_length, seed)
            if delayed_link_df is not None:
                delayed_link_dfs.append(delayed_link_df)

    all_links_df = from_delayed(delayed_link_dfs, meta=_GRN_SCHEMA)
    all_meta_df = from_delayed(delayed_meta_dfs, meta=_META_SCHEMA)

    if limit:
        maybe_limited_links_df = all_links_df.nlargest(limit, columns=['importance'])
    else:
        maybe_limited_links_df = all_links_df

    n_parts = len(client.ncores()) * repartition_multiplier
    if include_meta:
        return maybe_limited_links_df.repartition(npartitions=n_parts), \
               all_meta_df.repartition(npartitions=n_parts)
    else:
        return maybe_limited_links_df.repartition(npartitions=n_parts)


def _run_genie3_core(expr_df, tfs, seed, n_estimators, client):
    """Runs arboreto's GENIE3 engine directly via _create_graph_scatter_fixed
    (see its own docstring for why this is a drop-in for
    arboreto.core.create_graph, not a rewrite of arboreto.algo's
    genie3()/diy() convenience wrappers).

    Two issues in arboreto 0.1.6 (2019) make those convenience wrappers
    incompatible with modern dask/distributed (this pipeline pins none, to
    stay on maintained releases per the task spec):
      1. `_prepare_client` calls `LocalCluster(diagnostics_port=None)` for
         client_or_address='local'/None, and modern distributed has dropped
         that kwarg -> TypeError. Worked around by always passing our own
         live Client (see run_one_combo), which takes a different branch in
         `_prepare_client` that never constructs a LocalCluster itself.
      2. `create_graph(..., include_meta=False)` (diy()'s default) still
         unconditionally calls `from_delayed(delayed_meta_dfs, ...)` on the
         *empty* meta-dataframe list that only the include_meta=True branch
         populates. Older dask's from_delayed tolerated an empty list;
         current dask raises `TypeError: Must supply at least one delayed
         object`. Worked around by calling create_graph with
         include_meta=True (populating both lists) and discarding the meta
         half we don't need.
    Neither workaround changes GENIE3's actual algorithm/output -- both are
    purely about which internal arboreto code path builds the Dask graph.
    """
    from arboreto.algo import _prepare_input
    from arboreto.core import RF_KWARGS

    regressor_kwargs = dict(RF_KWARGS)
    if n_estimators is not None:
        regressor_kwargs["n_estimators"] = n_estimators

    expression_matrix, gene_names, tf_names = _prepare_input(expr_df, None, tfs)
    links_graph, _meta_graph = _create_graph_scatter_fixed(
        expression_matrix, gene_names, tf_names,
        client=client, regressor_type="RF", regressor_kwargs=regressor_kwargs,
        include_meta=True, limit=None, seed=seed,
    )
    network_df = client.compute(links_graph, sync=True).sort_values(
        by="importance", ascending=False,
    )
    return network_df


MAX_CELLS = 50000
# REVISED again, back to fully-uncapped: the actual root cause (arboreto's
# create_graph embedding each target column directly into the Dask graph
# instead of scattering it, confirmed by reading arboreto/core.py) is now
# fixed at the source via _create_graph_scatter_fixed() above -- capping
# cell count was only ever a workaround for that, not a real fix. The
# scatter-fixed graph construction is VALIDATED (not just hoped to work):
# tested locally against the original arboreto.core.create_graph on
# synthetic data (500 cells/30 genes/8 TFs, live local Dask cluster) --
# identical edge set, identical TF/target pairs, max importance
# difference 0.0. Since the transport bottleneck should now be gone,
# 50000 (comfortably above Tier 1's ~40,500 natural pool) lets all three
# tiers run genuinely uncapped, which is the original goal -- if Tier 1
# is STILL slow after this fix, that means something other than
# graph-embedding size is the bottleneck (worth reporting back fresh
# rather than guessing another cap value).


def run_one_combo(adata, gt_edges, seed, n_estimators=None):
    expr_df = pd.DataFrame(adata.X, columns=adata.var_names)
    tfs = adata.var_names[adata.var["is_TF"]].tolist()
    targets = adata.var_names.tolist()

    # Cap cells per run: RF-based GRN inference (GENIE3) scales with n_cells,
    # and beyond a few thousand cells the extra rows buy little signal for
    # much more compute -- standard practice for this class of method.
    if expr_df.shape[0] > MAX_CELLS:
        expr_df = expr_df.sample(n=MAX_CELLS, random_state=seed)

    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster()
    client = Client(cluster)
    try:
        network_df = _run_genie3_core(expr_df, tfs, seed, n_estimators, client)
    finally:
        client.close()
        cluster.close()

    A_full = build_A_full(network_df, tfs, targets)
    auprc, auroc = compute_auprc_auroc(A_full, tfs, targets, gt_edges)
    return network_df, tfs, targets, auprc, auroc


def random_baseline(tfs, targets, gt_edges):
    n_true_edges = len(gt_edges)
    n_genes = len(targets)
    n_tfs = len(tfs)
    auprc_baseline = n_true_edges / (n_genes * n_tfs)
    auroc_baseline = 0.5
    return auprc_baseline, auroc_baseline


def summarize(raw_df):
    tier_to_label = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
    rows = []
    for tier, n_tp in TIERS.items():
        sub = raw_df[raw_df["tier"] == tier]
        rows.append({
            "tier": tier_to_label[tier],
            "n_timepoints": n_tp,
            "mean_auprc_mean": sub["auprc"].mean(),
            "mean_auprc_std": sub["auprc"].std(),
            "mean_auroc_mean": sub["auroc"].mean(),
            "mean_auroc_std": sub["auroc"].std(),
            "n_seeds": len(sub),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT),
                     help="Where per-(tier,seed) data_tier<N>_seed<S>/ dirs live "
                          f"(default: {DEFAULT_DATA_ROOT})")
    ap.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT),
                     help=f"Output dir for results (default: {DEFAULT_RESULTS_ROOT})")
    ap.add_argument("--n_estimators", type=int, default=None,
                     help="Override arboreto's default n_estimators (e.g. to speed "
                          "up a benchmark run). Left unset uses arboreto's own default.")
    ap.add_argument("--log_file", type=str, default=str(DEFAULT_LOG_FILE),
                     help=f"Log file path (default: {DEFAULT_LOG_FILE})")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log(log_file, f"Starting GENIE3 density sweep: {len(TIERS)} tiers x {N_SEEDS} seeds "
                  f"(9 combos), data_root={data_root}")

    raw_rows = []
    baseline_printed = False

    for tier, n_tp in TIERS.items():
        for seed in range(N_SEEDS):
            combo_name = f"tier{tier}_seed{seed}"
            log(log_file, f"=== {combo_name} (n_timepoints_keep={n_tp}) ===")
            t0 = time.time()
            try:
                adata, gt_edges = load_combo(data_root, tier, seed)
                network_df, tfs, targets, auprc, auroc = run_one_combo(
                    adata, gt_edges, seed, n_estimators=args.n_estimators,
                )

                if not baseline_printed:
                    b_auprc, b_auroc = random_baseline(tfs, targets, gt_edges)
                    msg = (f"Theoretical random baseline: AUPRC={b_auprc:.4f} "
                           f"(n_true_edges/(n_genes*n_tfs)), AUROC={b_auroc:.4f}")
                    print(msg)
                    log(log_file, msg)
                    baseline_printed = True

                edges_path = results_root / f"rankedEdges_tier{tier}_seed{seed}.csv"
                network_df.to_csv(edges_path, index=False)

                raw_rows.append({
                    "tier": tier, "n_timepoints": n_tp, "seed": seed,
                    "auprc": auprc, "auroc": auroc,
                })

                elapsed_min = (time.time() - t0) / 60
                log(log_file, f"  AUPRC={auprc:.4f} AUROC={auroc:.4f}")
                log(log_file, f"  done in {elapsed_min:.1f} min")
            except Exception as e:
                elapsed_min = (time.time() - t0) / 60
                log(log_file, f"  FAILED after {elapsed_min:.1f} min: {type(e).__name__}: {e}")

    log(log_file, "")
    if not raw_rows:
        log(log_file, "No combos completed successfully; skipping summary outputs.")
        return

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(results_root / "genie3_raw_results.csv", index=False)

    summary_df = summarize(raw_df)
    summary_df.to_csv(results_root / "genie3_density_results_summary.csv", index=False)

    n_done = len(raw_df)
    log(log_file, f"Sweep finished. done={n_done} failed={9 - n_done} (of 9 total)")


if __name__ == "__main__":
    main()
