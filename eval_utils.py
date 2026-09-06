"""Dependency-light eval helpers for the GENIE3 positive-control baseline.

Only numpy, pandas, and scikit-learn are used here — nothing from the
`marlene` package, torch, or train_sergio.py. This keeps genie3/ fully
decoupled from the Marlene pipeline's environment, per the proposal's
isolated-environments requirement.

compute_auprc_auroc is copied verbatim from the Marlene eval code so that
this baseline is judged by identical scoring logic.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auprc_auroc(A_full, tfs, targets, gt_edges):
    tf_idx = {g: i for i, g in enumerate(tfs)}
    y_true, y_score = [], []
    for ti, target in enumerate(targets):
        for tf, tf_i in tf_idx.items():
            y_true.append(1 if (tf, target) in gt_edges else 0)
            y_score.append(A_full[ti, tf_i])
    y_true, y_score = np.array(y_true), np.array(y_score)
    if y_true.sum() == 0:
        return float("nan"), float("nan")
    return average_precision_score(y_true, y_score), roc_auc_score(y_true, y_score)


def build_A_full(network_df, tfs, targets):
    tf_idx = {g: i for i, g in enumerate(tfs)}
    target_idx = {g: i for i, g in enumerate(targets)}
    A = np.zeros((len(targets), len(tfs)), dtype=float)
    for row in network_df.itertuples(index=False):
        if row.target in target_idx and row.TF in tf_idx:
            A[target_idx[row.target], tf_idx[row.TF]] = row.importance
    return A
