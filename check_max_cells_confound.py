"""
Verifies whether GENIE3's MAX_CELLS=2000 cap (run_genie3_sergio.py) is
smaller than the natural per-tier cell pool, the same confound class
already found and fixed in PseudoGRN/MTGRN. Reads n_obs DIRECTLY from
each tier's actual SERGIO-Marlene.h5ad -- no timing/log inference, a
real number per tier.

USAGE (Kaggle, wherever marlene/Marlene/data_tier<N>_seed<S>/ actually
lives -- same data_root run_genie3_sergio.py itself defaults to):
    python check_max_cells_confound.py --data_root /path/to/marlene/Marlene
"""
import argparse
from pathlib import Path

import anndata

MAX_CELLS = 2000  # must match run_genie3_sergio.py's own constant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, default=".",
                     help="Directory containing data_tier<N>_seed<S>/SERGIO-Marlene.h5ad "
                          "(same default run_genie3_sergio.py uses).")
    ap.add_argument("--seed", type=int, default=0,
                     help="Only need one seed per tier -- natural pool size "
                          "doesn't depend on seed, only on n_timepoints_keep.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    print(f"MAX_CELLS = {MAX_CELLS}\n")
    print(f"{'tier':<6}{'n_obs (natural pool)':>22}{'> MAX_CELLS?':>14}")

    results = {}
    for tier in [1, 2, 3]:
        h5ad_path = data_root / f"data_tier{tier}_seed{args.seed}" / "SERGIO-Marlene.h5ad"
        if not h5ad_path.exists():
            print(f"{tier:<6}{'MISSING: ' + str(h5ad_path):>22}")
            continue
        adata = anndata.read_h5ad(h5ad_path)
        n_obs = adata.n_obs
        results[tier] = n_obs
        flag = "YES -- WOULD BE CAPPED" if n_obs > MAX_CELLS else "no (already below cap)"
        print(f"{tier:<6}{n_obs:>22}{flag:>14}")

    if len(results) == 3:
        print()
        capped_tiers = [t for t, n in results.items() if n > MAX_CELLS]
        if len(capped_tiers) == 3:
            print("CONFOUNDED: all 3 tiers exceed MAX_CELLS -- every tier would be "
                  "capped down to the SAME 2000 cells, same bug class as "
                  "PseudoGRN's original 3000 cap and MTGRN's original 1500 cap.")
        elif len(capped_tiers) == 0:
            print("NOT CONFOUNDED: no tier exceeds MAX_CELLS -- the cap never "
                  "actually triggers, all tiers already ran on their natural sizes.")
        else:
            print(f"PARTIALLY CONFOUNDED: tiers {capped_tiers} exceed MAX_CELLS and "
                  f"would be capped to the same 2000; tiers "
                  f"{[t for t in results if t not in capped_tiers]} run uncapped at "
                  f"their natural size -- still breaks apples-to-apples comparison "
                  f"for whichever tiers ARE capped.")


if __name__ == "__main__":
    main()
