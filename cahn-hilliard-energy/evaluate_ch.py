"""Evaluate Cahn-Hilliard energy metrics on prediction CSVs against a ground truth.

For each prediction this script prints two scalar values:

- ``avg_ch_energy_auc_diff_per_cell_type``: for every cell type, build the
  single-phase Cahn-Hilliard energy curve E(r) over a range of bump radii
  (default: 8 values linearly spaced in [0.001, 0.01]) for both the
  ground truth and the prediction (on a shared grid), take the area under
  each curve, and compare them. The per-cell-type score is the absolute
  difference of areas. The reported value is the mean over cell types.
- ``avg_pair_voronoi_ch_energy``: the mean over all unordered cell-type
  pairs of the Cahn-Hilliard energy of the smooth Voronoi two-phase
  landscape built from each pair.

Both metrics are computed on the prediction landscapes, rebuilt on a grid
that also covers the ground truth (via ``common_match_key``) so values are
comparable across predictions and against the ground truth.

Example
-------
    python evaluate_ch.py \
        --gt /path/to/metadata_true.csv \
        --pred /path/to/metadata_pred_a.csv /path/to/metadata_pred_b.csv
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from cahn_hilliard import (
    build_continuous_landscape_from_points,
    build_voronoi_phase_landscape_from_cell_types,
    common_match_key,
    sigmoid_bump,
)


def _shared_cell_types(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cell_class_col: str,
    *,
    min_points: int = 1,
) -> List[str]:
    """Return cell types present (with enough points) in both GT and prediction."""

    def counts(df: pd.DataFrame) -> pd.Series:
        return df[cell_class_col].value_counts()

    gt_counts = counts(gt_df)
    pred_counts = counts(pred_df)
    shared = sorted(set(gt_counts.index) & set(pred_counts.index))
    return [
        ct for ct in shared
        if gt_counts.get(ct, 0) >= min_points and pred_counts.get(ct, 0) >= min_points
    ]


def _energy_curve_for_cell_type(
    df: pd.DataFrame,
    cell_class_value: str,
    radii: np.ndarray,
    *,
    cell_class_col: str,
    coord_cols: Tuple[str, str],
    match_key,
    kappa: float,
) -> np.ndarray:
    """Compute the single-phase CH energy curve E(r) for one cell type.

    The landscape is rebuilt for every r in ``radii`` on the same fixed
    ``match_key`` grid so values can be compared point-by-point with another
    curve computed on the same grid.
    """

    energies = np.empty(len(radii), dtype=np.float64)
    for i, r in enumerate(radii):
        r = float(r)
        landscape = build_continuous_landscape_from_points(
            df,
            bump_fn=sigmoid_bump,
            d=4.0 / r,
            shift=128.0 * r,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            cell_class_value=cell_class_value,
            radius=r,
            match_key=match_key,
        )
        energies[i] = landscape.cahn_hilliard_energy(kappa=kappa)
    return energies


def average_per_cell_type_energy_auc_diff(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    cell_class_col: str = "cell_class",
    coord_cols: Tuple[str, str] = ("coord_X", "coord_Y"),
    radii: Optional[np.ndarray] = None,
    grid_radius: float = 0.005,
    kappa: float = 0.5,
) -> float:
    """Mean over cell types of |AUC(E_pred(r)) - AUC(E_gt(r))|.

    For every cell type, this builds the single-phase Cahn-Hilliard energy
    curve E(r) over ``radii`` for both the ground truth and the prediction
    on a grid shared between them. The area under each curve is computed
    with the trapezoidal rule, and the per-cell-type score is the absolute
    difference of the two areas. The returned value is the mean over cell
    types.

    Parameters
    ----------
    radii:
        Radii at which to evaluate E(r). Defaults to 8 values linearly
        spaced in [0.001, 0.01], matching ``ch_energy_tests.ipynb``.
    grid_radius:
        Radius used only to derive the common GT/prediction grid extent
        (the energies themselves are recomputed for every r in ``radii``
        on that fixed grid).
    """

    if radii is None:
        radii = np.linspace(0.001, 0.01, 8)
    radii = np.asarray(radii, dtype=np.float64)
    if radii.size < 2:
        raise ValueError("`radii` must contain at least two values to form a curve.")

    cell_types = _shared_cell_types(gt_df, pred_df, cell_class_col)
    if not cell_types:
        raise ValueError("No common cell types between ground truth and prediction.")

    auc_diffs: List[float] = []
    for ct in cell_types:
        gt_seed = build_continuous_landscape_from_points(
            gt_df,
            bump_fn=sigmoid_bump,
            d=4.0 / grid_radius,
            shift=128.0 * grid_radius,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            cell_class_value=ct,
            radius=grid_radius,
        )
        pred_seed = build_continuous_landscape_from_points(
            pred_df,
            bump_fn=sigmoid_bump,
            d=4.0 / grid_radius,
            shift=128.0 * grid_radius,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            cell_class_value=ct,
            radius=grid_radius,
        )
        key = common_match_key(gt_seed.global_key(), pred_seed.global_key())

        e_gt = _energy_curve_for_cell_type(
            gt_df, ct, radii,
            cell_class_col=cell_class_col, coord_cols=coord_cols,
            match_key=key, kappa=kappa,
        )
        e_pred = _energy_curve_for_cell_type(
            pred_df, ct, radii,
            cell_class_col=cell_class_col, coord_cols=coord_cols,
            match_key=key, kappa=kappa,
        )

        auc_gt = float(np.trapz(e_gt, radii))
        auc_pred = float(np.trapz(e_pred, radii))
        auc_diffs.append(abs(auc_pred - auc_gt))

    return float(np.mean(auc_diffs))


def average_pair_voronoi_energy(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    cell_class_col: str = "cell_class",
    coord_cols: Tuple[str, str] = ("coord_X", "coord_Y"),
    transition_width: float = 0.01,
    kappa: float = 0.5,
    normalize: bool = False,
) -> float:
    """Mean over unordered cell-type pairs of the Voronoi two-phase CH energy.

    The prediction Voronoi landscape is rebuilt on a grid shared with the
    ground truth so values from different predictions are directly
    comparable. When ``normalize`` is True, each pair energy is divided by
    ``log10(max(n_a, 10)) * log10(max(n_b, 10))`` (cell counts in the
    prediction) before averaging, mirroring the normalization used in the
    notebook.
    """

    cell_types = _shared_cell_types(gt_df, pred_df, cell_class_col)
    if len(cell_types) < 2:
        raise ValueError("Need at least two common cell types to form pairs.")

    energies: List[float] = []
    for a, b in combinations(cell_types, 2):
        gt_landscape = build_voronoi_phase_landscape_from_cell_types(
            df=gt_df,
            negative_cell_class_value=a,
            positive_cell_class_value=b,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            transition_width=transition_width,
        )
        pred_landscape = build_voronoi_phase_landscape_from_cell_types(
            df=pred_df,
            negative_cell_class_value=a,
            positive_cell_class_value=b,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            transition_width=transition_width,
        )
        key = common_match_key(gt_landscape.global_key(), pred_landscape.global_key())
        pred_on_common = build_voronoi_phase_landscape_from_cell_types(
            df=pred_df,
            negative_cell_class_value=a,
            positive_cell_class_value=b,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            transition_width=transition_width,
            match_key=key,
        )
        e = pred_on_common.cahn_hilliard_energy(kappa=kappa)
        if normalize:
            n_a = int((pred_df[cell_class_col] == a).sum())
            n_b = int((pred_df[cell_class_col] == b).sum())
            norm = np.log10(max(n_a, 10)) * np.log10(max(n_b, 10))
            e = e / norm
        energies.append(e)

    return float(np.mean(energies))


def evaluate(
    gt_path: Path,
    pred_paths: Iterable[Path],
    *,
    cell_class_col: str = "cell_class",
    coord_cols: Tuple[str, str] = ("coord_X", "coord_Y"),
    radii: Optional[np.ndarray] = None,
    grid_radius: float = 0.005,
    transition_width: float = 0.01,
    kappa: float = 1.0,
    normalize_pairs: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Compute the two CH metrics for every prediction file.

    Returns a mapping ``{prediction_path: {"avg_ch_energy_auc_diff_per_cell_type": ...,
    "avg_pair_voronoi_ch_energy": ...}}``.
    """

    gt_df = pd.read_csv(gt_path)

    results: Dict[str, Dict[str, float]] = {}
    for pred_path in pred_paths:
        pred_df = pd.read_csv(pred_path)

        avg_per_type_auc = average_per_cell_type_energy_auc_diff(
            gt_df,
            pred_df,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            radii=radii,
            grid_radius=grid_radius,
            kappa=kappa,
        )
        avg_pair = average_pair_voronoi_energy(
            gt_df,
            pred_df,
            cell_class_col=cell_class_col,
            coord_cols=coord_cols,
            transition_width=transition_width,
            kappa=kappa,
            normalize=normalize_pairs,
        )

        results[str(pred_path)] = {
            "avg_ch_energy_auc_diff_per_cell_type": avg_per_type_auc,
            "avg_pair_voronoi_ch_energy": avg_pair,
        }

        if verbose:
            print(f"{pred_path}")
            print(f"  avg CH energy AUC diff per cell type = {avg_per_type_auc:.6f}")
            print(f"  avg pair voronoi CH energy           = {avg_pair:.6f}")

    return results


'''def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", required=True, type=Path, help="Path to ground truth CSV.")
    parser.add_argument(
        "--pred",
        required=True,
        nargs="+",
        type=Path,
        help="One or more prediction CSV paths.",
    )
    parser.add_argument("--cell-class-col", default="cell_class")
    parser.add_argument("--x-col", default="coord_X")
    parser.add_argument("--y-col", default="coord_Y")
    parser.add_argument(
        "--radius-min",
        type=float,
        default=0.001,
        help="Minimum bump radius in the E(r) sweep for the per-cell-type metric.",
    )
    parser.add_argument(
        "--radius-max",
        type=float,
        default=0.01,
        help="Maximum bump radius in the E(r) sweep for the per-cell-type metric.",
    )
    parser.add_argument(
        "--n-radii",
        type=int,
        default=8,
        help="Number of radii to evaluate for the per-cell-type AUC metric.",
    )
    parser.add_argument(
        "--grid-radius",
        type=float,
        default=0.005,
        help="Bump radius used to derive the common GT/prediction grid extent.",
    )
    parser.add_argument(
        "--transition-width",
        type=float,
        default=0.01,
        help="tanh transition width for the Voronoi two-phase landscape.",
    )
    parser.add_argument("--kappa", type=float, default=1.0, help="Gradient-penalty coefficient.")
    parser.add_argument(
        "--normalize-pairs",
        action="store_true",
        help="Normalize pair energies by log10(n_a)*log10(n_b) before averaging.",
    )
    return parser.parse_args()'''


def main() -> None:
    # args = _parse_args()
    radii = np.linspace(0.001, 0.01, 8)
    evaluate(
        gt_path="/home/asalmona/Documents/Ricci/code/__some_results/outputs/2026-04-12/model_11-15-30-MERFISH_epoch_2999/mouse2_slice300_0/metadata_true.csv",
        pred_paths=["/home/asalmona/Documents/Ricci/code/__some_results/outputs/2026-04-12/model_11-15-30-MERFISH_epoch_2999/mouse2_slice300_0/metadata_pred.csv",
                       "/home/asalmona/Documents/Ricci/code/__some_results/exp3/model_23-10-26-MERFISH_epoch_1999/mouse2_slice300_0/metadata_pred.csv",
                        "/home/asalmona/Documents/Ricci/code/__some_results/exp3/model_23-10-26-MERFISH_epoch_2249/mouse2_slice300_0/metadata_pred.csv",
                        "/home/asalmona/Documents/Ricci/code/__some_results/exp3/model_23-10-26-MERFISH_epoch_2499/mouse2_slice300_0/metadata_pred.csv", 
                        "/home/asalmona/Documents/Ricci/code/__some_results/exp3/model_23-10-26-MERFISH_epoch_2749/mouse2_slice300_0/metadata_pred.csv",
                        "/home/asalmona/Documents/Ricci/code/__some_results/exp3/model_23-10-26-MERFISH_epoch_2999/mouse2_slice300_0/metadata_pred.csv"],
        cell_class_col="cell_class",
        coord_cols=("coord_X", "coord_Y"),
        radii=radii,
        grid_radius=0.005,
        transition_width=0.01,
        kappa=0.5,
        normalize_pairs=False,
    )


if __name__ == "__main__":
    main()
