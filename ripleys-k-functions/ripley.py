import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def _cell_type_coords(df, cell_type):
    required = {"cell_class", "coord_X", "coord_Y"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    sub = df[df["cell_class"] == cell_type]
    if sub.empty:
        raise ValueError(f"No rows found for cell_class == {cell_type!r}")

    coords = sub[["coord_X", "coord_Y"]].to_numpy(dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    coords = coords[finite]
    if coords.shape[0] < 2:
        raise ValueError(f"Need at least 2 finite cells for Ripley's K; got {coords.shape[0]}.")

    return coords


def ripley_k_function(df, cell_type, radius):
    """
    Compute Ripley's K for cells of one type over one or more radii.

    The normalization uses the area of the smallest axis-aligned square
    containing all selected cells: K(r) = area * pair_count(r) / n^2.
    """
    coords = _cell_type_coords(df, cell_type)
    radii = np.asarray(radius, dtype=float)
    scalar_radius = radii.ndim == 0
    radii = np.atleast_1d(radii)

    if np.any(radii < 0):
        raise ValueError("radius values must be non-negative.")

    n = coords.shape[0]
    ranges = np.ptp(coords, axis=0)
    side_length = float(np.max(ranges))
    if side_length <= 0:
        raise ValueError("Selected cells must span a positive square area.")

    area = side_length**2
    tree = cKDTree(coords)
    ordered_counts = tree.count_neighbors(tree, radii) - n
    k_values = area * ordered_counts / (n**2)

    return float(k_values[0]) if scalar_radius else k_values


def ripley_L_function(df, cell_type, radius):
    """Compute Besag's L function from Ripley's K."""
    k_values = ripley_k_function(df, cell_type, radius)
    return np.sqrt(np.asarray(k_values) / np.pi)


def ripley_pair_correlation_function(df, cell_type, radius, bandwidth=10e-4):
    """
    Compute the pair correlation curve from the analytic derivative of Ripley's K.

    For finite points, empirical K(r) is a step function:
        K(r) = area / n^2 * sum_{i != j} 1(distance(i, j) <= r)

    Its derivative is therefore a sum of Dirac spikes at the observed pairwise
    distances. To make this usable as a curve, each spike is smoothed with a
    Gaussian kernel of width `bandwidth`.
    """
    radii = np.asarray(radius, dtype=float)
    if radii.ndim == 0 or radii.size < 2:
        raise ValueError("pair correlation requires at least two radius values.")
    if np.any(radii < 0):
        raise ValueError("radius values must be non-negative.")
    if np.any(np.diff(radii) <= 0):
        raise ValueError("radius values must be strictly increasing.")
    if bandwidth is None:
        bandwidth = float(np.median(np.diff(radii)))
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    coords = _cell_type_coords(df, cell_type)
    n = coords.shape[0]
    ranges = np.ptp(coords, axis=0)
    side_length = float(np.max(ranges))
    if side_length <= 0:
        raise ValueError("Selected cells must span a positive square area.")

    area = side_length**2
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(radii[-1] + 4 * bandwidth), output_type="ndarray")
    derivative = np.zeros_like(radii, dtype=float)
    if pairs.size:
        distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        scaled_distances = (radii[:, None] - distances[None, :]) / bandwidth
        gaussian_kernel = np.exp(-0.5 * scaled_distances**2) / (bandwidth * np.sqrt(2 * np.pi))
        derivative = area * 2 * gaussian_kernel.sum(axis=1) / (n**2)

    pair_correlation = np.full_like(derivative, np.nan, dtype=float)
    nonzero = radii != 0
    pair_correlation[nonzero] = derivative[nonzero] / (radii[nonzero] * 2 * np.pi)
    return pair_correlation


def plot_ripley_curves(df, cell_type, r_min, r_max, r_steps, scale=1.0, logspace=False, bandwidth=None):
    """
    Plot Ripley's L and pair correlation curves for a cell type in separate subplots.

    Parameters
    ----------
    df : DataFrame
        Input data.
    cell_type : str
        Cell type.
    r_min, r_max : float
        Range for radius values.
    r_steps : int
        Number of steps.
    scale : float
        Plot scale/size multiplier.
    logspace : bool
        Whether to use log-spaced radii instead of linear.
    bandwidth : float or None
        Smoothing bandwidth for the pair correlation kernel. If None, the
        median spacing between consecutive radii is used.
    """
    if r_steps < 2:
        raise ValueError("r_steps must be at least 2.")
    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min.")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if logspace:
        if r_min <= 0:
            raise ValueError("r_min must be positive for logspace.")
        radii = np.logspace(np.log10(r_min), np.log10(r_max), int(r_steps), dtype=float)
    else:
        radii = np.linspace(r_min, r_max, int(r_steps), dtype=float)

    if bandwidth is None:
        bandwidth = float(np.median(np.diff(radii)))

    l_values = ripley_L_function(df, cell_type, radii)
    pair_correlation = ripley_pair_correlation_function(df, cell_type, radii, bandwidth=bandwidth)

    fig, axs = plt.subplots(2, 1, figsize=(8 * scale, 10 * scale), sharex=True)

    axs[0].plot(radii, l_values, color='b')
    axs[0].set_ylabel("Ripley L")
    axs[0].set_title(f"Ripley's L function for {cell_type}")
    axs[0].grid(True, alpha=0.3)
    if logspace:
        axs[0].set_xscale('log')

    axs[1].plot(radii, pair_correlation, color='g')
    axs[1].set_xlabel("Radius")
    axs[1].set_ylabel("Pair correlation")
    axs[1].set_title(f"Pair correlation function for {cell_type}")
    axs[1].grid(True, alpha=0.3)
    if logspace:
        axs[1].set_xscale('log')

    fig.tight_layout()

    return fig, axs


def plot_ripley_curves_multiple(dfs_and_names, cell_type, r_min, r_max, r_steps, scale=1.0, logspace=False, bandwidth=None):
    """
    Plot Ripley's L and pair correlation curves for multiple dataframes.

    Parameters
    ----------
    dfs_and_names:
        Tuple of (dataframes, names), where both entries are lists of the same length.
    cell_type : str
        Cell type.
    r_min, r_max : float
        Range for radius values.
    r_steps : int
        Number of steps.
    scale : float
        Plot scale/size multiplier.
    logspace : bool
        Whether to use log-spaced radii instead of linear.
    bandwidth : float or None
        Smoothing bandwidth for the pair correlation kernel. If None, the
        median spacing between consecutive radii is used (this rescales
        with the radii grid so the same call works for both small and
        large coordinate ranges).
    """
    if r_steps < 2:
        raise ValueError("r_steps must be at least 2.")
    if r_max <= r_min:
        raise ValueError("r_max must be greater than r_min.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    if logspace:
        if r_min <= 0:
            raise ValueError("r_min must be positive for logspace.")
        radii = np.logspace(np.log10(r_min), np.log10(r_max), int(r_steps), dtype=float)
    else:
        radii = np.linspace(r_min, r_max, int(r_steps), dtype=float)

    if bandwidth is None:
        bandwidth = float(np.median(np.diff(radii)))

    dfs, names = dfs_and_names
    if len(dfs) != len(names):
        raise ValueError("dataframes and names must have the same length.")
    if len(dfs) == 0:
        raise ValueError("At least one dataframe is required.")

    fig, axs = plt.subplots(2, 1, figsize=(8 * scale, 10 * scale), sharex=True)

    for df, name in zip(dfs, names):
        l_values = ripley_L_function(df, cell_type, radii)
        pair_correlation = ripley_pair_correlation_function(df, cell_type, radii, bandwidth=bandwidth)

        axs[0].plot(radii, l_values, label=name)
        axs[1].plot(radii, pair_correlation, label=name)

    axs[0].set_ylabel("Ripley L")
    axs[0].set_title(f"Ripley's L function for {cell_type}")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    if logspace:
        axs[0].set_xscale('log')

    axs[1].set_xlabel("Radius")
    axs[1].set_ylabel("Pair correlation")
    axs[1].set_title(f"Pair correlation function for {cell_type}")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    if logspace:
        axs[1].set_xscale('log')

    fig.tight_layout()

    return fig, axs

import pandas as pd
from typing import Tuple

def align_prediction_rigid_to_ground_truth_by_id(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    *,
    cell_class_value: str,
    id_col: int | str = 0,
    coord_cols: Tuple[str, str] = ("coord_X", "coord_Y"),
    cell_class_col: str = "cell_class",
    allow_reflection: bool = False,
) -> pd.DataFrame:
    """
    Compute the optimal 2D rigid transform (rotation + translation) that aligns predicted
    coordinates to ground-truth coordinates for one cell type, using shared unique IDs.

    The alignment minimizes the sum of squared distances between matched cells:
        min_{R,t} || (P R + t) - T ||_F^2
    where R is 2x2 orthonormal (det=+1 unless allow_reflection=True).

    Parameters
    ----------
    df_true, df_pred:
        DataFrames containing coordinates and cell type annotations.
    cell_class_value:
        Which `cell_class` to filter on before matching.
    id_col:
        Column holding the unique cell ID used to match rows between df_true and df_pred.
        If the CSV was read without an explicit name for the first column, it is commonly `0`.
        You can also pass a string column name.
    coord_cols:
        (x_col, y_col) column names for coordinates.
    cell_class_col:
        Column name holding the cell type (default 'cell_class').
    allow_reflection:
        If False (default), enforce a proper rotation (det(R)=+1).
        If True, allow reflection (det(R) may be -1).

    Returns
    -------
    pd.DataFrame
        Rows for the selected cell type and matched IDs, containing ground-truth columns
        plus *transformed* prediction coordinates as 'coord_X_pred_aligned', 'coord_Y_pred_aligned'.
        Also includes the raw prediction coordinates in 'coord_X_pred', 'coord_Y_pred'.
    """

    coord_X, coord_Y = coord_cols
    for dname, d in [("df_true", df_true), ("df_pred", df_pred)]:
        missing = {cell_class_col, coord_X, coord_Y} - set(d.columns)
        if isinstance(id_col, str):
            missing |= ({id_col} - set(d.columns))
        else:
            # int means positional column index
            if not (0 <= int(id_col) < d.shape[1]):
                raise KeyError(f"{dname}: id_col={id_col} is out of bounds for columns.")
        if missing:
            raise KeyError(f"{dname} missing required columns: {sorted(missing)}")

    true_sub = df_true[df_true[cell_class_col] == cell_class_value].copy()
    pred_sub = df_pred[df_pred[cell_class_col] == cell_class_value].copy()
    if true_sub.empty or pred_sub.empty:
        raise ValueError(
            f"No rows for cell_class == {cell_class_value!r} in "
            f"{'df_true' if true_sub.empty else ''}"
            f"{' and ' if true_sub.empty and pred_sub.empty else ''}"
            f"{'df_pred' if pred_sub.empty else ''}"
        )

    # Attach IDs (positional-safe)
    if isinstance(id_col, str):
        true_sub["_cell_id"] = true_sub[id_col]
        pred_sub["_cell_id"] = pred_sub[id_col]
    else:
        true_sub["_cell_id"] = true_sub.iloc[:, int(id_col)]
        pred_sub["_cell_id"] = pred_sub.iloc[:, int(id_col)]

    # Inner-join on shared IDs
    merged = true_sub.merge(
        pred_sub[["_cell_id", coord_X, coord_Y]],
        on="_cell_id",
        how="inner",
        suffixes=("_true", "_pred"),
    )
    if merged.shape[0] < 3:
        raise ValueError(f"Need at least 3 matched cells to fit rigid transform; got {merged.shape[0]}.")

    T = merged[[f"{coord_X}_true", f"{coord_Y}_true"]].to_numpy(dtype=float)  # target (true)
    P = merged[[f"{coord_X}_pred", f"{coord_Y}_pred"]].to_numpy(dtype=float)  # source (pred)

    finite = np.isfinite(T).all(axis=1) & np.isfinite(P).all(axis=1)
    T = T[finite]
    P = P[finite]
    merged = merged.loc[finite].copy()
    if T.shape[0] < 3:
        raise ValueError("Not enough finite matched points to fit rigid transform.")

    # Kabsch algorithm (2D)
    t_mean = T.mean(axis=0)
    p_mean = P.mean(axis=0)
    Tc = T - t_mean
    Pc = P - p_mean

    H = Pc.T @ Tc
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    if not allow_reflection and np.linalg.det(R) < 0:
        # Fix reflection by flipping last singular vector
        U[:, -1] *= -1
        R = U @ Vt

    t = t_mean - p_mean @ R

    P_aligned = P @ R + t

    out = merged.copy()
    out[f"{coord_X}_pred_aligned"] = P_aligned[:, 0]
    out[f"{coord_Y}_pred_aligned"] = P_aligned[:, 1]

    # Keep the original cell id column name for convenience if possible
    out.rename(columns={"_cell_id": "cell_id"}, inplace=True)
    out["cell_type"] = cell_class_value

    # Convenience columns: make aligned prediction available under coord_X/coord_Y
    # so downstream plotting utilities that expect these names work directly.
    out[coord_X] = out[f"{coord_X}_pred_aligned"]
    out[coord_Y] = out[f"{coord_Y}_pred_aligned"]
    return out
