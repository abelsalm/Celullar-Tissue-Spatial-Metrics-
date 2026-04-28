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


def plot_ripley_curves(df, cell_type, r_min, r_max, r_steps, scale=1.0, logspace=False):
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

    l_values = ripley_L_function(df, cell_type, radii)
    pair_correlation = ripley_pair_correlation_function(df, cell_type, radii)

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


def plot_ripley_curves_multiple(dfs_and_names, cell_type, r_min, r_max, r_steps, scale=1.0, logspace=False):
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

    dfs, names = dfs_and_names
    if len(dfs) != len(names):
        raise ValueError("dataframes and names must have the same length.")
    if len(dfs) == 0:
        raise ValueError("At least one dataframe is required.")

    fig, axs = plt.subplots(2, 1, figsize=(8 * scale, 10 * scale), sharex=True)

    for df, name in zip(dfs, names):
        l_values = ripley_L_function(df, cell_type, radii)
        pair_correlation = ripley_pair_correlation_function(df, cell_type, radii)

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

