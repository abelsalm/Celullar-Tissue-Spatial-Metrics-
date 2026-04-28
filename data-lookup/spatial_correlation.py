from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import anndata as ad
from scipy import sparse

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



DistanceMetric = Literal["abs", "sq"]


@dataclass(frozen=True)
class SpatialGeneGridResult:
    """
    Result of binning one gene's expression into a spatial grid.

    Attributes
    ----------
    distance_grid:
        2D array (ny, nx) with the per-bin distance to the global mean.
        Empty bins (no cells) are set to `-variance` so they can be colored distinctly.
    variance:
        Variance of the per-bin distances over non-empty bins only.
    global_mean:
        Global mean expression used as the reference: the mean of the per-bin local means
        over non-empty bins (each non-empty bin contributes equally).
    local_mean_grid:
        2D array (ny, nx) with the per-bin mean expression (0 for empty bins).
    cell_count_grid:
        2D array (ny, nx) with number of cells in each bin.
    origin:
        (x0, y0) used for binning: bin (0,0) spans [x0, x0+size) × [y0, y0+size).
    spatial_size:
        Bin side length in the same units as `adata.obsm["spatial"]`.
    """

    distance_grid: np.ndarray
    variance: float
    global_mean: float
    local_mean_grid: np.ndarray
    cell_count_grid: np.ndarray
    origin: Tuple[float, float]
    spatial_size: float

    def plot_distance(
        self,
        *,
        ax=None,
        figsize: Tuple[float, float] = (17.35, 10.2),
        cmap: str = "Reds",
        interpolation: str = "none",
        invert_x: bool = True,
        axis_off: bool = True,
        colorbar: bool = True,
        colorbar_label: str = "Distance to mean",
        title: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        aspect: str = "auto",
        negative_color: str = "#C4C4C4",
    ):
        """
        Plot `distance_grid` with matplotlib in the same style as the density plot
        used in `spatial_correlations.ipynb` (imshow + extent + origin='lower' +
        inverted x-axis + axes hidden).

        Parameters are intentionally close to `imshow`/`plt.subplots`.
        Returns (fig, ax, im).

        New behavior: Negative values are all shown in the same color (customizable by `negative_color`), 
        while positive values are mapped via `cmap`.
        """

        x0, y0 = self.origin
        ny, nx = self.distance_grid.shape
        x1 = x0 + nx * self.spatial_size
        y1 = y0 + ny * self.spatial_size

        data = self.distance_grid

        # Find suitable vmin/vmax ignoring negative bins
        if vmin is None:
            vmin_data = np.nanmin(data[data > 0]) if np.any(data > 0) else 0.0
            vmin = vmin_data
        if vmax is None:
            vmax_data = np.nanmax(data[data > 0]) if np.any(data > 0) else 1.0
            vmax = vmax_data

        # Create custom colormap: first color is for negative data, then the rest from the cmap
        # Map negative to color index 0, positive to 1..N
        base_cmap = plt.get_cmap(cmap)
        positive_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # We'll build a ListedColormap: first entry is for negative, rest comes from base cmap.
        n_cmap_bins = 256
        positive_colors = base_cmap(np.linspace(0, 1, n_cmap_bins - 1))
        custom_colors = np.vstack([
            mcolors.to_rgba(negative_color),  # All negative cells map here
            positive_colors
        ])
        custom_cmap = mcolors.ListedColormap(custom_colors)

        # Map data to indices for the custom colormap
        # All negatives map to 0, all non-negatives linearly between 1 and n_cmap_bins-1
        color_index_grid = np.zeros_like(data, dtype=float)
        positive_mask = data >= 0
        if np.any(positive_mask):
            # scale positive values into [1, n_cmap_bins-1]
            scaled = 1 + (n_cmap_bins - 2) * positive_norm(data[positive_mask])
            color_index_grid[positive_mask] = scaled
        color_index_grid[~positive_mask] = 0  # Negative always 0

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        im = ax.imshow(
            color_index_grid,
            cmap=custom_cmap,
            interpolation=interpolation,
            origin="lower",
            extent=[x0, x1, y0, y1],
            aspect=aspect,
            vmin=0,
            vmax=n_cmap_bins - 1
        )

        if invert_x:
            ax.set_xlim(x1, x0)

        if axis_off:
            ax.axis("off")

        if title is not None:
            ax.set_title(title)

        if colorbar:
            # Create a colorbar with correct ticks/labels: first tick for negatives, rest for positives
            cbar = plt.colorbar(im, ax=ax, ticks=[0, n_cmap_bins-1])
            cbar.set_ticklabels(
                [f"< 0", f"{vmax:.3g}"]
            )
            cbar.ax.set_ylabel(colorbar_label)
            # Also add in-between ticks for continuous legend if desired
            # For advanced custom ticks provide more detail here.

        plt.tight_layout()
        return fig, ax, im


def spatial_gene_distance_grid(
    adata: ad.AnnData,
    gene: str,
    spatial_size: float,
    *,
    use_raw: bool = False,
    layer: Optional[str] = None,
    metric: DistanceMetric = "abs",
) -> SpatialGeneGridResult:
    """
    Bin cells into a square grid and quantify how locally the gene deviates from the global mean.

    Parameters
    ----------
    adata:
        AnnData with coordinates in `adata.obsm["spatial"]` of shape (n_cells, 2).
    gene:
        Gene name; must be present in `adata.var_names` (or `adata.raw.var_names` if use_raw=True).
    spatial_size:
        Side length of each square bin, in the same units as the spatial coordinates.
    use_raw:
        If True, use `adata.raw[:, gene].X` (ignores `layer`).
    layer:
        If provided, use `adata.layers[layer]` instead of `adata.X` (ignored if use_raw=True).
    metric:
        "abs" for absolute distance |local_mean - global_mean|,
        "sq"  for squared distance (local_mean - global_mean)^2.

    Returns
    -------
    SpatialGeneGridResult
        Includes the distance grid and its variance, plus intermediate grids.
    """

    if spatial_size <= 0:
        raise ValueError("spatial_size must be > 0")

    if "spatial" not in adata.obsm:
        raise KeyError('Expected `adata.obsm["spatial"]` to exist.')

    spatial = np.asarray(adata.obsm["spatial"])
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError('`adata.obsm["spatial"]` must have shape (n_cells, 2).')

    x = spatial[:, 0].astype(float, copy=False)
    y = spatial[:, 1].astype(float, copy=False)

    # Define grid extents
    x0 = float(np.nanmin(x))
    y0 = float(np.nanmin(y))
    x1 = float(np.nanmax(x))
    y1 = float(np.nanmax(y))

    nx = int(np.ceil((x1 - x0) / spatial_size)) + 1
    ny = int(np.ceil((y1 - y0) / spatial_size)) + 1
    if nx <= 0 or ny <= 0:
        raise ValueError("Invalid grid size computed; check spatial coordinates and spatial_size.")

    # Bin assignment (clip to ensure right-edge max values stay in-grid)
    ix = np.floor((x - x0) / spatial_size).astype(int)
    iy = np.floor((y - y0) / spatial_size).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # Extract gene expression vector (n_cells,)
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but `adata.raw` is None.")
        if gene not in adata.raw.var_names:
            raise KeyError(f"Gene {gene!r} not found in `adata.raw.var_names`.")
        g = adata.raw[:, gene].X
    else:
        if gene not in adata.var_names:
            raise KeyError(f"Gene {gene!r} not found in `adata.var_names`.")
        if layer is None:
            g = adata[:, gene].X
        else:
            if layer not in adata.layers:
                raise KeyError(f"Layer {layer!r} not found in `adata.layers`.")
            # layers are (n_cells, n_genes); slice for the gene
            gene_idx = int(np.where(adata.var_names == gene)[0][0])
            g = adata.layers[layer][:, gene_idx]

    if sparse.issparse(g):
        g = g.toarray()
    g = np.asarray(g).reshape(-1)
    if g.shape[0] != adata.n_obs:
        raise ValueError("Gene expression vector length does not match number of cells.")

    # Accumulate counts and expression sums per bin
    cell_count_grid = np.zeros((ny, nx), dtype=np.int64)
    expr_sum_grid = np.zeros((ny, nx), dtype=np.float64)
    np.add.at(cell_count_grid, (iy, ix), 1)
    np.add.at(expr_sum_grid, (iy, ix), g.astype(np.float64, copy=False))

    # Local means: only defined for bins with cells (empty bins kept at 0 for convenience)
    local_mean_grid = np.zeros((ny, nx), dtype=np.float64)
    nonempty = cell_count_grid > 0
    local_mean_grid[nonempty] = expr_sum_grid[nonempty] / cell_count_grid[nonempty]

    # Global mean:
    # Mean of the local means over *non-empty* bins (each non-empty bin counts once).
    global_mean = float(local_mean_grid[nonempty].mean()) if np.any(nonempty) else 0.0

    # Distance to mean per bin: compute only for non-empty bins
    diff = local_mean_grid - global_mean
    distance_grid = np.zeros((ny, nx), dtype=np.float64)
    if metric == "abs":
        distance_grid[nonempty] = np.abs(diff[nonempty])
    elif metric == "sq":
        distance_grid[nonempty] = diff[nonempty] ** 2
    else:
        raise ValueError(f"Unknown metric {metric!r}. Use 'abs' or 'sq'.")

    # Variance over non-empty bins only (empty bins are not real measurements)
    variance = float(np.var(distance_grid[nonempty])) if np.any(nonempty) else 0.0

    # Mark empty bins so they show up in a different color when plotting
    if np.any(~nonempty):
        distance_grid[~nonempty] = -variance

    return SpatialGeneGridResult(
        distance_grid=distance_grid,
        variance=variance,
        global_mean=global_mean,
        local_mean_grid=local_mean_grid,
        cell_count_grid=cell_count_grid,
        origin=(x0, y0),
        spatial_size=float(spatial_size),
    )

