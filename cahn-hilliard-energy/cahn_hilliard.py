from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd


class LandscapeKey(TypedDict):
    top_left: Tuple[float, float]
    bottom_right: Tuple[float, float]
    dx: float
    dy: float


def common_match_key(
    key_a: LandscapeKey,
    key_b: LandscapeKey,
    *,
    spacing: Literal["finer", "coarser", "assert_equal"] = "finer",
) -> LandscapeKey:
    """
    Build a common match-key (global extent + spacing) from two landscape keys.

    The returned key covers the UNION of both extents, so that two landscapes can be
    rebuilt on the same grid and visualized/compared in the same coordinate frame.

    Parameters
    ----------
    key_a, key_b:
        Keys obtained from `ContinuousLandscape2D.global_key()`.
    spacing:
        How to choose the common dx/dy:
        - 'finer' (default): use min(dx), min(dy) to preserve resolution
        - 'coarser': use max(dx), max(dy) for a smaller grid
        - 'assert_equal': require dx/dy to match exactly, otherwise raise
    """

    ax_min = float(key_a["top_left"][0])
    ay_max = float(key_a["top_left"][1])
    ax_max = float(key_a["bottom_right"][0])
    ay_min = float(key_a["bottom_right"][1])

    bx_min = float(key_b["top_left"][0])
    by_max = float(key_b["top_left"][1])
    bx_max = float(key_b["bottom_right"][0])
    by_min = float(key_b["bottom_right"][1])

    x_min = min(ax_min, bx_min)
    x_max = max(ax_max, bx_max)
    y_min = min(ay_min, by_min)
    y_max = max(ay_max, by_max)

    if not (x_max > x_min and y_max > y_min):
        raise ValueError("Invalid extents when combining keys.")

    dx_a, dy_a = float(key_a["dx"]), float(key_a["dy"])
    dx_b, dy_b = float(key_b["dx"]), float(key_b["dy"])
    if dx_a <= 0 or dy_a <= 0 or dx_b <= 0 or dy_b <= 0:
        raise ValueError("Both keys must have dx/dy > 0")

    if spacing == "assert_equal":
        if not (np.isclose(dx_a, dx_b) and np.isclose(dy_a, dy_b)):
            raise ValueError(f"dx/dy differ: a=({dx_a},{dy_a}) b=({dx_b},{dy_b})")
        dx, dy = dx_a, dy_a
    elif spacing == "finer":
        dx, dy = min(dx_a, dx_b), min(dy_a, dy_b)
    elif spacing == "coarser":
        dx, dy = max(dx_a, dx_b), max(dy_a, dy_b)
    else:
        raise ValueError("spacing must be one of: 'finer', 'coarser', 'assert_equal'")

    return {
        "top_left": (x_min, y_max),
        "bottom_right": (x_max, y_min),
        "dx": float(dx),
        "dy": float(dy),
    }


@dataclass(frozen=True)
class ContinuousLandscape2D:
    """
    A continuous(-ish) 2D landscape built from discrete point locations.

    Attributes
    ----------
    field:
        2D array of shape (ny, nx) with values clipped to [-1, 1].
        Starts at -1 everywhere, then "bumps" are added around each point.
    x, y:
        1D coordinate arrays for grid cell centers (length nx and ny).
    extent:
        (x_min, x_max, y_min, y_max) matching matplotlib imshow/extent convention.
    dx, dy:
        Grid spacing in x and y.
    """

    field: np.ndarray
    x: np.ndarray
    y: np.ndarray
    extent: Tuple[float, float, float, float]
    dx: float
    dy: float

    def global_key(self) -> LandscapeKey:
        """
        Return a portable key describing the landscape's global coordinate frame
        and spacing, so another landscape can be built on the same grid.

        Key fields:
        - top_left: (x_min, y_max)
        - bottom_right: (x_max, y_min)
        - dx, dy: grid spacing
        """

        x_min, x_max, y_min, y_max = self.extent
        return {
            "top_left": (float(x_min), float(y_max)),
            "bottom_right": (float(x_max), float(y_min)),
            "dx": float(self.dx),
            "dy": float(self.dy),
        }

    def cahn_hilliard_energy_density(self, *, kappa: float = 1.0) -> np.ndarray:
        """
        Compute the Cahn–Hilliard energy density on the grid:

            e(x,y) = f(c) + kappa * |∇c|^2

        with potential:
            f(c) = (c^2 - 1)^2

        and c = self.field.

        Parameters
        ----------
        kappa:
            Gradient-penalty coefficient (controls smoothness cost).

        Returns
        -------
        np.ndarray
            2D array (ny, nx) of energy density values.
        """

        kappa = float(kappa)
        if kappa < 0:
            raise ValueError("kappa must be >= 0")

        c = np.asarray(self.field, dtype=np.float64)
        # np.gradient returns [d/dy, d/dx] for a 2D array
        dc_dy, dc_dx = np.gradient(c, self.dy, self.dx, edge_order=1)
        grad_sq = dc_dx * dc_dx + dc_dy * dc_dy
        f = (c * c - 1.0) ** 2
        return f + kappa * grad_sq

    def cahn_hilliard_energy(self, *, kappa: float = 1.0) -> float:
        """
        Compute the integrated Cahn–Hilliard energy over the whole landscape:

            E = ∬ ( f(c) + kappa * |∇c|^2 ) dx dy

        Approximated by a Riemann sum over the grid.
        """

        e = self.cahn_hilliard_energy_density(kappa=kappa)
        return float(np.sum(e) * self.dx * self.dy)


# bump_fn signature:
#   r: distance (same units as coords)
#   radius: smoothing radius / scale
# returns: bump value in [0, 1], where 1 at r=0 and 0 far away.
BumpFn = Callable[[np.ndarray, float], np.ndarray]


def bump_gaussian(r: np.ndarray, radius: float) -> np.ndarray:
    """
    Default smooth bump: Gaussian with scale=radius.

    Returns values in (0, 1] with bump(0)=1.
    """
    radius = float(radius)
    if radius <= 0:
        raise ValueError("radius must be > 0")
    return np.exp(-(r**2) / (2.0 * radius**2))

def sigmoid_bump(r: np.ndarray, cell_radius: float, decay_rate: float, shift: float = 0.0) -> np.ndarray:
    """
    Sigmoid bump function:

    Returns values in (0, 1] with bump(0)=1.
    """
    cell_radius = float(cell_radius)
    decay_rate = float(decay_rate)
    return 1.0 / (1.0 + np.exp(decay_rate * (r - cell_radius) - shift))


def _nearest_distances_to_points(
    x: np.ndarray,
    y: np.ndarray,
    pts: np.ndarray,
    *,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """
    Compute distance from every grid location to the closest point.

    Uses scipy's cKDTree when available, with a chunked NumPy fallback so this
    module still works in lightweight environments.
    """

    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        raise ValueError("pts must have shape (n_points, 2) with n_points > 0")

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    nx = int(x.shape[0])
    ny = int(y.shape[0])
    total = nx * ny
    distances = np.empty(total, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError:
        cKDTree = None

    if cKDTree is not None:
        tree = cKDTree(pts)
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            flat_idx = np.arange(start, stop)
            iy = flat_idx // nx
            ix = flat_idx - iy * nx
            query_pts = np.column_stack((x[ix], y[iy]))
            distances[start:stop] = tree.query(query_pts, k=1)[0]
        return distances.reshape(ny, nx)

    # Keep the fallback memory bounded when there are many points.
    max_pairwise_entries = 5_000_000
    fallback_chunk_size = min(chunk_size, max(1, max_pairwise_entries // pts.shape[0]))
    for start in range(0, total, fallback_chunk_size):
        stop = min(start + fallback_chunk_size, total)
        flat_idx = np.arange(start, stop)
        iy = flat_idx // nx
        ix = flat_idx - iy * nx
        dxs = x[ix, None] - pts[None, :, 0]
        dys = y[iy, None] - pts[None, :, 1]
        distances[start:stop] = np.sqrt(np.min(dxs * dxs + dys * dys, axis=1))

    return distances.reshape(ny, nx)


def build_continuous_landscape_from_points(
    df: pd.DataFrame,
    *,
    cell_class_value: str,
    cell_class_col: str = "cell_class",
    coord_cols: Tuple[str, str] = ("coord_X", "coord_Y"),
    # grid definition
    grid_spacing: Optional[float] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    pad: float = 0.0,
    w_range: Optional[Tuple[float, float, float, float]] = None,
    match_key: Optional[LandscapeKey] = None,
    d: float = 10.0,
    shift: float = 0.0,
    # bump definition
    radius: float = 1.0,
    bump_fn: BumpFn = bump_gaussian,
    # performance controls
    support: Optional[float] = None,
    clip: Tuple[float, float] = (-1.0, 1.0),
) -> ContinuousLandscape2D:
    """
    Create a continuous 2D scalar field from discrete points of one cell type.

    Construction:
    - Start with field = -1 everywhere.
    - For each point, add a localized bump of amplitude 2*bump_fn(distance, radius).
      This makes the center value increase by 2, so -1 + 2 = 1 at the point.
    - Overlapping bumps add, then the field is clipped to max=1 (and min=-1).

    Parameters
    ----------
    df:
        DataFrame containing coordinates and a cell type column.
    cell_class_value:
        Which cell type to select.
    grid_spacing:
        If provided, sets dx=dy=grid_spacing and grid size is derived from data bounds.
    grid_shape:
        If provided, explicitly sets (ny, nx). Exactly one of grid_spacing or grid_shape
        should be provided. If neither is provided, defaults to grid_shape=(256, 256).
    pad:
        Extra padding added around min/max bounds (in coordinate units).
    w_range:
        Optional explicit world-range for the grid as (x_min, x_max, y_min, y_max).
        If provided, these bounds are used instead of the tight bounds derived from
        the points (and `pad` is ignored).
    match_key:
        If provided, forces the new landscape to use the same global extent and spacing
        as another landscape. This overrides `w_range`, `pad`, `grid_shape`, and `grid_spacing`.
    radius:
        Radius/scale passed to bump_fn.
    bump_fn:
        Function controlling the smooth transition from center (1) to background (-1).
        Must return values in [0, 1] with bump_fn(0)=1 for the "point value = 1" rule.
    support:
        Optional cutoff distance: only grid cells with r <= support contribute for each
        point. If None, uses support = 3*radius (good default for Gaussian).
    clip:
        (min, max) clipping bounds for the final field. Should remain (-1, 1) for your use.

    Returns
    -------
    ContinuousLandscape2D
        The field and grid metadata (coordinates, spacing, extent).
    """

    xcol, ycol = coord_cols
    required = {cell_class_col, xcol, ycol}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    sub = df[df[cell_class_col] == cell_class_value].copy()
    if sub.empty:
        raise ValueError(f"No rows found for {cell_class_col} == {cell_class_value!r}")

    pts = sub[[xcol, ycol]].to_numpy(dtype=float)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        raise ValueError("No finite coordinates after filtering.")

    if match_key is not None:
        # Force global extent + spacing to match another landscape
        x_min = float(match_key["top_left"][0])
        y_max = float(match_key["top_left"][1])
        x_max = float(match_key["bottom_right"][0])
        y_min = float(match_key["bottom_right"][1])
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("match_key corners must satisfy x_max > x_min and y_max > y_min")
        dx = float(match_key["dx"])
        dy = float(match_key["dy"])
        if dx <= 0 or dy <= 0:
            raise ValueError("match_key dx/dy must be > 0")
    elif w_range is None:
        x_min = float(np.min(pts[:, 0]) - pad)
        x_max = float(np.max(pts[:, 0]) + pad)
        y_min = float(np.min(pts[:, 1]) - pad)
        y_max = float(np.max(pts[:, 1]) + pad)

        # Make the default domain a *square* bounding box so x/y are on the same scale.
        # This avoids "deforming" coordinates when one axis span is smaller.
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        side = max(x_max - x_min, y_max - y_min)
        half = 0.5 * side
        x_min, x_max = cx - half, cx + half
        y_min, y_max = cy - half, cy + half
    else:
        x_min, x_max, y_min, y_max = (float(w_range[0]), float(w_range[1]), float(w_range[2]), float(w_range[3]))
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("w_range must satisfy x_max > x_min and y_max > y_min")

    if grid_spacing is not None and grid_shape is not None:
        raise ValueError("Provide exactly one of grid_spacing or grid_shape (not both).")

    if match_key is not None:
        # Derive grid shape from extent and fixed spacing
        nx = int(np.round((x_max - x_min) / dx)) + 1
        ny = int(np.round((y_max - y_min) / dy)) + 1
        if nx <= 1 or ny <= 1:
            raise ValueError("match_key leads to invalid grid shape; check extent/dx/dy")
        x = np.linspace(x_min, x_max, nx, dtype=float)
        y = np.linspace(y_min, y_max, ny, dtype=float)
    else:
        if grid_spacing is None and grid_shape is None:
            grid_shape = (512, 512)

        if grid_spacing is not None:
            dx = dy = float(grid_spacing)
            if dx <= 0:
                raise ValueError("grid_spacing must be > 0")
            nx = int(np.ceil((x_max - x_min) / dx)) + 1
            ny = int(np.ceil((y_max - y_min) / dy)) + 1
        else:
            ny, nx = grid_shape  # type: ignore[misc]
            if ny <= 1 or nx <= 1:
                raise ValueError("grid_shape must be at least (2,2)")
            dx = (x_max - x_min) / (nx - 1)
            dy = (y_max - y_min) / (ny - 1)

        x = np.linspace(x_min, x_max, nx, dtype=float)
        y = np.linspace(y_min, y_max, ny, dtype=float)

    field = np.full((ny, nx), -1.0, dtype=np.float64)

    # cutoff window per point for performance
    if support is None:
        support = 10.0 * float(radius)
    support = float(support)
    if support <= 0:
        raise ValueError("support must be > 0")

    # Add bumps
    for (px, py) in pts:
        # compute index window around the point
        ix0 = int(np.floor((px - support - x_min) / dx))
        ix1 = int(np.ceil((px + support - x_min) / dx))
        iy0 = int(np.floor((py - support - y_min) / dy))
        iy1 = int(np.ceil((py + support - y_min) / dy))
        ix0 = max(ix0, 0)
        iy0 = max(iy0, 0)
        ix1 = min(ix1, nx - 1)
        iy1 = min(iy1, ny - 1)
        if ix1 < ix0 or iy1 < iy0:
            continue

        xs = x[ix0 : ix1 + 1]
        ys = y[iy0 : iy1 + 1]
        # local mesh distances
        dxs = xs[None, :] - float(px)
        dys = ys[:, None] - float(py)
        r = np.sqrt(dxs * dxs + dys * dys)
        if bump_fn == sigmoid_bump:
            bump = bump_fn(r, float(radius), decay_rate=d, shift=shift)
        else:
            bump = bump_fn(r, float(radius))
        # ensure numeric, and only apply within support
        bump = np.asarray(bump, dtype=np.float64)
        bump[r > support] = 0.0

        # Combine bumps by pointwise maximum (not sum):
        # candidate = -1 + 2*bump lifts the background -1 to +1 at the center.
        candidate = -1.0 + 2.0 * bump
        window = field[iy0 : iy1 + 1, ix0 : ix1 + 1]
        field[iy0 : iy1 + 1, ix0 : ix1 + 1] = np.maximum(window, candidate)

    # Clip so the landscape never exceeds 1 (and never goes below -1)
    field = np.clip(field, clip[0], clip[1])

    return ContinuousLandscape2D(
        field=field,
        x=x,
        y=y,
        extent=(x_min, x_max, y_min, y_max),
        dx=float(dx),
        dy=float(dy),
    )


def build_voronoi_phase_landscape_from_cell_types(
    df: pd.DataFrame,
    *,
    negative_cell_class_value: str,
    positive_cell_class_value: str,
    cell_class_col: str = "cell_class",
    coord_cols: Tuple[str, str] = ("coord_X", "coord_Y"),
    # grid definition
    grid_spacing: Optional[float] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    pad: float = 0.0,
    w_range: Optional[Tuple[float, float, float, float]] = None,
    match_key: Optional[LandscapeKey] = None,
    # frontier definition
    transition_width: float = 1.0,
    clip: Tuple[float, float] = (-1.0, 1.0),
    # performance controls
    chunk_size: int = 65_536,
) -> ContinuousLandscape2D:
    """
    Build a smooth two-phase Voronoi-like landscape from two cell types.

    For each grid location, the field is driven by the nearest cell of each
    type:

        field = tanh((distance_to_negative_type - distance_to_positive_type) / transition_width)

    This gives values near -1 where the closest cell is from
    `negative_cell_class_value`, values near +1 where the closest cell is from
    `positive_cell_class_value`, and a smooth sigmoid-like transition across
    the Voronoi frontier where the two nearest distances are equal.

    Parameters
    ----------
    df:
        DataFrame containing both cell types and coordinates.
    negative_cell_class_value:
        Cell type assigned to the -1 phase.
    positive_cell_class_value:
        Cell type assigned to the +1 phase.
    grid_spacing:
        If provided, sets dx=dy=grid_spacing and grid size is derived from data bounds.
    grid_shape:
        If provided, explicitly sets (ny, nx). Exactly one of grid_spacing or grid_shape
        should be provided. If neither is provided, defaults to grid_shape=(512, 512).
    pad:
        Extra padding added around min/max bounds (in coordinate units).
    w_range:
        Optional explicit world-range for the grid as (x_min, x_max, y_min, y_max).
        If provided, these bounds are used instead of the tight bounds derived from
        both selected cell types (and `pad` is ignored).
    match_key:
        If provided, forces the landscape to use that global extent and spacing.
        This overrides `w_range`, `pad`, `grid_shape`, and `grid_spacing`.
    transition_width:
        Width of the smooth frontier in coordinate units. Smaller values make a
        sharper Voronoi boundary; larger values make a broader transition.
    clip:
        (min, max) clipping bounds for the final field. Should remain (-1, 1) for your use.
    chunk_size:
        Number of grid locations processed per nearest-distance query chunk.

    Returns
    -------
    ContinuousLandscape2D
        The field and grid metadata (coordinates, spacing, extent).
    """

    xcol, ycol = coord_cols
    required = {cell_class_col, xcol, ycol}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    neg_sub = df[df[cell_class_col] == negative_cell_class_value]
    pos_sub = df[df[cell_class_col] == positive_cell_class_value]
    if neg_sub.empty:
        raise ValueError(f"No rows found for {cell_class_col} == {negative_cell_class_value!r}")
    if pos_sub.empty:
        raise ValueError(f"No rows found for {cell_class_col} == {positive_cell_class_value!r}")

    neg_pts = neg_sub[[xcol, ycol]].to_numpy(dtype=float)
    pos_pts = pos_sub[[xcol, ycol]].to_numpy(dtype=float)
    neg_pts = neg_pts[np.isfinite(neg_pts).all(axis=1)]
    pos_pts = pos_pts[np.isfinite(pos_pts).all(axis=1)]
    if neg_pts.shape[0] == 0:
        raise ValueError(f"No finite coordinates for {negative_cell_class_value!r}.")
    if pos_pts.shape[0] == 0:
        raise ValueError(f"No finite coordinates for {positive_cell_class_value!r}.")

    all_pts = np.vstack((neg_pts, pos_pts))

    if match_key is not None:
        x_min = float(match_key["top_left"][0])
        y_max = float(match_key["top_left"][1])
        x_max = float(match_key["bottom_right"][0])
        y_min = float(match_key["bottom_right"][1])
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("match_key corners must satisfy x_max > x_min and y_max > y_min")
        dx = float(match_key["dx"])
        dy = float(match_key["dy"])
        if dx <= 0 or dy <= 0:
            raise ValueError("match_key dx/dy must be > 0")
    elif w_range is None:
        x_min = float(np.min(all_pts[:, 0]) - pad)
        x_max = float(np.max(all_pts[:, 0]) + pad)
        y_min = float(np.min(all_pts[:, 1]) - pad)
        y_max = float(np.max(all_pts[:, 1]) + pad)

        # Keep x/y on the same scale, matching build_continuous_landscape_from_points.
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        side = max(x_max - x_min, y_max - y_min)
        half = 0.5 * side
        x_min, x_max = cx - half, cx + half
        y_min, y_max = cy - half, cy + half
    else:
        x_min, x_max, y_min, y_max = (float(w_range[0]), float(w_range[1]), float(w_range[2]), float(w_range[3]))
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("w_range must satisfy x_max > x_min and y_max > y_min")

    if grid_spacing is not None and grid_shape is not None:
        raise ValueError("Provide exactly one of grid_spacing or grid_shape (not both).")

    if match_key is not None:
        nx = int(np.round((x_max - x_min) / dx)) + 1
        ny = int(np.round((y_max - y_min) / dy)) + 1
        if nx <= 1 or ny <= 1:
            raise ValueError("match_key leads to invalid grid shape; check extent/dx/dy")
        x = np.linspace(x_min, x_max, nx, dtype=float)
        y = np.linspace(y_min, y_max, ny, dtype=float)
    else:
        if grid_spacing is None and grid_shape is None:
            grid_shape = (512, 512)

        if grid_spacing is not None:
            dx = dy = float(grid_spacing)
            if dx <= 0:
                raise ValueError("grid_spacing must be > 0")
            nx = int(np.ceil((x_max - x_min) / dx)) + 1
            ny = int(np.ceil((y_max - y_min) / dy)) + 1
        else:
            ny, nx = grid_shape  # type: ignore[misc]
            if ny <= 1 or nx <= 1:
                raise ValueError("grid_shape must be at least (2,2)")
            dx = (x_max - x_min) / (nx - 1)
            dy = (y_max - y_min) / (ny - 1)

        x = np.linspace(x_min, x_max, nx, dtype=float)
        y = np.linspace(y_min, y_max, ny, dtype=float)

    transition_width = float(transition_width)
    if transition_width <= 0:
        raise ValueError("transition_width must be > 0")

    dist_neg = _nearest_distances_to_points(x, y, neg_pts, chunk_size=chunk_size)
    dist_pos = _nearest_distances_to_points(x, y, pos_pts, chunk_size=chunk_size)
    field = np.tanh((dist_neg - dist_pos) / transition_width)
    field = np.clip(field, clip[0], clip[1])

    return ContinuousLandscape2D(
        field=field,
        x=x,
        y=y,
        extent=(x_min, x_max, y_min, y_max),
        dx=float(dx),
        dy=float(dy),
    )

