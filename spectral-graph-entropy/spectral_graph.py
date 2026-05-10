from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
import IPython
from IPython.display import HTML, display


GraphMode = Literal["knn", "radius"]
WeightMode = Literal["binary", "distance", "gaussian"]
LaplacianMode = Literal["unnormalized", "normalized"]
SolverMode = Literal["auto", "dense", "arpack"]


@dataclass(frozen=True)
class SpectralGraphResult:
    """
    Output of the spectral graph computation.

    Attributes
    ----------
    eigenvalues:
        Array of computed Laplacian eigenvalues (sorted ascending).
    adjacency:
        Sparse symmetric adjacency matrix (n x n).
    laplacian:
        Sparse Laplacian matrix (n x n).
    coords:
        Coordinates used to build the graph (n x 2).
    indices:
        Row indices from the original dataframe included in the graph (length n).

    trace_laplacian:
        Trace of the (unscaled) Laplacian matrix, i.e. sum of diagonal entries.
    von_neumann_entropy:
        Von Neumann entropy of the scaled Laplacian rho = L / trace(L), computed as:
            S(rho) = - sum_i p_i log(p_i)
        where p_i are eigenvalues of rho (sum to 1).
        For large graphs this may be an approximation (see computation in function).
    """

    eigenvalues: np.ndarray
    adjacency: csr_matrix
    laplacian: csr_matrix
    coords: np.ndarray
    indices: np.ndarray
    trace_laplacian: float
    von_neumann_entropy: float

    def n_connected_components(self, *, tol: float = 1e-6) -> int:
        """
        Estimate the number of connected components from the Laplacian spectrum.

        For an (ideal) graph Laplacian, the multiplicity of eigenvalue 0 equals the
        number of connected components. With numerics, we count eigenvalues <= tol.

        Note: This is only reliable if `self.eigenvalues` includes the smallest part
        of the spectrum and `n_eigs` was large enough to capture all near-zero modes.
        """

        ev = np.asarray(self.eigenvalues, dtype=float)
        if ev.size == 0:
            return 0
        return int(np.sum(ev <= tol))

    def spectral_gap(self, *, tol: float = 1e-6) -> float:
        """
        Return the spectral gap (Fiedler value), i.e. the smallest *non-zero* eigenvalue.

        For connected graphs this is typically λ₂ (since λ₁ ≈ 0). If the graph has
        multiple components, λ₂ may also be near zero; we therefore skip eigenvalues
        <= tol and return the first above tol.
        """

        ev = np.sort(np.asarray(self.eigenvalues, dtype=float))
        nz = ev[ev > tol]
        return float(nz[0]) if nz.size else 0.0

    def plot_gap_and_connectivity(
        self,
        *,
        ax=None,
        figsize: Tuple[float, float] = (6.5, 3.2),
        tol: float = 1e-6,
        title: Optional[str] = None,
        color_gap: str = "#1f77b4",
        color_cc: str = "#ff7f0e",
    ):
        """
        Plot a compact summary of connectivity from the spectrum:
        - spectral gap (Fiedler value)
        - estimated number of connected components

        Returns (fig, ax).
        """

        import matplotlib.pyplot as plt

        gap = self.spectral_gap(tol=tol)
        ncc = self.n_connected_components(tol=tol)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Two bars with different scales: put components on secondary axis
        ax2 = ax.twinx()

        ax.bar([0], [gap], width=0.6, color=color_gap, label=f"Spectral gap (tol={tol:g})")
        ax.set_ylabel("Spectral gap (λ₂ / first non-zero λ)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["gap", "components"])

        ax2.bar([1], [ncc], width=0.6, color=color_cc, label="# connected components")
        ax2.set_ylabel("# connected components")

        if title is not None:
            ax.set_title(title)

        # Build a combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", frameon=False)

        fig.tight_layout()
        return fig, ax


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


def spectral_laplacian_spectrum_by_cell_class(
    df: pd.DataFrame,
    *,
    cell_class_value: str,
    mode: GraphMode = "knn",
    k: int = 15,
    radius: Optional[float] = None,
    weight: WeightMode = "binary",
    sigma: Optional[float] = None,
    laplacian: LaplacianMode = "normalized",
    n_eigs: int = 50,
    drop_self_loops: bool = True,
    random_state: int = 0,
    full_entropy_max_n: int = 5000,
    use_shift_invert: bool = True,
    solver: SolverMode = "auto",
    dense_solver_max_n: int = 3000,
) -> SpectralGraphResult:
    """
    Build a spatial graph from (coord_X, coord_Y) for one `cell_class` and compute
    a sparse Graph Laplacian spectrum using Lanczos (`scipy.sparse.linalg.eigsh`).

    This is intended to be lightweight:
    - graph construction via KD-tree: ~ O(N log N)
    - sparse eigenvalues via Lanczos: fast for large N when n_eigs is small

    Parameters
    ----------
    df:
        Pandas dataframe containing at least columns: 'cell_class', 'coord_X', 'coord_Y'.
    cell_class_value:
        Value to select in df['cell_class'].
    mode:
        'knn' for k-NN graph, or 'radius' for fixed-radius graph.
    k:
        Number of nearest neighbors (mode='knn'). Typical 10-30.
    radius:
        Neighborhood radius (mode='radius'). Required when mode='radius'.
    weight:
        Edge weighting:
        - 'binary': weight=1 for all edges
        - 'distance': weight=1/(d+eps)
        - 'gaussian': weight=exp(-d^2/(2*sigma^2))  (sigma required or auto-estimated)
    sigma:
        Gaussian kernel bandwidth when weight='gaussian'. If None, auto-estimate from
        median neighbor distance (knn mode) or median edge distance (radius mode).
    laplacian:
        'normalized' uses symmetric normalized Laplacian, 'unnormalized' uses combinatorial.
    n_eigs:
        Number of eigenvalues to compute (smallest eigenvalues). Must be < N.
    drop_self_loops:
        If True, remove any i->i edges.
    random_state:
        Seed passed to eigsh for reproducibility (initial vector).
    full_entropy_max_n:
        If number of nodes n <= this threshold, compute von Neumann entropy from the
        full Laplacian spectrum (dense eigendecomposition). Otherwise, compute an
        approximation based on the `n_eigs` smallest eigenvalues.
    use_shift_invert:
        If True, try shift-invert mode around sigma=0 to robustly compute eigenvalues
        near 0 (important when the Laplacian has multiple 0-eigenvalues / components).
        Falls back to a standard call if shift-invert fails.
    solver:
        Eigen-solver backend:
        - 'auto': use dense eigensolve when n <= dense_solver_max_n, otherwise ARPACK
        - 'dense': always use dense eigensolve (robust for small n)
        - 'arpack': always use ARPACK (eigsh)
    dense_solver_max_n:
        Node threshold for using the dense solver when solver='auto'.

    Returns
    -------
    SpectralGraphResult
        Contains eigenvalues and the matrices used.
    """

    required = {"cell_class", "coord_X", "coord_Y"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    sub = df[df["cell_class"] == cell_class_value].copy()
    if sub.empty:
        raise ValueError(f"No rows found for cell_class == {cell_class_value!r}")

    coords = sub[["coord_X", "coord_Y"]].to_numpy(dtype=float)
    indices = sub.index.to_numpy()

    # Drop rows with non-finite coordinates
    finite = np.isfinite(coords).all(axis=1)
    coords = coords[finite]
    indices = indices[finite]

    n = coords.shape[0]
    if n < 3:
        raise ValueError(f"Need at least 3 cells to build a graph; got {n}.")

    if n_eigs >= n:
        raise ValueError(f"n_eigs must be < number of nodes (n={n}); got n_eigs={n_eigs}.")

    tree = cKDTree(coords)

    # Build edge list (i, j, d)
    rows: list[int] = []
    cols: list[int] = []
    dists: list[float] = []

    if mode == "knn":
        if k <= 0:
            raise ValueError("k must be > 0 for mode='knn'")
        # query k+1 to include self, then drop it
        dd, jj = tree.query(coords, k=min(k + 1, n))
        # Ensure 2D
        dd = np.atleast_2d(dd)
        jj = np.atleast_2d(jj)

        for i in range(n):
            neigh = jj[i]
            dist = dd[i]
            for j, d in zip(neigh, dist):
                if drop_self_loops and int(j) == i:
                    continue
                rows.append(i)
                cols.append(int(j))
                dists.append(float(d))

    elif mode == "radius":
        if radius is None or radius <= 0:
            raise ValueError("radius must be provided and > 0 for mode='radius'")
        # neighbors within radius (includes self)
        neighs = tree.query_ball_point(coords, r=radius)
        for i, neigh in enumerate(neighs):
            for j in neigh:
                j = int(j)
                if drop_self_loops and j == i:
                    continue
                d = float(np.linalg.norm(coords[i] - coords[j]))
                rows.append(i)
                cols.append(j)
                dists.append(d)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'knn' or 'radius'.")

    if len(rows) == 0:
        raise ValueError("No edges were created. Try increasing k or radius.")

    rows_a = np.asarray(rows, dtype=np.int64)
    cols_a = np.asarray(cols, dtype=np.int64)
    d_a = np.asarray(dists, dtype=np.float64)

    # Weights
    if weight == "binary":
        w = np.ones_like(d_a, dtype=np.float64)
    elif weight == "distance":
        eps = 1e-8
        w = 1.0 / (d_a + eps)
    elif weight == "gaussian":
        if sigma is None:
            # robust auto-estimate
            sigma = float(np.median(d_a[d_a > 0])) if np.any(d_a > 0) else 1.0
        if sigma <= 0:
            raise ValueError("sigma must be > 0 for gaussian weights")
        w = np.exp(-(d_a**2) / (2.0 * float(sigma) ** 2))
    else:
        raise ValueError(f"Unknown weight {weight!r}. Use 'binary', 'distance', or 'gaussian'.")

    # Build sparse adjacency (directed), then symmetrize with max/mean.
    A = coo_matrix((w, (rows_a, cols_a)), shape=(n, n)).tocsr()

    # Symmetrize: keep an undirected graph
    # Using maximum makes sure if either direction exists, edge exists.
    A = A.maximum(A.T)

    if drop_self_loops:
        A.setdiag(0.0)
        A.eliminate_zeros()

    # Laplacian
    normed = laplacian == "normalized"
    L = csgraph_laplacian(A, normed=normed).tocsr()

    # Trace and scaled Laplacian rho = L / tr(L)
    trL = float(L.diagonal().sum())
    if trL <= 0:
        # Degenerate case: no edges or invalid graph; treat entropy as 0
        trL = 0.0
        vne = 0.0
    else:
        # Von Neumann entropy: S(rho) = -sum p log p, where p are eigenvalues of rho.
        # For small n, compute the full spectrum. For large n, approximate using the
        # smallest `n_eigs` eigenvalues of L (already computed below) and lump the
        # remaining probability mass into a single value (lower-bound style).
        vne = np.nan  # filled below

    # Compute eigenvalues near 0 (smallest).
    # For Laplacians with multiple connected components, there are multiple exact 0-eigenvalues.
    # ARPACK can be finicky about returning the correct multiplicity near 0 when weights vary.
    # Shift-invert around sigma=0 is typically the most robust approach.
    #
    # Note: normalized Laplacian eigenvalues lie in [0, 2].
    evals_full: Optional[np.ndarray] = None
    use_dense = (solver == "dense") or (solver == "auto" and n <= dense_solver_max_n)

    if use_dense:
        # For small graphs (e.g. a few hundred nodes), dense symmetric eigensolve
        # is robust and avoids ARPACK convergence issues.
        evals_full = np.linalg.eigvalsh(L.toarray())
        evals_full = np.sort(np.real(evals_full))
        evals = evals_full[:n_eigs]
    else:
        v0 = np.random.default_rng(random_state).normal(size=n)
        if use_shift_invert:
            try:
                # Shift-invert: eigenvalues closest to sigma via the inverted operator.
                evals = eigsh(
                    L,
                    k=n_eigs,
                    sigma=1e-6,
                    which="LM",
                    return_eigenvectors=False,
                    v0=v0,
                )
            except Exception:
                evals = eigsh(L, k=n_eigs, which="SA", return_eigenvectors=False, v0=v0)
        else:
            evals = eigsh(L, k=n_eigs, which="SA", return_eigenvectors=False, v0=v0)
        evals = np.sort(np.real(evals))

    if trL > 0:
        if n <= full_entropy_max_n:
            # Full spectrum entropy (dense).
            if evals_full is None:
                evals_full = np.linalg.eigvalsh(L.toarray())
                evals_full = np.sort(np.real(evals_full))
            p = np.clip(np.real(evals_full) / trL, 0.0, None)
            p_sum = float(p.sum())
            if p_sum > 0:
                p = p / p_sum  # renormalize for numerical stability
            p_pos = p[p > 0]
            vne = float(-(p_pos * np.log(p_pos)).sum())
        else:
            # Approximate entropy from computed smallest eigenvalues of L.
            p_small = np.clip(evals / trL, 0.0, None)
            mass_small = float(p_small.sum())
            # Remaining mass in uncomputed eigenvalues
            mass_rest = max(0.0, 1.0 - mass_small)
            p_pos = p_small[p_small > 0]
            vne = float(-(p_pos * np.log(p_pos)).sum())
            if mass_rest > 0:
                vne += float(-(mass_rest * np.log(mass_rest)))

    return SpectralGraphResult(
        eigenvalues=evals,
        adjacency=A,
        laplacian=L,
        coords=coords,
        indices=indices,
        trace_laplacian=trL,
        von_neumann_entropy=float(vne),
    )


def hks_per_node(
    graph: csr_matrix | np.ndarray | SpectralGraphResult,
    *,
    t: float,
    laplacian: LaplacianMode = "normalized",
    n_eigs: int = 128,
    random_state: int = 0,
    use_shift_invert: bool = True,
    solver: SolverMode = "auto",
    dense_solver_max_n: int = 3000,
) -> np.ndarray:
    """
    Compute Heat Kernel Signature (HKS) at a single diffusion time `t`.

    For a graph Laplacian L with eigenpairs (λ_k, φ_k), the HKS at node i is:
        HKS(i, t) = sum_k exp(-λ_k t) * φ_k(i)^2

    This function returns a vector of length n_nodes with the HKS value per node.

    Parameters
    ----------
    graph:
        One of:
        - `SpectralGraphResult` (uses its stored Laplacian)
        - a Laplacian matrix (dense or sparse)
        - an adjacency matrix (dense or sparse), in which case a Laplacian is built
          according to `laplacian` ('normalized' or 'unnormalized').
    t:
        Diffusion time. Must be positive.
    laplacian:
        Only used when `graph` is an adjacency matrix (to decide how to build L).
    n_eigs:
        Number of eigenpairs to use. Must be < n_nodes. If larger than n_nodes-1,
        it is clipped to n_nodes-1.
    random_state:
        Seed for ARPACK initial vector.
    use_shift_invert:
        If True, try shift-invert around 0 for robustness near the Laplacian nullspace.
    solver:
        'auto' uses dense solver when n <= dense_solver_max_n, else ARPACK.
    dense_solver_max_n:
        Node threshold for dense eigendecomposition when solver='auto'.

    Returns
    -------
    np.ndarray
        Vector of shape (n_nodes,) with HKS values at diffusion time t.
    """

    if not np.isfinite(t) or t <= 0:
        raise ValueError(f"t must be a positive finite number; got t={t!r}.")

    # Normalize input to a Laplacian matrix L
    if isinstance(graph, SpectralGraphResult):
        L = graph.laplacian
    else:
        L = graph

    # Heuristic: treat as adjacency if it doesn't look like a Laplacian
    # (i.e., if diagonal isn't mostly positive or row sums aren't near 0).
    # Users can pass an actual Laplacian directly and it will work as-is.
    if isinstance(L, csr_matrix):
        n = int(L.shape[0])
        if L.shape[0] != L.shape[1]:
            raise ValueError(f"graph must be square; got shape={L.shape}.")
        diag = np.asarray(L.diagonal(), dtype=float)
        row_sum = np.asarray(L.sum(axis=1)).reshape(-1)
        looks_like_laplacian = (np.nanmedian(diag) > 0) and (np.nanmedian(np.abs(row_sum)) < 1e-6)
        if not looks_like_laplacian:
            normed = laplacian == "normalized"
            L = csgraph_laplacian(L, normed=normed).tocsr()
    else:
        L = np.asarray(L, dtype=float)
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(f"graph must be square; got shape={getattr(L, 'shape', None)}.")
        n = int(L.shape[0])
        diag = np.diag(L)
        row_sum = L.sum(axis=1)
        looks_like_laplacian = (np.nanmedian(diag) > 0) and (np.nanmedian(np.abs(row_sum)) < 1e-6)
        if not looks_like_laplacian:
            normed = laplacian == "normalized"
            L = csgraph_laplacian(csr_matrix(L), normed=normed).tocsr()

    n = int(L.shape[0])
    if n < 2:
        return np.zeros((n,), dtype=float)

    k = int(min(max(1, n_eigs), n - 1))

    use_dense = (solver == "dense") or (solver == "auto" and n <= dense_solver_max_n)
    if use_dense:
        if isinstance(L, csr_matrix):
            Ld = L.toarray()
        else:
            Ld = np.asarray(L, dtype=float)
        evals, evecs = np.linalg.eigh(Ld)
        evals = np.real(evals[:k])
        evecs = np.real(evecs[:, :k])
    else:
        v0 = np.random.default_rng(random_state).normal(size=n)
        if use_shift_invert:
            try:
                evals, evecs = eigsh(L, k=k, sigma=1e-6, which="LM", v0=v0)
            except Exception:
                evals, evecs = eigsh(L, k=k, which="SA", v0=v0)
        else:
            evals, evecs = eigsh(L, k=k, which="SA", v0=v0)
        order = np.argsort(np.real(evals))
        evals = np.real(evals[order])
        evecs = np.real(evecs[:, order])

    weights = np.exp(-evals * float(t))  # (k,)
    hks = (evecs**2) @ weights  # (n,)
    return np.asarray(hks, dtype=float)


def _spatial_adjacency_from_coords(
    coords: np.ndarray,
    *,
    mode: GraphMode = "knn",
    k: int = 15,
    radius: Optional[float] = None,
    weight: WeightMode = "binary",
    sigma: Optional[float] = None,
    drop_self_loops: bool = True,
) -> csr_matrix:
    """Build the same symmetric spatial adjacency used by the spectral utilities."""

    n = coords.shape[0]
    tree = cKDTree(coords)

    rows: list[int] = []
    cols: list[int] = []
    dists: list[float] = []

    if mode == "knn":
        if k <= 0:
            raise ValueError("k must be > 0 for mode='knn'")
        dd, jj = tree.query(coords, k=min(k + 1, n))
        dd = np.atleast_2d(dd)
        jj = np.atleast_2d(jj)

        for i in range(n):
            for j, d in zip(jj[i], dd[i]):
                if drop_self_loops and int(j) == i:
                    continue
                rows.append(i)
                cols.append(int(j))
                dists.append(float(d))
    elif mode == "radius":
        if radius is None or radius <= 0:
            raise ValueError("radius must be provided and > 0 for mode='radius'")
        neighs = tree.query_ball_point(coords, r=radius)
        for i, neigh in enumerate(neighs):
            for j in neigh:
                j = int(j)
                if drop_self_loops and j == i:
                    continue
                rows.append(i)
                cols.append(j)
                dists.append(float(np.linalg.norm(coords[i] - coords[j])))
    else:
        raise ValueError(f"Unknown mode {mode!r}. Use 'knn' or 'radius'.")

    if len(rows) == 0:
        raise ValueError("No edges were created. Try increasing k or radius.")

    rows_a = np.asarray(rows, dtype=np.int64)
    cols_a = np.asarray(cols, dtype=np.int64)
    d_a = np.asarray(dists, dtype=np.float64)

    if weight == "binary":
        w = np.ones_like(d_a, dtype=np.float64)
    elif weight == "distance":
        w = 1.0 / (d_a + 1e-8)
    elif weight == "gaussian":
        if sigma is None:
            sigma = float(np.median(d_a[d_a > 0])) if np.any(d_a > 0) else 1.0
        if sigma <= 0:
            raise ValueError("sigma must be > 0 for gaussian weights")
        w = np.exp(-(d_a**2) / (2.0 * float(sigma) ** 2))
    else:
        raise ValueError(f"Unknown weight {weight!r}. Use 'binary', 'distance', or 'gaussian'.")

    A = coo_matrix((w, (rows_a, cols_a)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    if drop_self_loops:
        A.setdiag(0.0)
        A.eliminate_zeros()
    return A


def plot_hks_video(
    df: pd.DataFrame,
    *,
    min_time: float,
    max_time: float,
    n_points: int,
    cell_class_value: Optional[str] = None,
    coord_cols: Tuple[str, str] = ("coord_X", "coord_Y"),
    cell_class_col: str = "cell_class",
    mode: GraphMode = "knn",
    scale: Literal["linear", "log"] = "log",
    k: int = 15,
    radius: Optional[float] = None,
    weight: WeightMode = "binary",
    sigma: Optional[float] = None,
    laplacian: LaplacianMode = "normalized",
    n_eigs: int = 128,
    drop_self_loops: bool = True,
    random_state: int = 0,
    use_shift_invert: bool = True,
    solver: SolverMode = "auto",
    dense_solver_max_n: int = 3000,
    cmap: str = "magma",
    node_size: float = 22.0,
    edge_color: str = "#9a9a9a",
    edge_alpha: float = 0.14,
    max_edges: int = 6000,
    figsize: Tuple[float, float] = (7.0, 6.0),
    interval: int = 250,
    interactive_sliders: bool = True,
    slider_backend: Literal["ipywidgets", "matplotlib"] = "ipywidgets",
    save_path: Optional[str] = None,
    dpi: int = 150,
    title: Optional[str] = None,
):
    """
    Animate or interactively explore Heat Kernel Signature (HKS) values on a spatial graph.

    Nodes are plotted at their original dataframe coordinates. If `interactive_sliders`
    is True, the output includes one slider for diffusion time and one slider for
    the number of eigenpairs used in the HKS truncation. If False, it returns a
    time animation using all `n_eigs` eigenpairs.

    Parameters
    ----------
    df:
        DataFrame containing coordinate columns, by default 'coord_X' and 'coord_Y'.
    min_time, max_time:
        Inclusive diffusion-time range to animate. Both must be positive.
    n_points:
        Number of diffusion times / animation frames.
    cell_class_value:
        Optional value used to filter `df[cell_class_col]` before building the graph.
        If None, all rows are used.
    coord_cols:
        Coordinate column names as (x_col, y_col).
    mode, k, radius, weight, sigma, laplacian:
        Spatial graph and Laplacian construction options. These match the options used
        by `spectral_laplacian_spectrum_by_cell_class`.
    n_eigs:
        Maximum number of Laplacian eigenpairs used for HKS. In interactive mode,
        the eigenvalue slider lets you choose any truncation from 1 to `n_eigs`.
    max_edges:
        Maximum number of undirected edges drawn in the background. Set to 0 to hide
        edges, or None to draw all edges.
    interactive_sliders:
        If True, return an interactive figure with time and eigenvalue-count sliders.
        If False, return a `FuncAnimation` over time using all `n_eigs`.
    slider_backend:
        'ipywidgets' creates notebook-native sliders that work with inline figures.
        'matplotlib' creates Matplotlib sliders inside the figure canvas and requires
        an interactive Matplotlib backend such as `%matplotlib widget`.
    save_path:
        Optional path ending in .mp4 or .gif. Only used when `interactive_sliders=False`.

    Returns
    -------
    matplotlib.animation.FuncAnimation or tuple
        If `interactive_sliders=False`, returns a FuncAnimation. If
        `interactive_sliders=True`, returns `(fig, ax, controls)` where `controls`
        contains the two slider objects and, for `slider_backend='ipywidgets'`, the
        displayed widget UI.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    if not np.isfinite(min_time) or min_time <= 0:
        raise ValueError(f"min_time must be a positive finite number; got {min_time!r}.")
    if not np.isfinite(max_time) or max_time <= 0:
        raise ValueError(f"max_time must be a positive finite number; got {max_time!r}.")
    if max_time < min_time:
        raise ValueError("max_time must be >= min_time.")
    if n_points < 1:
        raise ValueError("n_points must be >= 1.")

    x_col, y_col = coord_cols
    required = {x_col, y_col}
    if cell_class_value is not None:
        required.add(cell_class_col)
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    sub = df.copy()
    if cell_class_value is not None:
        sub = sub[sub[cell_class_col] == cell_class_value].copy()
        if sub.empty:
            raise ValueError(f"No rows found for {cell_class_col} == {cell_class_value!r}")

    coords = sub[[x_col, y_col]].to_numpy(dtype=float)
    finite = np.isfinite(coords).all(axis=1)
    coords = coords[finite]
    if coords.shape[0] < 3:
        raise ValueError(f"Need at least 3 finite points to build a graph; got {coords.shape[0]}.")

    A = _spatial_adjacency_from_coords(
        coords,
        mode=mode,
        k=k,
        radius=radius,
        weight=weight,
        sigma=sigma,
        drop_self_loops=drop_self_loops,
    )
    L = csgraph_laplacian(A, normed=(laplacian == "normalized")).tocsr()

    n = coords.shape[0]
    k_eigs = int(min(max(1, n_eigs), n - 1))
    use_dense = (solver == "dense") or (solver == "auto" and n <= dense_solver_max_n)
    if use_dense:
        evals, evecs = np.linalg.eigh(L.toarray())
        evals = np.real(evals[:k_eigs])
        evecs = np.real(evecs[:, :k_eigs])
    else:
        v0 = np.random.default_rng(random_state).normal(size=n)
        if use_shift_invert:
            try:
                evals, evecs = eigsh(L, k=k_eigs, sigma=1e-6, which="LM", v0=v0)
            except Exception:
                evals, evecs = eigsh(L, k=k_eigs, which="SA", v0=v0)
        else:
            evals, evecs = eigsh(L, k=k_eigs, which="SA", v0=v0)
        order = np.argsort(np.real(evals))
        evals = np.real(evals[order])
        evecs = np.real(evecs[:, order])
    if scale == "log":
        times = np.logspace(np.log10(min_time), np.log10(max_time), int(n_points))
    elif scale == "linear":
        times = np.linspace(float(min_time), float(max_time), int(n_points))
    else:
        raise ValueError(f"Unknown scale {scale!r}. Use 'log' or 'linear'.")

    evecs_sq = evecs**2
    first_eig_hks = evecs_sq[:, :1] @ np.exp(-np.outer(evals[:1], times))
    full_hks_by_time = evecs_sq @ np.exp(-np.outer(evals, times))
    first_eig_hks = np.asarray(first_eig_hks, dtype=float)
    full_hks_by_time = np.asarray(full_hks_by_time, dtype=float)
    vmin = float(np.nanmin(first_eig_hks))
    vmax = float(np.nanmax(full_hks_by_time))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, ax = plt.subplots(figsize=figsize)
    if interactive_sliders and slider_backend == "matplotlib":
        fig.subplots_adjust(bottom=0.20)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("#101014")
    fig.patch.set_facecolor("#101014")

    if max_edges is None or max_edges != 0:
        row, col = A.nonzero()
        edge_mask = row < col
        edges = np.column_stack([row[edge_mask], col[edge_mask]])
        if max_edges is not None and edges.shape[0] > max_edges:
            edge_idx = np.linspace(0, edges.shape[0] - 1, int(max_edges), dtype=int)
            edges = edges[edge_idx]
        segments = np.stack([coords[edges[:, 0]], coords[edges[:, 1]]], axis=1) if edges.size else []
        ax.add_collection(LineCollection(segments, colors=edge_color, linewidths=0.45, alpha=edge_alpha))

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=full_hks_by_time[:, 0],
        s=node_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
        alpha=0.96,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("HKS", color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.outline.set_edgecolor("white")

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
    ax.set_xlabel(x_col, color="white")
    ax.set_ylabel(y_col, color="white")

    base_title = title or "Heat Kernel Signature on Spatial Graph"
    title_artist = ax.set_title("", color="white", pad=12)

    def hks_for(time_idx: int, eig_count: int) -> np.ndarray:
        eig_count = int(np.clip(eig_count, 1, k_eigs))
        time_idx = int(np.clip(time_idx, 0, len(times) - 1))
        return np.asarray(
            evecs_sq[:, :eig_count] @ np.exp(-evals[:eig_count] * times[time_idx]),
            dtype=float,
        )

    def set_frame(time_idx: int, eig_count: int):
        hks = hks_for(time_idx, eig_count)
        scatter.set_array(hks)
        title_artist.set_text(
            f"{base_title}\nt = {times[time_idx]:.4g}, eigenpairs = {eig_count}/{k_eigs}"
        )
        return scatter, title_artist

    if interactive_sliders and slider_backend == "ipywidgets":
        try:
            import ipywidgets as widgets
            from IPython.display import clear_output, display
        except ImportError as exc:
            raise ImportError(
                "slider_backend='ipywidgets' requires ipywidgets and IPython. "
                "Install ipywidgets or use slider_backend='matplotlib' with "
                "`%matplotlib widget`."
            ) from exc

        time_slider = widgets.SelectionSlider(
            options=[(f"{t:.4g}", i) for i, t in enumerate(times)],
            value=0,
            description="time",
            continuous_update=False,
            layout=widgets.Layout(width="80%"),
        )
        eig_slider = widgets.IntSlider(
            value=k_eigs,
            min=1,
            max=k_eigs,
            step=1,
            description="eigenpairs",
            continuous_update=False,
            layout=widgets.Layout(width="80%"),
        )
        output = widgets.Output()

        def render_widget_plot(_=None):
            set_frame(int(time_slider.value), int(eig_slider.value))
            with output:
                clear_output(wait=True)
                display(fig)

        time_slider.observe(render_widget_plot, names="value")
        eig_slider.observe(render_widget_plot, names="value")

        # Avoid an extra static Matplotlib output before the widget-controlled plot.
        plt.close(fig)
        render_widget_plot()
        ui = widgets.VBox([time_slider, eig_slider, output])
        display(ui)

        controls = {
            "time_slider": time_slider,
            "eigenpairs_slider": eig_slider,
            "output": output,
            "ui": ui,
            "times": times,
            "max_eigenpairs": k_eigs,
        }
        return fig, ax, controls

    if interactive_sliders and slider_backend == "matplotlib":
        from matplotlib.widgets import Slider

        time_ax = fig.add_axes([0.18, 0.09, 0.66, 0.03], facecolor="#24242c")
        eig_ax = fig.add_axes([0.18, 0.04, 0.66, 0.03], facecolor="#24242c")

        time_slider = Slider(
            time_ax,
            "time",
            float(times[0]),
            float(times[-1]),
            valinit=float(times[0]),
            valstep=times,
            color="#ffb000",
        )  
        eig_slider = Slider(
            eig_ax,
            "eigenpairs",
            1,
            k_eigs,
            valinit=k_eigs,
            valstep=1,
            color="#ffb000",
        )
        time_slider.label.set_color("white")
        time_slider.valtext.set_color("white")
        eig_slider.label.set_color("white")
        eig_slider.valtext.set_color("white")

        def on_slider_change(_):
            time_idx = int(np.argmin(np.abs(times - float(time_slider.val))))
            set_frame(time_idx, int(eig_slider.val))
            fig.canvas.draw_idle()

        time_slider.on_changed(on_slider_change)
        eig_slider.on_changed(on_slider_change)
        set_frame(0, k_eigs)

        controls = {
            "time_slider": time_slider,
            "eigenpairs_slider": eig_slider,
            "times": times,
            "max_eigenpairs": k_eigs,
        }
        return fig, ax, controls

    if interactive_sliders:
        raise ValueError(f"Unknown slider_backend {slider_backend!r}. Use 'ipywidgets' or 'matplotlib'.")

    def update(frame: int):
        return set_frame(frame, k_eigs)

    anim = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=False)
    set_frame(0, k_eigs)

    if save_path is not None:
        anim.save(save_path, dpi=dpi)

    return anim

