from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
try:
    import anndata as ad
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'anndata'. Install it (e.g. `pip install anndata` or `conda install -c conda-forge anndata`) "
        "or run this script inside your scanpy/anndata environment."
    ) from e

try:
    import scipy.sparse as sp
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "Missing dependency 'scipy'. Install it (e.g. `pip install scipy` or `conda install -c conda-forge scipy`)."
    ) from e


DEFAULT_META_COLS = ("cell_id", "cell_class", "coord_X", "coord_Y")


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _infer_gene_cols(df: pd.DataFrame, meta_cols: Iterable[str]) -> list[str]:
    meta_cols = list(meta_cols)
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
    return [c for c in df.columns if c not in meta_cols]


def _coerce_expression_matrix(df: pd.DataFrame, gene_cols: list[str]) -> np.ndarray:
    # Robust conversion: object/mixed columns -> numeric, non-numeric -> NaN -> 0.
    X = df[gene_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _to_adata(df: pd.DataFrame, *, library: str, meta_cols=DEFAULT_META_COLS) -> ad.AnnData:
    gene_cols = _infer_gene_cols(df, meta_cols)
    X = _coerce_expression_matrix(df, gene_cols)

    obs = df[list(meta_cols)].copy()
    # Ensure stable string index and uniqueness once we add library.
    obs["cell_id"] = obs["cell_id"].astype(str)
    obs["library"] = library
    obs_names = obs["cell_id"].astype(str) + f"__{library}"
    obs.index = obs_names

    var = pd.DataFrame(index=pd.Index(gene_cols, name="gene"))

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=obs,
        var=var,
        dtype=np.float32,
    )

    # Scanpy's spatial plotting expects coordinates in obsm["spatial"].
    adata.obsm["spatial"] = obs[["coord_X", "coord_Y"]].to_numpy(dtype=np.float32)
    return adata


def build_fused_adata(
    pred_csv: str | Path,
    true_csv: str | Path,
    *,
    library_key: str = "library",
    pred_label: str = "pred",
    true_label: str = "gt",
) -> ad.AnnData:
    df_pred = _read_csv(pred_csv)
    df_true = _read_csv(true_csv)

    pred_genes = _infer_gene_cols(df_pred, DEFAULT_META_COLS)
    true_genes = _infer_gene_cols(df_true, DEFAULT_META_COLS)
    if pred_genes != true_genes:
        # Keep ordering consistent; fail loudly because mismatched genes would silently corrupt results.
        missing_in_pred = sorted(set(true_genes) - set(pred_genes))
        missing_in_true = sorted(set(pred_genes) - set(true_genes))
        raise ValueError(
            "Gene columns differ between CSVs.\n"
            f"- missing in pred: {missing_in_pred[:20]}{' ...' if len(missing_in_pred) > 20 else ''}\n"
            f"- missing in true: {missing_in_true[:20]}{' ...' if len(missing_in_true) > 20 else ''}\n"
            f"- pred gene count: {len(pred_genes)}\n"
            f"- true gene count: {len(true_genes)}"
        )

    ad_pred = _to_adata(df_pred, library=pred_label)
    ad_true = _to_adata(df_true, library=true_label)

    fused = ad.concat(
        {true_label: ad_true, pred_label: ad_pred},
        axis=0,
        join="outer",
        label=library_key,
        index_unique=None,
        merge="same",
    )

    # `anndata.concat` already adds `library_key`; ensure it's a simple column for filtering.
    fused.obs[library_key] = fused.obs[library_key].astype(str)
    return fused


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build a single AnnData from pred/true CSVs (metadata + gene columns), "
            "adding obs['library'] = 'gt' or 'pred' and obsm['spatial'] from coord_X/coord_Y."
        )
    )
    p.add_argument(
        "--pred",
        required=True,
        help="Path to pd_pred_copy_with_genes.csv",
    )
    p.add_argument(
        "--true",
        required=True,
        help="Path to pd_true_copy_with_genes.csv",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output .h5ad path to write.",
    )
    p.add_argument(
        "--library-key",
        default="library",
        help="obs column name to store library labels (default: library).",
    )
    p.add_argument(
        "--pred-label",
        default="pred",
        help="Label for prediction rows (default: pred).",
    )
    p.add_argument(
        "--true-label",
        default="gt",
        help="Label for ground-truth rows (default: gt).",
    )

    args = p.parse_args()
    fused = build_fused_adata(
        args.pred,
        args.true,
        library_key=args.library_key,
        pred_label=args.pred_label,
        true_label=args.true_label,
    )

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fused.write_h5ad(out_path)


if __name__ == "__main__":
    main()
