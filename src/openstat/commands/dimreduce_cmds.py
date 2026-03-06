"""Dimensionality reduction: t-SNE, UMAP, PCA plot."""

from __future__ import annotations
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("tsne", usage="tsne [cols...] [--n=2] [--perplexity=30] [--out=tsne.png] [--color=col]")
def cmd_tsne(session: Session, args: str) -> str:
    """t-SNE dimensionality reduction and visualization.

    Options:
      --n=<dim>          output dimensions (2 or 3, default: 2)
      --perplexity=<p>   perplexity (5–50, default: 30)
      --iter=<n>         iterations (default: 1000)
      --color=<col>      column to colour points by
      --out=<path>       output image path

    Examples:
      tsne x1 x2 x3 x4 --color=label
      tsne --perplexity=20 --iter=2000 --out=tsne_result.png
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    ca = CommandArgs(args)
    n_dim = int(ca.options.get("n", 2))
    perplexity = float(ca.options.get("perplexity", 30))
    n_iter = int(ca.options.get("iter", 1000))
    color_col = ca.options.get("color")
    out_path = ca.options.get("out", str(session.output_dir / "tsne.png"))

    try:
        df = session.require_data()
        NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        if ca.positional:
            cols = [c for c in ca.positional if c in df.columns]
        else:
            cols = [c for c in df.columns if df[c].dtype in NUMERIC]

        if len(cols) < 2:
            return "Need at least 2 numeric columns for t-SNE."

        sub_cols = cols[:]
        if color_col and color_col in df.columns and color_col not in sub_cols:
            sub_cols.append(color_col)

        sub = df.select(sub_cols).drop_nulls()
        X = sub.select(cols).to_numpy().astype(float)

        if len(X) < 5:
            return "Need at least 5 rows for t-SNE."

        perplexity = min(perplexity, len(X) - 1)
        tsne = TSNE(n_components=n_dim, perplexity=perplexity, max_iter=n_iter,
                    random_state=42)
        embedding = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        if color_col and color_col in sub.columns:
            cats = sub[color_col].cast(pl.Utf8).to_list()
            unique_cats = sorted(set(cats))
            cmap = plt.colormaps.get_cmap("tab10")
            for i, cat in enumerate(unique_cats):
                mask = [c == cat for c in cats]
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           label=str(cat), alpha=0.7, s=20,
                           color=cmap(i / max(len(unique_cats), 1)))
            ax.legend(title=color_col, markerscale=2, fontsize=8)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20, color="#4C72B0")

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"t-SNE ({len(cols)} features, perplexity={perplexity:.0f})")
        fig.tight_layout()

        session.output_dir.mkdir(parents=True, exist_ok=True)
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        session.plot_paths.append(out_path)
        return f"t-SNE plot saved: {out_path}  (n={len(X)}, features={len(cols)})"
    except Exception as e:
        return friendly_error(e, "tsne")


@command("umap", usage="umap [cols...] [--n=2] [--neighbors=15] [--out=umap.png] [--color=col]")
def cmd_umap(session: Session, args: str) -> str:
    """UMAP dimensionality reduction and visualization.

    Options:
      --n=<dim>          output dimensions (2 or 3, default: 2)
      --neighbors=<k>    number of neighbors (default: 15)
      --mindist=<d>      minimum distance (default: 0.1)
      --color=<col>      column to colour points by
      --out=<path>       output image path

    Examples:
      umap x1 x2 x3 x4 --color=label
      umap --neighbors=20 --mindist=0.05
    """
    try:
        import umap as umap_lib
    except ImportError:
        return "umap-learn required. Install: pip install umap-learn"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl

    ca = CommandArgs(args)
    n_dim = int(ca.options.get("n", 2))
    n_neighbors = int(ca.options.get("neighbors", 15))
    min_dist = float(ca.options.get("mindist", 0.1))
    color_col = ca.options.get("color")
    out_path = ca.options.get("out", str(session.output_dir / "umap.png"))

    try:
        df = session.require_data()
        NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                   pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        if ca.positional:
            cols = [c for c in ca.positional if c in df.columns]
        else:
            cols = [c for c in df.columns if df[c].dtype in NUMERIC]

        if len(cols) < 2:
            return "Need at least 2 numeric columns for UMAP."

        sub_cols = cols[:]
        if color_col and color_col in df.columns and color_col not in sub_cols:
            sub_cols.append(color_col)

        sub = df.select(sub_cols).drop_nulls()
        X = sub.select(cols).to_numpy().astype(float)

        if len(X) < 4:
            return "Need at least 4 rows for UMAP."

        n_neighbors = min(n_neighbors, len(X) - 1)
        reducer = umap_lib.UMAP(n_components=n_dim, n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=42)
        embedding = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        if color_col and color_col in sub.columns:
            cats = sub[color_col].cast(pl.Utf8).to_list()
            unique_cats = sorted(set(cats))
            cmap = plt.colormaps.get_cmap("tab10")
            for i, cat in enumerate(unique_cats):
                mask = [c == cat for c in cats]
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           label=str(cat), alpha=0.7, s=20,
                           color=cmap(i / max(len(unique_cats), 1)))
            ax.legend(title=color_col, markerscale=2, fontsize=8)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20, color="#4C72B0")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP ({len(cols)} features, neighbors={n_neighbors})")
        fig.tight_layout()

        session.output_dir.mkdir(parents=True, exist_ok=True)
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        session.plot_paths.append(out_path)
        return f"UMAP plot saved: {out_path}  (n={len(X)}, features={len(cols)})"
    except Exception as e:
        return friendly_error(e, "umap")
