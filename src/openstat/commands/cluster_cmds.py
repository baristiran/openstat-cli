"""Clustering, MDS, and discriminant analysis commands."""

from __future__ import annotations

import re

from openstat.commands.base import command
from openstat.session import Session


def _stata_opts(raw: str) -> tuple[list[str], dict[str, str]]:
    opts: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\(([^)]*)\)', raw):
        opts[m.group(1).lower()] = m.group(2)
    rest = re.sub(r'\w+\([^)]*\)', '', raw)
    positional = [t.strip(',') for t in rest.split() if t.strip(',')]
    return positional, opts


@command("cluster", usage="cluster kmeans|hierarchical varlist [, k(3) linkage(ward)]")
def cmd_cluster(session: Session, args: str) -> str:
    """K-means or hierarchical clustering."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return (
            "Usage: cluster kmeans varlist [, k(3)]\n"
            "       cluster hierarchical varlist [, k(3) linkage(ward)]"
        )
    sub = positional[0].lower()
    cols = [c for c in positional[1:] if c in df.columns]
    if not cols:
        return "No valid columns found."

    k = int(opts.get("k", 3))

    try:
        if sub in ("kmeans", "k-means"):
            from openstat.stats.clustering import fit_kmeans
            result = fit_kmeans(df, cols, k=k)
            lines = [f"\nK-Means Clustering (k={k})", "=" * 50]
            lines.append(f"  {'N observations':<25}  {result['n_obs']}")
            lines.append(f"  {'Inertia':<25}  {result['inertia']:.4f}")
            lines.append(f"  {'Silhouette score':<25}  {result['silhouette_score']:.4f}")
            lines.append(f"  {'Calinski-Harabasz':<25}  {result['calinski_harabasz']:.4f}")
            lines.append("\nCluster sizes:")
            for cl, n in result["cluster_sizes"].items():
                lines.append(f"  Cluster {cl + 1}: {n} obs ({n/result['n_obs']*100:.1f}%)")
            lines.append("\nCentroids (original scale):")
            lines.append("  " + f"{'Cluster':<10}" + "".join(f"  {c[:8]:>8}" for c in cols))
            for i, cent in enumerate(result["centroids"]):
                row = f"  {'Cl.' + str(i+1):<10}"
                for v in cent:
                    row += f"  {v:>8.3f}"
                lines.append(row)
            lines.append("=" * 50)
            session._last_model = result
            return "\n".join(lines)

        elif sub in ("hierarchical", "hier", "agglomerative"):
            linkage = opts.get("linkage", "ward")
            from openstat.stats.clustering import fit_hierarchical
            result = fit_hierarchical(df, cols, k=k, linkage=linkage)
            lines = [f"\nHierarchical Clustering (k={k}, linkage={linkage})", "=" * 50]
            lines.append(f"  {'N observations':<25}  {result['n_obs']}")
            lines.append(f"  {'Silhouette score':<25}  {result['silhouette_score']:.4f}")
            lines.append(f"  {'Calinski-Harabasz':<25}  {result['calinski_harabasz']:.4f}")
            lines.append("\nCluster sizes:")
            for cl, n in result["cluster_sizes"].items():
                lines.append(f"  Cluster {cl + 1}: {n} obs")
            lines.append("=" * 50)
            session._last_model = result
            return "\n".join(lines)

        else:
            return f"Unknown cluster method: {sub}. Use 'kmeans' or 'hierarchical'."

    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"cluster error: {exc}"


@command("mds", usage="mds varlist [, n(2) metric]")
def cmd_mds(session: Session, args: str) -> str:
    """Multidimensional scaling."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    cols = [c for c in positional if c in df.columns]
    if len(cols) < 2:
        return "mds requires at least 2 numeric variables."
    n_comp = int(opts.get("n", 2))
    metric = "nonmetric" not in positional

    try:
        from openstat.stats.clustering import fit_mds
        result = fit_mds(df, cols, n_components=n_comp, metric=metric)
        lines = [f"\nMDS ({'metric' if metric else 'non-metric'})", "=" * 50]
        lines.append(f"  {'N observations':<25}  {result['n_obs']}")
        lines.append(f"  {'N components':<25}  {n_comp}")
        lines.append(f"  {'Stress':<25}  {result['stress']:.6f}")
        lines.append("\nFirst 5 coordinates (Dim 1, Dim 2):")
        for i, coord in enumerate(result["coordinates"][:5]):
            lines.append("  " + "  ".join(f"{v:>8.4f}" for v in coord))
        if len(result["coordinates"]) > 5:
            lines.append(f"  ... ({len(result['coordinates'])} total rows)")
        lines.append("=" * 50)
        session._last_model = result
        return "\n".join(lines)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"mds error: {exc}"


@command("discriminant", usage="discriminant groupvar indepvars [, method(lda|qda)]")
def cmd_discriminant(session: Session, args: str) -> str:
    """Linear (LDA) or Quadratic (QDA) Discriminant Analysis."""
    df = session.require_data()
    positional, opts = _stata_opts(args)
    if len(positional) < 2:
        return "Usage: discriminant groupvar indepvar1 indepvar2 ... [, method(lda)]"

    dep = positional[0]
    indeps = [c for c in positional[1:] if c in df.columns]
    method = opts.get("method", "lda").lower()

    try:
        from openstat.stats.clustering import fit_discriminant
        result = fit_discriminant(df, dep, indeps, method=method)
        lines = [f"\n{result['method']}: {dep}", "=" * 55]
        lines.append(f"  {'N observations':<25}  {result['n_obs']}")
        lines.append(f"  {'N classes':<25}  {result['n_classes']}")
        lines.append(f"  {'Classes':<25}  {', '.join(str(c) for c in result['classes'])}")
        lines.append(f"  {'Accuracy (train)':<25}  {result['accuracy']:>12.4f}")
        if result.get("priors"):
            lines.append("\nClass priors:")
            for cls, p in zip(result["classes"], result["priors"]):
                lines.append(f"  {str(cls):<20}  {p:.4f}")
        if "coefficients" in result:
            lines.append("\nDiscriminant function coefficients:")
            for func, coefs in result["coefficients"].items():
                lines.append(f"  Function ({func}):")
                for var, val in coefs.items():
                    lines.append(f"    {var:<22}  {val:>10.4f}")
        lines.append("=" * 55)
        session._last_model = result
        return "\n".join(lines)
    except ImportError as e:
        return str(e)
    except Exception as exc:
        return f"discriminant error: {exc}"
