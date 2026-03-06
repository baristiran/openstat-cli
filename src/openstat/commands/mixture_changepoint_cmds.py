"""Mixture models and changepoint detection."""

from __future__ import annotations
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("mixture", usage="mixture <col> [--k=3] [--covariance=full]")
def cmd_mixture(session: Session, args: str) -> str:
    """Gaussian Mixture Model clustering / density estimation.

    Options:
      --k=<n>              number of components (default: 3)
      --covariance=<type>  full, tied, diag, spherical (default: full)
      --maxiter=<n>        max EM iterations (default: 200)
      --assign             add component assignment column to data

    Examples:
      mixture income --k=3
      mixture income --k=4 --covariance=diag --assign
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return "scikit-learn required. Install: pip install scikit-learn"

    import numpy as np
    import polars as pl

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: mixture <col> [--k=3]"

    col = ca.positional[0]
    k = int(ca.options.get("k", 3))
    cov_type = ca.options.get("covariance", "full")
    max_iter = int(ca.options.get("maxiter", 200))
    assign = "assign" in ca.flags

    try:
        df = session.require_data()
        if col not in df.columns:
            return f"Column not found: {col}"

        X = df[col].drop_nulls().to_numpy().reshape(-1, 1)
        if len(X) < k * 2:
            return f"Too few observations ({len(X)}) for {k} components."

        gm = GaussianMixture(n_components=k, covariance_type=cov_type,
                             max_iter=max_iter, random_state=42)
        gm.fit(X)

        lines = [f"Gaussian Mixture Model — {col} (k={k}, cov={cov_type})", ""]
        lines.append(f"  Log-likelihood : {gm.lower_bound_:.4f}")
        lines.append(f"  BIC            : {gm.bic(X):.4f}")
        lines.append(f"  AIC            : {gm.aic(X):.4f}")
        lines.append(f"  Converged      : {gm.converged_}")
        lines.append("")
        lines.append(f"  {'Component':<12} {'Weight':>10} {'Mean':>12} {'Std':>12}")
        lines.append("  " + "-" * 50)
        for i in range(k):
            w = gm.weights_[i]
            mu = gm.means_[i, 0]
            if cov_type == "full":
                sigma = float(np.sqrt(gm.covariances_[i, 0, 0]))
            elif cov_type == "tied":
                sigma = float(np.sqrt(gm.covariances_[0, 0]))
            elif cov_type == "diag":
                sigma = float(np.sqrt(gm.covariances_[i, 0]))
            else:
                sigma = float(np.sqrt(gm.covariances_[i]))
            lines.append(f"  {i+1:<12} {w:>10.4f} {mu:>12.4f} {sigma:>12.4f}")

        if assign:
            all_X = df[col].to_numpy().reshape(-1, 1)
            labels = gm.predict(all_X)
            session.df = df.with_columns(
                pl.Series(f"{col}_component", labels.astype(int))
            )
            lines.append(f"\nComponent assignments added as '{col}_component'.")

        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "mixture")


@command("changepoint", usage="changepoint <col> [--model=rbf] [--n=5] [--pen=10]")
def cmd_changepoint(session: Session, args: str) -> str:
    """Change point detection in time series.

    Uses the ruptures library to detect structural breaks in a variable.

    Options:
      --model=<m>   cost model: rbf, l1, l2, normal, ar (default: rbf)
      --n=<k>       number of change points to find (default: 5)
      --pen=<p>     penalty for automatic detection via PELT (overrides --n)
      --algo=<a>    algorithm: pelt, binseg, dynp, window (default: pelt if --pen, else binseg)

    Examples:
      changepoint price --n=3
      changepoint gdp --pen=5 --model=l2
      changepoint returns --algo=binseg --n=4
    """
    try:
        import ruptures as rpt
    except ImportError:
        return "ruptures required. Install: pip install ruptures"

    import numpy as np

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: changepoint <col> [--model=rbf] [--n=5]"

    col = ca.positional[0]
    model = ca.options.get("model", "rbf")
    n_bkps = int(ca.options.get("n", 5))
    pen = ca.options.get("pen")
    algo_name = ca.options.get("algo", "pelt" if pen else "binseg")

    try:
        df = session.require_data()
        if col not in df.columns:
            return f"Column not found: {col}"

        signal = df[col].drop_nulls().to_numpy().astype(float)
        if len(signal) < 10:
            return f"Need at least 10 observations (got {len(signal)})."

        # Select algorithm
        algos = {"pelt": rpt.Pelt, "binseg": rpt.Binseg,
                 "dynp": rpt.Dynp, "window": rpt.Window}
        AlgoClass = algos.get(algo_name.lower(), rpt.Binseg)
        algo = AlgoClass(model=model).fit(signal)

        if pen is not None:
            breakpoints = algo.predict(pen=float(pen))
        else:
            breakpoints = algo.predict(n_bkps=n_bkps)

        # Remove the last element (= length of signal, not a real break)
        breaks = [b for b in breakpoints if b < len(signal)]

        lines = [f"Change Point Detection — {col}", ""]
        lines.append(f"  Algorithm : {algo_name}")
        lines.append(f"  Model     : {model}")
        lines.append(f"  Signal length: {len(signal)}")
        lines.append(f"  Change points found: {len(breaks)}")
        lines.append("")

        if breaks:
            lines.append(f"  {'#':<6} {'Index':>8} {'Segment mean':>14} {'Segment std':>12}")
            lines.append("  " + "-" * 45)
            prev = 0
            for i, bp in enumerate(breaks + [len(signal)], 1):
                seg = signal[prev:bp]
                lines.append(
                    f"  {i:<6} {bp:>8} {seg.mean():>14.4f} {seg.std():>12.4f}"
                )
                prev = bp
        else:
            lines.append("  No change points detected.")

        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "changepoint")
