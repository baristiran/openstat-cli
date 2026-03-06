"""Mediation and moderated-mediation analysis commands."""

from __future__ import annotations

import numpy as np

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


def _bootstrap_indirect(x, m, y, n_boot: int = 1000, seed: int | None = None):
    """Return bootstrap distribution of indirect effect a*b."""
    rng = np.random.default_rng(seed)
    n = len(x)
    ab_boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, mb, yb = x[idx], m[idx], y[idx]
        # a path: m ~ x
        xb_ = np.column_stack([np.ones(n), xb])
        try:
            a = np.linalg.lstsq(xb_, mb, rcond=None)[0][1]
            # b path: y ~ x + m
            xmb = np.column_stack([np.ones(n), xb, mb])
            b = np.linalg.lstsq(xmb, yb, rcond=None)[0][2]
            ab_boots.append(a * b)
        except Exception:
            continue
    return np.array(ab_boots)


@command("mediate", usage="mediate <y> <m> <x> [--boot=1000] [--seed=N]")
def cmd_mediate(session: Session, args: str) -> str:
    """Baron-Kenny mediation analysis with bootstrap CI for indirect effect.

    Tests whether m mediates the effect of x on y.

    Paths:
      a: x → m
      b: m → y (controlling x)
      c: x → y (total)
      c': x → y (direct, controlling m)
      indirect = a × b

    Examples:
      mediate income educ age
      mediate y mediator x --boot=5000 --seed=42
    """
    import polars as pl
    ca = CommandArgs(args)
    if len(ca.positional) < 3:
        return "Usage: mediate <y> <m> <x> [--boot=1000] [--seed=N]"

    y_col, m_col, x_col = ca.positional[0], ca.positional[1], ca.positional[2]
    n_boot = int(ca.options.get("boot", 1000))
    seed = int(ca.options["seed"]) if "seed" in ca.options else getattr(session, "_repro_seed", None)

    try:
        df = session.require_data()
        sub = df.select([y_col, m_col, x_col]).drop_nulls()
        if sub.height < 10:
            return "Need at least 10 complete cases."

        y = sub[y_col].to_numpy().astype(float)
        m = sub[m_col].to_numpy().astype(float)
        x = sub[x_col].to_numpy().astype(float)
        n = len(y)
        ones = np.ones(n)

        def ols(X, y_):
            coef, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
            y_hat = X @ coef
            resid = y_ - y_hat
            sigma2 = np.dot(resid, resid) / max(n - X.shape[1], 1)
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            return coef, se

        # a path: m ~ x
        Xa = np.column_stack([ones, x])
        a_coef, a_se = ols(Xa, m)
        a, se_a = a_coef[1], a_se[1]

        # b & c' paths: y ~ x + m
        Xbc = np.column_stack([ones, x, m])
        bc_coef, bc_se = ols(Xbc, y)
        c_prime, se_cp = bc_coef[1], bc_se[1]
        b, se_b = bc_coef[2], bc_se[2]

        # c path: y ~ x (total)
        Xc = np.column_stack([ones, x])
        c_coef, c_se = ols(Xc, y)
        c, se_c = c_coef[1], c_se[1]

        indirect = a * b

        # Bootstrap CI for indirect
        ab_dist = _bootstrap_indirect(x, m, y, n_boot=n_boot, seed=seed)
        ci_lo = float(np.percentile(ab_dist, 2.5))
        ci_hi = float(np.percentile(ab_dist, 97.5))

        # z-stats (paths a, b, c, c')
        from scipy import stats as _st
        def _p(coef, se): return 2 * _st.t.sf(abs(coef / se), df=n - 2) if se > 0 else float("nan")

        mediated_pct = 100 * abs(indirect / c) if abs(c) > 1e-12 else float("nan")

        lines = [
            f"Mediation Analysis: {y_col} ~ {x_col} → {m_col} → {y_col}",
            f"N = {n}",
            "=" * 56,
            f"  {'Path':<20} {'Coef':>9} {'SE':>9} {'p':>9}",
            "-" * 56,
            f"  {'a  (x->m)':<20} {a:9.4f} {se_a:9.4f} {_p(a,se_a):9.4f}",
            f"  {'b  (m->y|x)':<20} {b:9.4f} {se_b:9.4f} {_p(b,se_b):9.4f}",
            f"  {'c  total (x->y)':<20} {c:9.4f} {se_c:9.4f} {_p(c,se_c):9.4f}",
            "  {:<20} {:9.4f} {:9.4f} {:9.4f}".format("c' direct(x->y|m)", c_prime, se_cp, _p(c_prime, se_cp)),
            "=" * 56,
            f"  Indirect (a×b):     {indirect:9.4f}",
            f"  Bootstrap 95% CI:   [{ci_lo:.4f}, {ci_hi:.4f}]  (B={n_boot})",
            f"  % Mediated:         {mediated_pct:.1f}%" if not np.isnan(mediated_pct) else "  % Mediated:         N/A",
            "",
            "Mediation: " + ("YES — CI excludes 0" if ci_lo * ci_hi > 0 else "NOT significant (CI includes 0)"),
        ]
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "mediate")


@command("modmediate", usage="modmediate <y> <m> <x> <w> [--boot=1000]")
def cmd_modmediate(session: Session, args: str) -> str:
    """Moderated mediation (Hayes PROCESS Model 7 style).

    Tests whether the indirect effect of x on y through m
    is moderated by w (moderator of the a-path: x->m).

    Index of Moderated Mediation (IMM) with bootstrap CI.

    Examples:
      modmediate outcome mediator predictor moderator --boot=2000
    """
    import polars as pl
    ca = CommandArgs(args)
    if len(ca.positional) < 4:
        return "Usage: modmediate <y> <m> <x> <w> [--boot=1000] [--seed=N]"

    y_col, m_col, x_col, w_col = (ca.positional[i] for i in range(4))
    n_boot = int(ca.options.get("boot", 1000))
    seed = int(ca.options["seed"]) if "seed" in ca.options else getattr(session, "_repro_seed", None)

    try:
        df = session.require_data()
        sub = df.select([y_col, m_col, x_col, w_col]).drop_nulls()
        if sub.height < 20:
            return "Need at least 20 complete cases."

        y = sub[y_col].to_numpy().astype(float)
        m = sub[m_col].to_numpy().astype(float)
        x = sub[x_col].to_numpy().astype(float)
        w = sub[w_col].to_numpy().astype(float)
        n = len(y)
        ones = np.ones(n)

        # Standardise w for interaction
        w_c = w - w.mean()
        xw = x * w_c

        def ols(X, y_):
            coef, _, _, _ = np.linalg.lstsq(X, y_, rcond=None)
            return coef

        # a-path moderated: m ~ x + w + x*w
        Xa = np.column_stack([ones, x, w_c, xw])
        a_coef = ols(Xa, m)
        a1, a3 = a_coef[1], a_coef[3]  # a1=x coef, a3=interaction

        # b-path: y ~ x + m
        Xb = np.column_stack([ones, x, m])
        b_coef = ols(Xb, y)
        b = b_coef[2]

        # Conditional indirect at w = mean ± 1SD
        w_sd = w_c.std()
        w_vals = {"Low (−1SD)": -w_sd, "Mean": 0.0, "High (+1SD)": w_sd}

        def _boot_imm(rng):
            idx = rng.integers(0, n, size=n)
            xb, mb, yb, wb = x[idx], m[idx], y[idx], w_c[idx]
            xwb = xb * wb
            Xab = np.column_stack([np.ones(n), xb, wb, xwb])
            try:
                ac = ols(Xab, mb)
                Xbb = np.column_stack([np.ones(n), xb, mb])
                bc = ols(Xbb, yb)
                return ac[3] * bc[2]  # IMM = a3 * b
            except Exception:
                return np.nan

        rng = np.random.default_rng(seed)
        imm_boots = np.array([_boot_imm(rng) for _ in range(n_boot)])
        imm_boots = imm_boots[~np.isnan(imm_boots)]
        imm = a3 * b
        ci_lo = float(np.percentile(imm_boots, 2.5))
        ci_hi = float(np.percentile(imm_boots, 97.5))

        lines = [
            f"Moderated Mediation: {y_col} ~ {x_col}→{m_col}→{y_col}, moderated by {w_col}",
            f"N = {n}",
            "=" * 60,
            "Conditional Indirect Effects (a×b at levels of moderator):",
            f"  {'Level':<15} {'a+a3*w':>10} {'Indirect':>10}",
            "-" * 60,
        ]
        for label, wv in w_vals.items():
            cond_a = a1 + a3 * wv
            indirect = cond_a * b
            lines.append(f"  {label:<15} {cond_a:10.4f} {indirect:10.4f}")

        lines += [
            "=" * 60,
            f"  Index of Moderated Mediation (a3×b): {imm:.4f}",
            f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (B={n_boot})",
            "",
            "Moderated mediation: " + (
                "YES — IMM CI excludes 0" if ci_lo * ci_hi > 0
                else "NOT significant (CI includes 0)"
            ),
        ]
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "modmediate")
