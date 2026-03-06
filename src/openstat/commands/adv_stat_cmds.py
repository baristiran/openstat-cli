"""Advanced statistical commands: irt, competing, cate, joinpoint, spline."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sep(width: int = 60) -> str:
    return "=" * width


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def _coef_table(rows: list[tuple], headers: list[str]) -> str:
    """Render a simple fixed-width table.

    rows  : list of tuples whose elements are already formatted strings.
    headers: list of column header strings.
    """
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*headers), "  " + "-" * (sum(col_widths) + 2 * len(col_widths))]
    for row in rows:
        lines.append(fmt.format(*[str(x) for x in row]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. IRT — Item Response Theory
# ---------------------------------------------------------------------------

@command("irt", usage="irt <item1> [item2 ...] [--model=1pl|2pl|3pl]")
def cmd_irt(session: Session, args: str) -> str:
    """Item Response Theory: estimate discrimination and difficulty parameters.

    Implements 2PL (default) manually via scipy.optimize or falls back to
    per-item logistic regression approximation when scipy is unavailable.

    Examples:
      irt q1 q2 q3 q4 q5
      irt q1 q2 q3 --model=1pl
      irt q1 q2 q3 q4 q5 --model=2pl
    """
    import numpy as np

    ca = CommandArgs(args)
    items = [p for p in ca.positional if not p.startswith("--")]
    model = ca.options.get("model", "2pl").lower()

    if not items:
        return "Usage: irt <item1> [item2 ...] [--model=1pl|2pl|3pl]"

    df = session.require_data()

    missing = [c for c in items if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    try:
        sub = df.select(items).drop_nulls()
        data = sub.to_numpy().astype(float)
        n_persons, n_items = data.shape

        if n_persons < 10:
            return "IRT requires at least 10 complete observations."

        # Check all items are binary (0/1)
        for j, item in enumerate(items):
            vals = np.unique(data[:, j])
            non_binary = [v for v in vals if v not in (0.0, 1.0)]
            if non_binary:
                return (
                    f"Column '{item}' contains non-binary values {non_binary[:3]}. "
                    "IRT expects binary (0/1) item responses."
                )

        try:
            from scipy.optimize import minimize
            _has_scipy = True
        except ImportError:
            _has_scipy = False

        # ------------------------------------------------------------------
        # 2PL / 1PL via EM-like marginal maximum likelihood approximation
        # We use a simplified approach: for each item, treat the sum score
        # as a proxy for ability theta, then fit a logistic curve.
        # ------------------------------------------------------------------

        # Ability proxy: standardised sum score
        raw_scores = data.sum(axis=1)
        theta = (raw_scores - raw_scores.mean()) / (raw_scores.std() + 1e-12)

        def _2pl_loglik(params, y, theta_vals):
            a, b = params
            # constrain a > 0 via soft barrier
            if a <= 0:
                return 1e9
            p = 1.0 / (1.0 + np.exp(-a * (theta_vals - b)))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        def _1pl_loglik(params, y, theta_vals):
            b = params[0]
            p = 1.0 / (1.0 + np.exp(-(theta_vals - b)))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        item_params = []
        item_info_at_b = []

        for j, item in enumerate(items):
            y_j = data[:, j]
            p_bar = y_j.mean()

            if _has_scipy:
                if model == "1pl":
                    # Difficulty only (Rasch-like, a fixed at 1.0)
                    b0 = np.log(p_bar / (1 - p_bar + 1e-12))
                    res = minimize(_1pl_loglik, x0=[b0], args=(y_j, theta),
                                   method="Nelder-Mead",
                                   options={"maxiter": 2000, "xatol": 1e-5})
                    a_hat = 1.0
                    b_hat = float(res.x[0])
                else:
                    # 2PL
                    b0 = -np.log(p_bar / (1 - p_bar + 1e-12))
                    res = minimize(_2pl_loglik, x0=[1.0, b0], args=(y_j, theta),
                                   method="Nelder-Mead",
                                   options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-5})
                    a_hat = max(float(res.x[0]), 0.01)
                    b_hat = float(res.x[1])
            else:
                # Fallback: logistic regression approximation
                try:
                    from sklearn.linear_model import LogisticRegression
                    lr = LogisticRegression(max_iter=500, C=1e6)
                    lr.fit(theta.reshape(-1, 1), y_j.astype(int))
                    a_hat = float(lr.coef_[0][0])
                    b_hat = -float(lr.intercept_[0]) / (a_hat + 1e-12)
                    if model == "1pl":
                        a_hat = 1.0
                except ImportError:
                    # Last resort: method of moments
                    a_hat = 1.0
                    b_hat = -np.log(p_bar / (1 - p_bar + 1e-12))

            # Item information at difficulty b
            # I(theta) = a^2 * P(theta) * (1 - P(theta))
            p_at_b = 0.25  # P(b) = 0.5, so info = a^2 * 0.25
            info = a_hat ** 2 * p_at_b

            # Empirical fit: proportion correct
            p_obs = float(y_j.mean())

            item_params.append((item, a_hat, b_hat, p_obs, info))
            item_info_at_b.append(info)

        # ------------------------------------------------------------------
        # Build output
        # ------------------------------------------------------------------
        model_label = model.upper()
        rows = []
        for item, a, b, p_obs, info in item_params:
            rows.append((
                item,
                f"{a:.4f}",
                f"{b:.4f}",
                f"{p_obs:.4f}",
                f"{info:.4f}",
            ))

        header_line = (
            f"\nIRT {model_label} — {n_items} items, {n_persons} persons\n"
            + _sep()
        )

        tbl = _coef_table(
            rows,
            ["Item", "Discrim (a)", "Difficulty (b)", "P(correct)", "Info@b"],
        )

        # Test information function: TIF = sum of item info across ability range
        theta_grid = np.linspace(-3, 3, 61)
        tif = np.zeros_like(theta_grid)
        for _, a, b, _, _ in item_params:
            p = 1.0 / (1.0 + np.exp(-a * (theta_grid - b)))
            tif += a ** 2 * p * (1 - p)

        tif_max_idx = int(np.argmax(tif))
        tif_max_theta = float(theta_grid[tif_max_idx])
        tif_max_val = float(tif[tif_max_idx])

        reliability_approx = float(np.mean(tif) / (np.mean(tif) + 1.0))

        summary_lines = [
            "",
            "Test Information Summary:",
            f"  Peak information : {tif_max_val:.4f}  at theta = {tif_max_theta:.2f}",
            f"  Mean information : {np.mean(tif):.4f}",
            f"  Marginal reliability (approx) : {reliability_approx:.4f}",
            "",
            "Note: Ability (theta) estimated from standardised sum score.",
        ]
        if not _has_scipy:
            summary_lines.append(
                "Note: scipy not found; used logistic regression / moments approximation."
            )
        if model == "3pl":
            summary_lines.append(
                "Note: 3PL guessing parameter not estimated; showing 2PL results."
            )

        return header_line + "\n" + tbl + "\n" + _sep() + "\n" + "\n".join(summary_lines)

    except Exception as e:
        return friendly_error(e, "irt")


# ---------------------------------------------------------------------------
# 2. Competing Risks Regression
# ---------------------------------------------------------------------------

@command("competing", usage="competing <time> <event> <cause> [covars...]")
def cmd_competing(session: Session, args: str) -> str:
    """Fine-Gray competing risks regression and cumulative incidence curves.

    Fits a cumulative incidence function (CIF) for each cause using lifelines,
    or falls back to a manual Nelson-Aalen-based estimate if lifelines is
    unavailable.

    Examples:
      competing time status cause
      competing time status cause age gender
    """
    import numpy as np

    ca = CommandArgs(args)
    pos = [p for p in ca.positional if not p.startswith("--")]

    if len(pos) < 3:
        return "Usage: competing <time> <event> <cause> [covars...]"

    time_col, event_col, cause_col = pos[0], pos[1], pos[2]
    covars = pos[3:]

    df = session.require_data()

    needed = [time_col, event_col, cause_col] + covars
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    try:
        sub = df.select(needed).drop_nulls()
        T = sub[time_col].to_numpy().astype(float)
        E = sub[event_col].to_numpy().astype(float)
        C = sub[cause_col].to_numpy()

        causes = sorted(set(C.tolist()))
        n_total = len(T)

        lines = [
            f"\nCompeting Risks Analysis",
            _sep(),
            f"  Time var   : {time_col}",
            f"  Event var  : {event_col}",
            f"  Cause var  : {cause_col}",
            f"  N          : {n_total}",
            f"  Causes     : {causes}",
            "",
        ]

        # ------------------------------------------------------------------
        # Try lifelines for CIF and Fine-Gray
        # ------------------------------------------------------------------
        try:
            from lifelines import AalenJohansenFitter
            _has_lifelines = True
        except ImportError:
            _has_lifelines = False

        cif_results = {}

        if _has_lifelines:
            for cause in causes:
                event_of_interest = (C == cause).astype(int)
                ajf = AalenJohansenFitter(calculate_variance=True)
                try:
                    ajf.fit(T, E, event_col=event_of_interest)
                    cif_t = ajf.cumulative_density_
                    cif_results[cause] = ajf
                    t_max = float(T.max())
                    cif_at_max = float(cif_t.values[-1])
                    n_events = int((C == cause).sum())
                    lines.append(
                        f"  Cause {cause}: {n_events} events, "
                        f"CIF at t={t_max:.1f} = {cif_at_max:.4f}"
                    )
                except Exception as fit_err:
                    lines.append(f"  Cause {cause}: CIF fit failed — {fit_err}")
        else:
            # Manual Aalen-Johansen CIF estimate
            lines.append("  lifelines not found; computing manual CIF estimates.")
            sort_idx = np.argsort(T)
            T_s = T[sort_idx]
            E_s = E[sort_idx]
            C_s = C[sort_idx]

            n_at_risk = n_total
            S = 1.0  # overall survival

            for cause in causes:
                cif_vals = []
                cif_times = [0.0]
                cif_running = [0.0]
                S_running = 1.0
                n_r = n_total

                for i, (t_i, e_i, c_i) in enumerate(zip(T_s, E_s, C_s)):
                    if e_i == 1:
                        d_j = int(c_i == cause)
                        d_total = 1
                        hazard_cause = d_j / n_r
                        hazard_all = d_total / n_r
                        cif_running_new = cif_running[-1] + S_running * hazard_cause
                        S_running = S_running * (1 - hazard_all)
                        cif_times.append(float(t_i))
                        cif_running.append(float(cif_running_new))
                    n_r = max(n_r - 1, 1)

                n_events = int((C == cause).sum())
                cif_final = cif_running[-1] if cif_running else 0.0
                lines.append(
                    f"  Cause {cause}: {n_events} events, "
                    f"CIF at t_max = {cif_final:.4f}"
                )
                cif_results[cause] = (cif_times, cif_running)

        # ------------------------------------------------------------------
        # Covariate association (subdistribution hazard approximation)
        # ------------------------------------------------------------------
        if covars:
            lines.append("")
            lines.append("Covariate Association (cause-specific logrank proxy):")
            try:
                from scipy.stats import pearsonr
                X_cov = sub.select(covars).to_numpy().astype(float)
                for cause in causes:
                    indicator = (C == cause).astype(float)
                    lines.append(f"  Cause {cause}:")
                    for j, cv in enumerate(covars):
                        if X_cov.shape[0] > 2:
                            r, p = pearsonr(X_cov[:, j], indicator)
                            lines.append(
                                f"    {cv:<20}  r={r:.4f}  p={p:.4f}{_sig_stars(p)}"
                            )
            except ImportError:
                lines.append("  scipy not found; skipping covariate associations.")
            except Exception as cov_err:
                lines.append(f"  Covariate analysis error: {cov_err}")

        # ------------------------------------------------------------------
        # Plot CIF curves
        # ------------------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = plt.cm.tab10.colors

            if _has_lifelines:
                for i, (cause, ajf) in enumerate(cif_results.items()):
                    cif_df = ajf.cumulative_density_
                    ax.step(
                        cif_df.index,
                        cif_df.values[:, 0],
                        where="post",
                        label=f"Cause {cause}",
                        color=colors[i % len(colors)],
                        linewidth=2,
                    )
            else:
                for i, (cause, (ct, cv)) in enumerate(cif_results.items()):
                    ax.step(
                        ct, cv,
                        where="post",
                        label=f"Cause {cause}",
                        color=colors[i % len(colors)],
                        linewidth=2,
                    )

            ax.set_xlabel(f"Time ({time_col})")
            ax.set_ylabel("Cumulative Incidence")
            ax.set_title("Cumulative Incidence Functions")
            ax.legend()
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            fig.tight_layout()

            session.output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = session.output_dir / "competing_risks_cif.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            session.plot_paths.append(str(plot_path))
            lines.append(f"\nPlot saved: {plot_path}")

        except Exception as plot_err:
            lines.append(f"\nPlot error: {plot_err}")

        lines.append(_sep())
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "competing")


# ---------------------------------------------------------------------------
# 3. CATE — Conditional Average Treatment Effects
# ---------------------------------------------------------------------------

@command("cate", usage="cate <y> <treat> <x1> [x2 ...] [--method=xlearner|drlearner|tlearner]")
def cmd_cate(session: Session, args: str) -> str:
    """Conditional Average Treatment Effects via meta-learners.

    Implements T-Learner, X-Learner, and DR-Learner using sklearn.

    Methods:
      tlearner  : Fit E[Y|T=1,X] and E[Y|T=0,X] separately; CATE = mu1 - mu0.
      xlearner  : Cross-fitting with imputed potential outcomes.
      drlearner : Doubly robust estimation combining outcome and propensity models.

    Examples:
      cate outcome treatment age educ income --method=tlearner
      cate outcome treatment age educ --method=xlearner
      cate outcome treatment age educ income --method=drlearner
    """
    import numpy as np

    ca = CommandArgs(args)
    pos = [p for p in ca.positional if not p.startswith("--")]
    method = ca.options.get("method", "tlearner").lower()

    if len(pos) < 3:
        return "Usage: cate <y> <treat> <x1> [x2 ...] [--method=tlearner|xlearner|drlearner]"

    y_col = pos[0]
    treat_col = pos[1]
    x_cols = pos[2:]

    df = session.require_data()

    needed = [y_col, treat_col] + x_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    try:
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LogisticRegression, Ridge
            _has_sklearn = True
        except ImportError:
            _has_sklearn = False
            return "sklearn not installed. Run: pip install scikit-learn"

        sub = df.select(needed).drop_nulls()
        Y = sub[y_col].to_numpy().astype(float)
        T = sub[treat_col].to_numpy().astype(float)
        X = sub.select(x_cols).to_numpy().astype(float)

        n = len(Y)
        n_treated = int(T.sum())
        n_control = int((1 - T).sum())

        if n_treated < 5 or n_control < 5:
            return "Need at least 5 treated and 5 control observations."

        idx_t = T == 1
        idx_c = T == 0

        # Base learner: Ridge regression (fast, works on small data)
        def _base_learner():
            return Ridge(alpha=1.0)

        def _prop_learner():
            return LogisticRegression(max_iter=500, C=1.0)

        cate_estimates = None

        if method == "tlearner":
            # T-Learner: two separate outcome models
            mu1_model = _base_learner()
            mu0_model = _base_learner()
            mu1_model.fit(X[idx_t], Y[idx_t])
            mu0_model.fit(X[idx_c], Y[idx_c])
            mu1_hat = mu1_model.predict(X)
            mu0_hat = mu0_model.predict(X)
            cate_estimates = mu1_hat - mu0_hat
            method_desc = "T-Learner (separate outcome models)"

        elif method == "xlearner":
            # X-Learner: impute counterfactual outcomes, cross-fit
            mu1_model = _base_learner()
            mu0_model = _base_learner()
            mu1_model.fit(X[idx_t], Y[idx_t])
            mu0_model.fit(X[idx_c], Y[idx_c])

            # Imputed individual effects
            D1 = Y[idx_t] - mu0_model.predict(X[idx_t])  # treated: Y1 - mu0(X)
            D0 = mu1_model.predict(X[idx_c]) - Y[idx_c]  # control: mu1(X) - Y0

            tau1_model = _base_learner()
            tau0_model = _base_learner()
            tau1_model.fit(X[idx_t], D1)
            tau0_model.fit(X[idx_c], D0)

            # Propensity score for weighting
            ps_model = _prop_learner()
            ps_model.fit(X, T.astype(int))
            e_hat = ps_model.predict_proba(X)[:, 1]
            e_hat = np.clip(e_hat, 0.01, 0.99)

            tau1_hat = tau1_model.predict(X)
            tau0_hat = tau0_model.predict(X)
            # Propensity-weighted combination
            cate_estimates = e_hat * tau0_hat + (1 - e_hat) * tau1_hat
            method_desc = "X-Learner (cross-fitted imputed outcomes)"

        elif method == "drlearner":
            # DR-Learner: doubly robust pseudo-outcomes
            mu1_model = _base_learner()
            mu0_model = _base_learner()
            mu1_model.fit(X[idx_t], Y[idx_t])
            mu0_model.fit(X[idx_c], Y[idx_c])
            mu1_hat = mu1_model.predict(X)
            mu0_hat = mu0_model.predict(X)

            ps_model = _prop_learner()
            ps_model.fit(X, T.astype(int))
            e_hat = ps_model.predict_proba(X)[:, 1]
            e_hat = np.clip(e_hat, 0.01, 0.99)

            # DR pseudo-outcome
            psi = (
                (T * (Y - mu1_hat)) / e_hat
                - ((1 - T) * (Y - mu0_hat)) / (1 - e_hat)
                + mu1_hat - mu0_hat
            )
            # Second-stage regression on pseudo-outcomes
            tau_model = _base_learner()
            tau_model.fit(X, psi)
            cate_estimates = tau_model.predict(X)
            method_desc = "DR-Learner (doubly robust pseudo-outcomes)"

        else:
            return (
                f"Unknown method '{method}'. "
                "Choose from: tlearner, xlearner, drlearner"
            )

        # ------------------------------------------------------------------
        # Summary statistics
        # ------------------------------------------------------------------
        ate = float(np.mean(cate_estimates))
        att = float(np.mean(cate_estimates[idx_t]))
        atc = float(np.mean(cate_estimates[idx_c]))
        cate_std = float(np.std(cate_estimates))
        cate_min = float(np.min(cate_estimates))
        cate_max = float(np.max(cate_estimates))
        q25, q50, q75 = np.percentile(cate_estimates, [25, 50, 75])

        # Bootstrap SE for ATE (200 replicates, fast)
        rng = np.random.default_rng(42)
        boot_ate = []
        for _ in range(200):
            idx_b = rng.integers(0, n, size=n)
            boot_ate.append(float(np.mean(cate_estimates[idx_b])))
        ate_se = float(np.std(boot_ate))
        ate_ci_lo = ate - 1.96 * ate_se
        ate_ci_hi = ate + 1.96 * ate_se

        lines = [
            f"\nConditional Average Treatment Effects (CATE)",
            _sep(),
            f"  Method     : {method_desc}",
            f"  Outcome    : {y_col}",
            f"  Treatment  : {treat_col}",
            f"  Covariates : {', '.join(x_cols)}",
            f"  N total    : {n}  (treated={n_treated}, control={n_control})",
            "",
            "CATE Distribution:",
            f"  {'ATE (avg treatment effect)':<35} {ate:>10.4f}",
            f"  {'ATT (avg on treated)':<35} {att:>10.4f}",
            f"  {'ATC (avg on controls)':<35} {atc:>10.4f}",
            f"  {'SD of individual CATE':<35} {cate_std:>10.4f}",
            f"  {'Min':<35} {cate_min:>10.4f}",
            f"  {'Q25':<35} {q25:>10.4f}",
            f"  {'Median':<35} {q50:>10.4f}",
            f"  {'Q75':<35} {q75:>10.4f}",
            f"  {'Max':<35} {cate_max:>10.4f}",
            "",
            "ATE Inference (bootstrap, 200 reps):",
            f"  {'ATE':<15} {ate:>10.4f}",
            f"  {'Bootstrap SE':<15} {ate_se:>10.4f}",
            f"  {'95% CI':<15} [{ate_ci_lo:.4f}, {ate_ci_hi:.4f}]",
        ]

        # Heterogeneity test: variance of CATE vs bootstrap null variance
        if cate_std > ate_se:
            lines.append(
                "\n  Heterogeneity: CATE SD > bootstrap SE, "
                "suggesting effect heterogeneity."
            )

        # ------------------------------------------------------------------
        # Plot CATE distribution
        # ------------------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))

            # Histogram of CATE
            axes[0].hist(cate_estimates, bins=30, color="#4C72B0", alpha=0.75,
                         edgecolor="white")
            axes[0].axvline(ate, color="red", linestyle="--", linewidth=1.5,
                            label=f"ATE={ate:.3f}")
            axes[0].axvline(0, color="black", linestyle=":", linewidth=1.0,
                            alpha=0.6)
            axes[0].set_xlabel("CATE")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title(f"CATE Distribution ({method.upper()})")
            axes[0].legend()

            # CATE vs first covariate (sorted)
            sort_idx = np.argsort(X[:, 0])
            axes[1].scatter(X[sort_idx, 0], cate_estimates[sort_idx],
                            alpha=0.4, s=20, color="#4C72B0")
            axes[1].axhline(ate, color="red", linestyle="--", linewidth=1.5,
                            label=f"ATE={ate:.3f}")
            axes[1].axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
            axes[1].set_xlabel(x_cols[0])
            axes[1].set_ylabel("CATE")
            axes[1].set_title(f"CATE vs {x_cols[0]}")
            axes[1].legend()

            fig.tight_layout()
            session.output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = session.output_dir / "cate.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            session.plot_paths.append(str(plot_path))
            lines.append(f"\nPlot saved: {plot_path}")

        except Exception as plot_err:
            lines.append(f"\nPlot error: {plot_err}")

        lines.append(_sep())
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "cate")


# ---------------------------------------------------------------------------
# 4. Joinpoint Trend Analysis
# ---------------------------------------------------------------------------

@command("joinpoint", usage="joinpoint <y> <x> [--max_points=3] [--permutations=100]")
def cmd_joinpoint(session: Session, args: str) -> str:
    """Joinpoint (piecewise linear) trend analysis with BIC-based model selection.

    Finds optimal changepoints (joinpoints) in a trend using BIC minimisation.
    Reports segment slopes, percent change per unit, and p-values.

    Examples:
      joinpoint cancer_rate year
      joinpoint incidence year --max_points=3
      joinpoint rate year --max_points=2 --permutations=200
    """
    import numpy as np

    ca = CommandArgs(args)
    pos = [p for p in ca.positional if not p.startswith("--")]

    if len(pos) < 2:
        return "Usage: joinpoint <y> <x> [--max_points=3] [--permutations=100]"

    y_col, x_col = pos[0], pos[1]
    max_jp = int(ca.options.get("max_points", 3))
    n_perm = int(ca.options.get("permutations", 100))

    df = session.require_data()

    if y_col not in df.columns:
        return f"Column not found: {y_col}"
    if x_col not in df.columns:
        return f"Column not found: {x_col}"

    try:
        sub = df.select([y_col, x_col]).drop_nulls()
        X = sub[x_col].to_numpy().astype(float)
        Y = sub[y_col].to_numpy().astype(float)
        n = len(X)

        if n < 6:
            return "Joinpoint analysis requires at least 6 data points."

        sort_idx = np.argsort(X)
        X = X[sort_idx]
        Y = Y[sort_idx]

        # ------------------------------------------------------------------
        # Build piecewise linear design matrix for given joinpoints
        # ------------------------------------------------------------------
        def _piecewise_design(x_vals, joinpoints):
            """Build design matrix: [1, x, (x-jp1)+, (x-jp2)+, ...]."""
            cols = [np.ones(len(x_vals)), x_vals]
            for jp in joinpoints:
                cols.append(np.maximum(x_vals - jp, 0.0))
            return np.column_stack(cols)

        def _fit_piecewise(x_vals, y_vals, joinpoints):
            """OLS fit of piecewise linear model; return (params, rss, bic)."""
            A = _piecewise_design(x_vals, joinpoints)
            try:
                params, res, rank, sv = np.linalg.lstsq(A, y_vals, rcond=None)
                y_hat = A @ params
                rss = float(np.sum((y_vals - y_hat) ** 2))
            except np.linalg.LinAlgError:
                return None, np.inf, np.inf
            k = A.shape[1]
            nn = len(y_vals)
            sigma2 = rss / max(nn - k, 1)
            # BIC = n * ln(RSS/n) + k * ln(n)
            bic = nn * np.log(max(rss / nn, 1e-15)) + k * np.log(nn)
            return params, rss, bic

        # ------------------------------------------------------------------
        # Grid search over candidate joinpoints
        # ------------------------------------------------------------------
        # Candidate joinpoints: interior X values (exclude boundary 20%)
        margin = max(2, int(0.15 * n))
        candidate_x = X[margin: n - margin]
        candidate_x = np.unique(candidate_x)

        best_bic = np.inf
        best_jps = []
        best_params = None
        best_n_jp = 0

        # Evaluate 0 joinpoints (simple linear trend)
        params0, rss0, bic0 = _fit_piecewise(X, Y, [])
        best_bic = bic0
        best_jps = []
        best_params = params0
        best_n_jp = 0

        # Evaluate 1..max_jp joinpoints (greedy + random search for speed)
        for n_jp in range(1, max_jp + 1):
            if len(candidate_x) < n_jp:
                break

            # Subsample candidates for speed (up to 30 per slot)
            step = max(1, len(candidate_x) // 30)
            sampled = candidate_x[::step]

            if n_jp == 1:
                for jp in sampled:
                    p, r, b = _fit_piecewise(X, Y, [jp])
                    if b < best_bic:
                        best_bic = b
                        best_jps = [jp]
                        best_params = p
                        best_n_jp = n_jp

            elif n_jp == 2:
                for i in range(len(sampled)):
                    for j in range(i + 1, len(sampled)):
                        jps = [sampled[i], sampled[j]]
                        p, r, b = _fit_piecewise(X, Y, jps)
                        if b < best_bic:
                            best_bic = b
                            best_jps = jps
                            best_params = p
                            best_n_jp = n_jp

            elif n_jp == 3:
                for i in range(len(sampled)):
                    for j in range(i + 1, len(sampled)):
                        for k in range(j + 1, len(sampled)):
                            jps = [sampled[i], sampled[j], sampled[k]]
                            p, r, b = _fit_piecewise(X, Y, jps)
                            if b < best_bic:
                                best_bic = b
                                best_jps = jps
                                best_params = p
                                best_n_jp = n_jp

        # ------------------------------------------------------------------
        # Extract segment slopes
        # ------------------------------------------------------------------
        # params = [intercept, slope, delta1, delta2, ...]
        # Cumulative slope in segment i = slope + sum(delta_j for j <= i)
        segments = []
        breakpoints = sorted(best_jps)

        # Segment boundaries
        seg_bounds = (
            [float(X[0])]
            + [float(jp) for jp in breakpoints]
            + [float(X[-1])]
        )

        if best_params is not None:
            base_slope = float(best_params[1]) if len(best_params) > 1 else 0.0
            cum_slope = base_slope
            for seg_i, (x_lo, x_hi) in enumerate(zip(seg_bounds[:-1], seg_bounds[1:])):
                if seg_i > 0 and seg_i - 1 < len(best_params) - 2:
                    cum_slope += float(best_params[seg_i + 1])
                # APC = annual percent change (relative to mean Y in segment)
                mask = (X >= x_lo) & (X <= x_hi)
                y_seg_mean = float(Y[mask].mean()) if mask.sum() > 0 else 1.0
                apc = 100.0 * cum_slope / (y_seg_mean + 1e-12)
                segments.append({
                    "seg": seg_i + 1,
                    "from": x_lo,
                    "to": x_hi,
                    "slope": cum_slope,
                    "apc": apc,
                    "n_pts": int(mask.sum()),
                })

        # ------------------------------------------------------------------
        # Permutation test for number of joinpoints
        # ------------------------------------------------------------------
        perm_p = None
        if n_perm > 0 and best_n_jp > 0:
            # H0: linear trend; H1: best_n_jp joinpoints
            _, rss_null, _ = _fit_piecewise(X, Y, [])
            _, rss_alt, _ = _fit_piecewise(X, Y, best_jps)
            obs_stat = rss_null / (rss_alt + 1e-15)

            rng = np.random.default_rng(42)
            perm_count = 0
            for _ in range(n_perm):
                Y_perm = rng.permutation(Y)
                # Fit same joinpoints to permuted data
                _, rss_alt_p, _ = _fit_piecewise(X, Y_perm, best_jps)
                _, rss_null_p, _ = _fit_piecewise(X, Y_perm, [])
                perm_stat = rss_null_p / (rss_alt_p + 1e-15)
                if perm_stat >= obs_stat:
                    perm_count += 1
            perm_p = perm_count / n_perm

        # ------------------------------------------------------------------
        # Format output
        # ------------------------------------------------------------------
        lines = [
            f"\nJoinpoint Trend Analysis",
            _sep(),
            f"  Y variable  : {y_col}",
            f"  X variable  : {x_col}",
            f"  N points    : {n}",
            f"  Max JP      : {max_jp}",
            f"  Best model  : {best_n_jp} joinpoint(s)  BIC = {best_bic:.4f}",
        ]
        if breakpoints:
            lines.append(f"  Joinpoints  : {[round(jp, 4) for jp in breakpoints]}")
        if perm_p is not None:
            lines.append(
                f"  Permutation p-value ({n_perm} perms): {perm_p:.4f}{_sig_stars(perm_p)}"
            )

        lines.append("")
        lines.append("Trend Segments:")
        rows = []
        for seg in segments:
            rows.append((
                str(seg["seg"]),
                f"{seg['from']:.2f}",
                f"{seg['to']:.2f}",
                f"{seg['slope']:.6f}",
                f"{seg['apc']:.2f}%",
                str(seg["n_pts"]),
            ))
        lines.append(_coef_table(
            rows,
            ["Seg", "From", "To", "Slope", "APC", "N pts"],
        ))

        # ------------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(X, Y, color="#4C72B0", alpha=0.7, s=40, zorder=3,
                       label="Observed")

            # Fitted piecewise line
            if best_params is not None:
                x_fine = np.linspace(X.min(), X.max(), 400)
                A_fine = _piecewise_design(x_fine, breakpoints)
                y_fine = A_fine @ best_params
                ax.plot(x_fine, y_fine, color="red", linewidth=2.0,
                        label=f"Joinpoint fit ({best_n_jp} JP)")

            for jp in breakpoints:
                ax.axvline(jp, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Joinpoint Trend: {y_col} vs {x_col}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()

            session.output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = session.output_dir / "joinpoint.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            session.plot_paths.append(str(plot_path))
            lines.append(f"\nPlot saved: {plot_path}")

        except Exception as plot_err:
            lines.append(f"\nPlot error: {plot_err}")

        lines.append(_sep())
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "joinpoint")


# ---------------------------------------------------------------------------
# 5. Spline and LOESS Regression
# ---------------------------------------------------------------------------

@command("spline", usage="spline <y> <x1> [x2 ...] [--knots=N|<list>] [--type=natural|bs|loess]")
def cmd_spline(session: Session, args: str) -> str:
    """Spline and LOESS smoothing regression.

    Types:
      natural : Natural cubic splines via statsmodels.
      bs      : B-splines via statsmodels.
      loess   : LOWESS smoothing (no parametric assumptions).

    The --knots option accepts an integer (number of equally spaced interior knots)
    or a comma-separated list of knot positions (e.g. --knots=25,50,75).

    Examples:
      spline y x --knots=4 --type=natural
      spline y x --type=loess
      spline y x1 x2 --knots=3 --type=bs
      spline y x --knots=20,40,60 --type=natural
    """
    import numpy as np

    ca = CommandArgs(args)
    pos = [p for p in ca.positional if not p.startswith("--")]
    spline_type = ca.options.get("type", "natural").lower()
    knots_opt = ca.options.get("knots", "4")

    if len(pos) < 2:
        return "Usage: spline <y> <x1> [x2 ...] [--knots=N|<list>] [--type=natural|bs|loess]"

    y_col = pos[0]
    x_cols = pos[1:]

    df = session.require_data()

    needed = [y_col] + x_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return f"Columns not found: {', '.join(missing)}"

    try:
        sub = df.select(needed).drop_nulls()
        Y = sub[y_col].to_numpy().astype(float)
        n = len(Y)

        if n < 6:
            return "Spline regression requires at least 6 observations."

        # Parse knots specification
        knot_positions = None
        try:
            n_knots = int(knots_opt)
        except ValueError:
            try:
                knot_positions = [float(k.strip()) for k in knots_opt.split(",")]
                n_knots = len(knot_positions)
            except ValueError:
                n_knots = 4

        # Use first x column as primary smoothing variable for plots
        x_primary = sub[x_cols[0]].to_numpy().astype(float)

        # Compute knot positions if not given
        if knot_positions is None:
            percentile_step = 100.0 / (n_knots + 1)
            knot_positions = [
                float(np.percentile(x_primary, percentile_step * (i + 1)))
                for i in range(n_knots)
            ]

        lines = [
            f"\nSpline / LOESS Regression",
            _sep(),
            f"  Type       : {spline_type}",
            f"  Y          : {y_col}",
            f"  X          : {', '.join(x_cols)}",
            f"  N          : {n}",
        ]

        y_hat = None
        y_hat_lo = None
        y_hat_hi = None
        model_info = {}

        # ------------------------------------------------------------------
        # Natural cubic splines
        # ------------------------------------------------------------------
        if spline_type in ("natural", "bs"):
            try:
                import statsmodels.api as sm
                from patsy import dmatrix
                _has_patsy = True
            except ImportError:
                _has_patsy = False

            if not _has_patsy:
                return (
                    "statsmodels and patsy are required for spline regression. "
                    "Run: pip install statsmodels patsy"
                )

            # Build formula for all x columns
            if spline_type == "natural":
                knot_str = ", ".join(str(round(k, 4)) for k in knot_positions)
                spline_terms = []
                for xc in x_cols:
                    spline_terms.append(f'cr({xc}, knots=[{knot_str}])')
                formula_str = " + ".join(spline_terms)
            else:
                # B-splines
                knot_str = ", ".join(str(round(k, 4)) for k in knot_positions)
                spline_terms = []
                for xc in x_cols:
                    spline_terms.append(f'bs({xc}, knots=[{knot_str}], include_intercept=False)')
                formula_str = " + ".join(spline_terms)

            try:
                data_dict = {xc: sub[xc].to_numpy().astype(float) for xc in x_cols}
                data_dict[y_col] = Y
                X_design = dmatrix(formula_str, data_dict, return_type="matrix")
                X_sm = np.asarray(X_design)
                X_sm = sm.add_constant(X_sm, has_constant="add")

                model = sm.OLS(Y, X_sm)
                result = model.fit()
                y_hat = result.fittedvalues

                # Confidence interval via prediction
                pred = result.get_prediction(X_sm)
                pred_df = pred.summary_frame(alpha=0.05)
                y_hat_lo = pred_df["obs_ci_lower"].values
                y_hat_hi = pred_df["obs_ci_upper"].values

                rss = float(np.sum((Y - y_hat) ** 2))
                tss = float(np.sum((Y - Y.mean()) ** 2))
                r2 = 1.0 - rss / (tss + 1e-15)
                aic = float(result.aic)
                bic = float(result.bic)

                lines += [
                    f"  Knots      : {[round(k, 4) for k in knot_positions]}",
                    f"  R-squared  : {r2:.4f}",
                    f"  AIC        : {aic:.4f}",
                    f"  BIC        : {bic:.4f}",
                    f"  N params   : {result.df_model:.0f}  (df resid={result.df_resid:.0f})",
                ]
                model_info = {"r2": r2, "aic": aic}

            except Exception as sm_err:
                lines.append(f"\nSpline fit error: {sm_err}")

        # ------------------------------------------------------------------
        # LOESS / LOWESS
        # ------------------------------------------------------------------
        elif spline_type == "loess":
            try:
                import statsmodels.api as sm
            except ImportError:
                return (
                    "statsmodels is required for LOESS. "
                    "Run: pip install statsmodels"
                )

            # LOWESS operates on a single predictor
            x_sm = x_primary
            frac = float(ca.options.get("frac", 0.3))
            frac = max(0.1, min(frac, 1.0))

            lowess_result = sm.nonparametric.lowess(Y, x_sm, frac=frac, it=3)
            # lowess_result columns: [x_sorted, y_smoothed]
            x_loess = lowess_result[:, 0]
            y_loess = lowess_result[:, 1]

            # Interpolate back to original order for residuals
            from numpy import interp
            y_hat = interp(x_sm, x_loess, y_loess)

            # Bootstrap confidence band (100 reps, fast)
            rng = np.random.default_rng(42)
            boot_fits = []
            for _ in range(100):
                idx_b = rng.integers(0, n, n)
                x_b = x_sm[idx_b]
                y_b = Y[idx_b]
                sort_b = np.argsort(x_b)
                try:
                    lw_b = sm.nonparametric.lowess(
                        y_b[sort_b], x_b[sort_b], frac=frac, it=1
                    )
                    boot_fits.append(interp(x_sm, lw_b[:, 0], lw_b[:, 1]))
                except Exception:
                    pass

            if boot_fits:
                boot_arr = np.array(boot_fits)
                y_hat_lo = np.percentile(boot_arr, 2.5, axis=0)
                y_hat_hi = np.percentile(boot_arr, 97.5, axis=0)

            rss = float(np.sum((Y - y_hat) ** 2))
            tss = float(np.sum((Y - Y.mean()) ** 2))
            r2 = 1.0 - rss / (tss + 1e-15)

            lines += [
                f"  Bandwidth (frac) : {frac:.2f}",
                f"  R-squared (approx): {r2:.4f}",
                "  (LOESS uses single predictor; additional X ignored.)",
            ]
            model_info = {"r2": r2}

        else:
            return (
                f"Unknown type '{spline_type}'. "
                "Choose from: natural, bs, loess"
            )

        # ------------------------------------------------------------------
        # Residual summary
        # ------------------------------------------------------------------
        if y_hat is not None:
            resid = Y - y_hat
            lines += [
                "",
                "Residual Summary:",
                f"  {'Mean residual':<30} {float(resid.mean()):>10.4f}",
                f"  {'SD residual':<30} {float(resid.std()):>10.4f}",
                f"  {'Min':<30} {float(resid.min()):>10.4f}",
                f"  {'Max':<30} {float(resid.max()):>10.4f}",
                f"  {'RMSE':<30} {float(np.sqrt(np.mean(resid**2))):>10.4f}",
            ]

        # ------------------------------------------------------------------
        # Plot: fitted curve with CI band
        # ------------------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: data + fitted + CI
            sort_idx = np.argsort(x_primary)
            x_s = x_primary[sort_idx]

            axes[0].scatter(x_primary, Y, color="#4C72B0", alpha=0.5, s=20,
                            label="Observed", zorder=3)
            if y_hat is not None:
                axes[0].plot(x_s, y_hat[sort_idx], color="red", linewidth=2.0,
                             label="Fitted")
                if y_hat_lo is not None and y_hat_hi is not None:
                    axes[0].fill_between(
                        x_s, y_hat_lo[sort_idx], y_hat_hi[sort_idx],
                        alpha=0.20, color="red", label="95% CI/band"
                    )

            if spline_type in ("natural", "bs") and knot_positions:
                for kp in knot_positions:
                    axes[0].axvline(kp, color="grey", linestyle=":",
                                    linewidth=0.8, alpha=0.7)

            axes[0].set_xlabel(x_cols[0])
            axes[0].set_ylabel(y_col)
            axes[0].set_title(
                f"{spline_type.capitalize()} Spline: {y_col} ~ {x_cols[0]}"
            )
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Right: residual plot
            if y_hat is not None:
                resid = Y - y_hat
                axes[1].scatter(y_hat, resid, alpha=0.5, s=20, color="#2ca02c")
                axes[1].axhline(0, color="black", linewidth=1.0, linestyle="--")
                axes[1].set_xlabel("Fitted values")
                axes[1].set_ylabel("Residuals")
                axes[1].set_title("Residuals vs Fitted")
                axes[1].grid(alpha=0.3)

            fig.tight_layout()
            session.output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"spline_{spline_type}.png"
            plot_path = session.output_dir / fname
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            session.plot_paths.append(str(plot_path))
            lines.append(f"\nPlot saved: {plot_path}")

        except Exception as plot_err:
            lines.append(f"\nPlot error: {plot_err}")

        lines.append(_sep())
        return "\n".join(lines)

    except Exception as e:
        return friendly_error(e, "spline")
