"""Post-hoc comparison commands: posthoc (Tukey HSD, Bonferroni, Scheffé)."""

from __future__ import annotations

import re

import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison

from openstat.commands.base import command
from openstat.session import Session


def _fmt_table(title: str, headers: list[str], rows: list[tuple]) -> str:
    all_rows = [tuple(str(v) for v in r) for r in rows]
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in all_rows), default=0))
        for i in range(len(headers))
    ]
    sep = "-" * (sum(widths) + 3 * len(widths) + 1)
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    lines = [f"\n{title}", "=" * len(sep), header_line, sep]
    for row in all_rows:
        lines.append(" | ".join(f"{v:<{w}}" for v, w in zip(row, widths)))
    lines.append("=" * len(sep))
    return "\n".join(lines)


@command("posthoc", usage="posthoc <var> by(<group>) [--tukey|--bonferroni|--scheffe]")
def cmd_posthoc(session: Session, args: str) -> str:
    """Post-hoc pairwise comparisons after ANOVA (Tukey HSD, Bonferroni, Scheffé).

    Examples:
      posthoc score by(region)
      posthoc income by(education) --bonferroni
      posthoc age by(group) --scheffe
    """
    df = session.require_data()

    m = re.search(r"by\((\w+)\)", args)
    if not m:
        return "Usage: posthoc <var> by(<group>) [--tukey|--bonferroni|--scheffe]"
    group_col = m.group(1)

    rest = re.sub(r"by\([^)]*\)", "", args)
    tokens = [t for t in rest.split() if not t.startswith("--")]
    if not tokens:
        return "Usage: posthoc <var> by(<group>) [--tukey|--bonferroni|--scheffe]"
    var = tokens[0]

    if var not in df.columns:
        return f"Column not found: {var}"
    if group_col not in df.columns:
        return f"Column not found: {group_col}"

    method = "tukey"
    if "--bonferroni" in args:
        method = "bonferroni"
    elif "--scheffe" in args:
        method = "scheffe"

    sub = df.select([var, group_col]).drop_nulls()
    values = sub[var].to_numpy(allow_copy=True).astype(float)
    groups = sub[group_col].to_list()

    group_labels = sorted(set(str(g) for g in groups))
    groups_str = [str(g) for g in groups]

    if len(group_labels) < 2:
        return "Need at least 2 groups for post-hoc comparison."

    # Run overall ANOVA first
    group_arrays = [values[np.array([g == lbl for g in groups_str])] for lbl in group_labels]
    f_stat, p_overall = stats.f_oneway(*group_arrays)
    anova_line = f"Overall ANOVA: F = {f_stat:.4f}, p = {p_overall:.4f}"

    try:
        if method == "tukey":
            mc = MultiComparison(values, groups_str)
            result = mc.tukeyhsd()
            summary_data = result.summary().data
            headers = ["Group1", "Group2", "MeanDiff", "Lower", "Upper", "p-adj", "Reject H0"]
            rows = []
            for row in summary_data[1:]:
                g1, g2, meandiff, p_adj, lower, upper, reject = row
                rows.append((
                    str(g1), str(g2),
                    f"{float(meandiff):.4f}",
                    f"{float(lower):.4f}",
                    f"{float(upper):.4f}",
                    f"{float(p_adj):.4f}",
                    "Yes" if reject else "No",
                ))
            return anova_line + _fmt_table("Tukey HSD Post-hoc Comparison", headers, rows)

        elif method == "bonferroni":
            n_pairs = len(group_labels) * (len(group_labels) - 1) // 2
            alpha_adj = 0.05 / n_pairs
            rows = []
            for i in range(len(group_labels)):
                for j in range(i + 1, len(group_labels)):
                    a = group_arrays[i]
                    b = group_arrays[j]
                    t_stat, p_raw = stats.ttest_ind(a, b)
                    p_adj = min(p_raw * n_pairs, 1.0)
                    diff = np.mean(a) - np.mean(b)
                    rows.append((
                        group_labels[i], group_labels[j],
                        f"{diff:.4f}",
                        f"{t_stat:.4f}",
                        f"{p_raw:.4f}",
                        f"{p_adj:.4f}",
                        "Yes" if p_adj < 0.05 else "No",
                    ))
            headers = ["Group1", "Group2", "MeanDiff", "t-stat", "p-raw", "p-adj(Bonf)", "Reject H0"]
            note = f"\n  (Bonferroni correction: α_adj = 0.05/{n_pairs} = {alpha_adj:.5f})"
            return anova_line + _fmt_table("Bonferroni Post-hoc Comparison", headers, rows) + note

        elif method == "scheffe":
            k = len(group_labels)
            n_total = sum(len(d) for d in group_arrays)
            n_per = [len(d) for d in group_arrays]
            means = [np.mean(d) for d in group_arrays]

            ss_within = sum(np.sum((d - np.mean(d)) ** 2) for d in group_arrays)
            df_within = n_total - k
            mse = ss_within / df_within
            f_crit = stats.f.ppf(0.95, k - 1, df_within)
            critical = (k - 1) * f_crit

            rows = []
            for i in range(k):
                for j in range(i + 1, k):
                    diff = means[i] - means[j]
                    f_s = diff ** 2 / (mse * (1.0 / n_per[i] + 1.0 / n_per[j]))
                    p_val = 1.0 - stats.f.cdf(f_s / (k - 1), k - 1, df_within)
                    rows.append((
                        group_labels[i], group_labels[j],
                        f"{diff:.4f}",
                        f"{f_s:.4f}",
                        f"{critical:.4f}",
                        f"{p_val:.4f}",
                        "Yes" if f_s > critical else "No",
                    ))
            headers = ["Group1", "Group2", "MeanDiff", "F*", "F-critical", "p-value", "Reject H0"]
            note = f"\n  (Scheffé critical value = (k-1)×F_crit = {k-1}×{f_crit:.4f} = {critical:.4f})"
            return anova_line + _fmt_table("Scheffé Post-hoc Comparison", headers, rows) + note

    except Exception as exc:
        return f"posthoc error: {exc}"

    return "Unknown method."
