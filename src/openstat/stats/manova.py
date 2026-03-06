"""MANOVA and two-way ANOVA."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats


# ── Two-way ANOVA ─────────────────────────────────────────────────────────

def twoway_anova(
    df: pl.DataFrame,
    dep: str,
    factor1: str,
    factor2: str,
    *,
    interaction: bool = True,
) -> dict:
    """Two-way ANOVA with optional interaction term.

    Uses OLS approach (type III sums of squares via statsmodels).
    """
    import statsmodels.formula.api as smf

    formula = f"Q('{dep}') ~ C(Q('{factor1}')) + C(Q('{factor2}'))"
    if interaction:
        formula += f" + C(Q('{factor1}')):C(Q('{factor2}'))"

    pdf = df.select([dep, factor1, factor2]).drop_nulls().to_pandas()
    # rename cols to safe names for formula
    pdf.columns = ["dep", "f1", "f2"]
    formula = "dep ~ C(f1) + C(f2)"
    if interaction:
        formula += " + C(f1):C(f2)"

    model = smf.ols(formula, data=pdf).fit()

    from statsmodels.stats.anova import anova_lm
    anova_table = anova_lm(model, typ=3)

    rows = []
    for source, row in anova_table.iterrows():
        rows.append({
            "source": str(source).replace("C(f1)", factor1).replace("C(f2)", factor2),
            "df": int(row.get("df", 0)),
            "SS": float(row.get("sum_sq", float("nan"))),
            "MS": float(row.get("mean_sq", float("nan"))),
            "F": float(row.get("F", float("nan"))),
            "p_value": float(row.get("PR(>F)", float("nan"))),
        })

    return {
        "test": "Two-way ANOVA",
        "dep": dep,
        "factor1": factor1,
        "factor2": factor2,
        "interaction": interaction,
        "n_obs": int(pdf.shape[0]),
        "r_squared": float(model.rsquared),
        "table": rows,
    }


# ── MANOVA ─────────────────────────────────────────────────────────────────

def fit_manova(
    df: pl.DataFrame,
    dep_vars: list[str],
    group: str,
) -> dict:
    """
    One-way MANOVA via statsmodels.

    Tests whether group means differ on a set of dependent variables.
    """
    try:
        from statsmodels.multivariate.manova import MANOVA

        dep_str = " + ".join(f"Q('{d}')" for d in dep_vars)
        pdf = df.select(dep_vars + [group]).drop_nulls().to_pandas()
        # safe column names
        safe_deps = [f"y{i}" for i in range(len(dep_vars))]
        safe_group = "group_var"
        mapping = dict(zip(dep_vars + [group], safe_deps + [safe_group]))
        pdf.rename(columns=mapping, inplace=True)
        dep_formula = " + ".join(safe_deps)
        formula = f"{dep_formula} ~ C({safe_group})"

        mv = MANOVA.from_formula(formula, data=pdf)
        res = mv.mv_test()

        # Extract Wilks' Lambda from the test
        stats_dict = {}
        for effect, effect_res in res.results.items():
            for stat_name, vals in effect_res["stat"].items():
                stats_dict[f"{effect}_{stat_name}"] = vals

        # Build a clean summary
        effects = []
        for effect_name, effect_res in res.results.items():
            stat_df = effect_res["stat"]
            for test_name in stat_df.index:
                effects.append({
                    "effect": str(effect_name),
                    "test": str(test_name),
                    "statistic": float(stat_df.loc[test_name, "Value"]),
                    "F": float(stat_df.loc[test_name, "F Value"]),
                    "num_df": float(stat_df.loc[test_name, "Num DF"]),
                    "den_df": float(stat_df.loc[test_name, "Den DF"]),
                    "p_value": float(stat_df.loc[test_name, "Pr > F"]),
                })

        return {
            "test": "MANOVA",
            "dep_vars": dep_vars,
            "group": group,
            "n_obs": len(pdf),
            "n_groups": int(pdf[safe_group].nunique()),
            "effects": effects,
        }

    except Exception as exc:
        raise RuntimeError(f"MANOVA failed: {exc}") from exc
