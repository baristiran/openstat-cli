"""Association rules: Apriori / FP-Growth."""

from __future__ import annotations
from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("arules", usage="arules <item_col> [<id_col>] [--minsup=0.05] [--minconf=0.5] [--algo=fpgrowth]")
def cmd_arules(session: Session, args: str) -> str:
    """Association rule mining (Apriori / FP-Growth).

    Mines frequent itemsets and generates association rules.
    Expects one-row-per-transaction with an item column, OR
    a transaction-id column + item column.

    Options:
      --minsup=<f>    minimum support (0–1, default: 0.05)
      --minconf=<f>   minimum confidence (0–1, default: 0.5)
      --minlift=<f>   minimum lift (default: 1.0)
      --algo=<a>      apriori or fpgrowth (default: fpgrowth)
      --top=<n>       show top N rules (default: 20)

    Examples:
      arules item transaction_id --minsup=0.1 --minconf=0.6
      arules product --minsup=0.05 --algo=apriori --top=10
    """
    try:
        from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        return "mlxtend required. Install: pip install mlxtend"

    import polars as pl
    import pandas as pd

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: arules <item_col> [<id_col>]"

    item_col = ca.positional[0]
    id_col = ca.positional[1] if len(ca.positional) > 1 else None
    min_sup = float(ca.options.get("minsup", 0.05))
    min_conf = float(ca.options.get("minconf", 0.5))
    min_lift = float(ca.options.get("minlift", 1.0))
    algo = ca.options.get("algo", "fpgrowth").lower()
    top_n = int(ca.options.get("top", 20))

    try:
        df = session.require_data()
        if item_col not in df.columns:
            return f"Column not found: {item_col}"
        if id_col and id_col not in df.columns:
            return f"Column not found: {id_col}"

        # Build transactions
        if id_col:
            # Group items by transaction id
            transactions = (
                df.select([id_col, item_col])
                .group_by(id_col)
                .agg(pl.col(item_col).cast(pl.Utf8).alias("items"))
                ["items"].to_list()
            )
        else:
            # Each row is a transaction; split by comma if needed
            col_vals = df[item_col].cast(pl.Utf8).to_list()
            transactions = [[v.strip() for v in row.split(",") if v.strip()]
                            for row in col_vals if row]

        if len(transactions) < 2:
            return "Need at least 2 transactions."

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        basket_df = pd.DataFrame(te_array, columns=te.columns_)

        if algo == "apriori":
            frequent = apriori(basket_df, min_support=min_sup, use_colnames=True)
        else:
            frequent = fpgrowth(basket_df, min_support=min_sup, use_colnames=True)

        if frequent.empty:
            return f"No frequent itemsets found at min_support={min_sup}. Try lowering --minsup."

        rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift].sort_values("lift", ascending=False)

        lines = [
            f"Association Rules ({algo.upper()}) — {item_col}",
            f"  Transactions: {len(transactions)}, Items: {len(te.columns_)}",
            f"  min_support={min_sup}, min_confidence={min_conf}, min_lift={min_lift}",
            f"  Frequent itemsets: {len(frequent)}, Rules: {len(rules)}",
            "",
        ]

        if rules.empty:
            lines.append("  No rules found. Try lowering --minconf or --minlift.")
        else:
            lines.append(f"  Top {min(top_n, len(rules))} rules by lift:")
            lines.append(f"  {'Antecedent':<30} {'Consequent':<20} {'Sup':>7} {'Conf':>7} {'Lift':>7}")
            lines.append("  " + "-" * 75)
            for _, row in rules.head(top_n).iterrows():
                ant = ", ".join(sorted(row["antecedents"]))[:28]
                con = ", ".join(sorted(row["consequents"]))[:18]
                lines.append(
                    f"  {ant:<30} {con:<20} {row['support']:>7.3f} {row['confidence']:>7.3f} {row['lift']:>7.3f}"
                )

        return "\n".join(lines)
    except Exception as e:
        return friendly_error(e, "arules")
