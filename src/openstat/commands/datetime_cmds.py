"""Datetime operations: extract, arithmetic, format."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("datetime", usage="datetime extract|diff|format|parse <col> [options]")
def cmd_datetime(session: Session, args: str) -> str:
    """Datetime column operations.

    Sub-commands:
      datetime extract <col> [into(<prefix>)]
          — extract year, month, day, hour, minute, weekday, quarter
      datetime diff <col1> <col2> [unit=days|hours|minutes] [into(<newcol>)]
          — compute difference between two date columns
      datetime format <col> <fmt> [into(<newcol>)]
          — reformat datetime as string (strftime format)
      datetime parse <col> [fmt=<format>] [into(<newcol>)]
          — parse a string column as datetime
      datetime shift <col> <N> <unit> [into(<newcol>)]
          — add/subtract time (e.g. shift date 7 days)

    Examples:
      datetime extract created_at into(dt)
        → creates dt_year, dt_month, dt_day, dt_weekday, dt_quarter
      datetime diff end_date start_date unit=days into(duration)
      datetime format created_at "%Y-%m" into(year_month)
      datetime parse date_str fmt="%d/%m/%Y" into(date)
      datetime shift order_date 30 days into(delivery_date)
    """
    import polars as pl

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: datetime extract|diff|format|parse|shift <col> ..."

    subcmd = ca.positional[0].lower()

    try:
        df = session.require_data()

        if subcmd == "extract":
            if len(ca.positional) < 2:
                return "Usage: datetime extract <col> [into(<prefix>)]"
            col = ca.positional[1]
            if col not in df.columns:
                return f"Column not found: {col}"

            into_raw = ca.rest_after("into")
            prefix = into_raw.strip().strip("()") if into_raw else col

            # Cast to datetime if string
            series = df[col]
            if series.dtype == pl.Utf8:
                series = series.str.to_datetime(strict=False)

            dt = series.dt
            new_cols = {
                f"{prefix}_year": dt.year(),
                f"{prefix}_month": dt.month(),
                f"{prefix}_day": dt.day(),
                f"{prefix}_hour": dt.hour(),
                f"{prefix}_minute": dt.minute(),
                f"{prefix}_weekday": dt.weekday(),
                f"{prefix}_quarter": dt.quarter(),
            }
            session.snapshot()
            session.df = df.with_columns([
                pl.Series(name, vals) for name, vals in new_cols.items()
            ])
            parts = ", ".join(new_cols.keys())
            return f"Extracted datetime components: {parts}"

        elif subcmd == "diff":
            if len(ca.positional) < 3:
                return "Usage: datetime diff <col1> <col2> [unit=days] [into(<newcol>)]"
            c1, c2 = ca.positional[1], ca.positional[2]
            unit = ca.options.get("unit", "days")
            into_raw = ca.rest_after("into")
            newcol = into_raw.strip().strip("()") if into_raw else f"{c1}_minus_{c2}"

            for c in [c1, c2]:
                if c not in df.columns:
                    return f"Column not found: {c}"

            def _to_dt(s):
                if s.dtype == pl.Utf8:
                    return s.str.to_datetime(strict=False)
                return s.cast(pl.Datetime)

            s1 = _to_dt(df[c1])
            s2 = _to_dt(df[c2])
            diff_dur = s1 - s2

            unit_map = {
                "days": 86_400_000_000,
                "hours": 3_600_000_000,
                "minutes": 60_000_000,
                "seconds": 1_000_000,
            }
            divisor = unit_map.get(unit, 86_400_000_000)
            diff_num = (diff_dur.dt.total_microseconds() / divisor).cast(pl.Float64)

            session.snapshot()
            session.df = df.with_columns(diff_num.alias(newcol))
            return f"Date difference stored in '{newcol}' ({unit}). Mean: {diff_num.mean():.2f}"

        elif subcmd == "format":
            if len(ca.positional) < 3:
                return "Usage: datetime format <col> <fmt> [into(<newcol>)]"
            col, fmt = ca.positional[1], ca.positional[2]
            if col not in df.columns:
                return f"Column not found: {col}"
            into_raw = ca.rest_after("into")
            newcol = into_raw.strip().strip("()") if into_raw else f"{col}_fmt"

            series = df[col]
            if series.dtype == pl.Utf8:
                series = series.str.to_datetime(strict=False)
            formatted = series.dt.strftime(fmt)

            session.snapshot()
            session.df = df.with_columns(formatted.alias(newcol))
            return f"Formatted '{col}' → '{newcol}' using format '{fmt}'"

        elif subcmd == "parse":
            if len(ca.positional) < 2:
                return "Usage: datetime parse <col> [fmt=<format>] [into(<newcol>)]"
            col = ca.positional[1]
            if col not in df.columns:
                return f"Column not found: {col}"
            fmt = ca.options.get("fmt")
            into_raw = ca.rest_after("into")
            newcol = into_raw.strip().strip("()") if into_raw else f"{col}_dt"

            if fmt:
                parsed = df[col].str.to_datetime(format=fmt, strict=False)
            else:
                parsed = df[col].str.to_datetime(strict=False)

            session.snapshot()
            session.df = df.with_columns(parsed.alias(newcol))
            n_ok = parsed.drop_nulls().len()
            return f"Parsed '{col}' → '{newcol}': {n_ok}/{df.height} rows parsed."

        elif subcmd == "shift":
            if len(ca.positional) < 4:
                return "Usage: datetime shift <col> <N> <unit> [into(<newcol>)]"
            col = ca.positional[1]
            n = int(ca.positional[2])
            unit = ca.positional[3].lower().rstrip("s")  # days→day, hours→hour
            if col not in df.columns:
                return f"Column not found: {col}"
            into_raw = ca.rest_after("into")
            newcol = into_raw.strip().strip("()") if into_raw else f"{col}_shifted"

            series = df[col]
            if series.dtype == pl.Utf8:
                series = series.str.to_datetime(strict=False)

            from datetime import timedelta
            unit_map2 = {"day": "days", "hour": "hours", "minute": "minutes", "second": "seconds", "week": "weeks"}
            td_key = unit_map2.get(unit, "days")
            shifted = series + pl.duration(**{td_key: n})

            session.snapshot()
            session.df = df.with_columns(shifted.alias(newcol))
            return f"Shifted '{col}' by {n} {unit}(s) → '{newcol}'"

        else:
            return f"Unknown sub-command: {subcmd}. Use extract, diff, format, parse, or shift."

    except Exception as e:
        return friendly_error(e, "datetime")
