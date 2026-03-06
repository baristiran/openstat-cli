"""R bridge command: run R code from OpenStat (requires rpy2)."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


@command("r", usage='r "<R code>"')
def cmd_r(session: Session, args: str) -> str:
    """Execute R code in the current session (requires rpy2).

    The current dataset is available in R as 'data'.
    Results printed in R are captured and returned.
    Modified 'data' is pulled back into the OpenStat session.

    Examples:
      r "summary(data)"
      r "cor(data[, sapply(data, is.numeric)])"
      r "data$log_income <- log(data$income + 1)"
      r "lm_result <- lm(y ~ x1 + x2, data=data); summary(lm_result)"
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        import rpy2.rinterface_lib.callbacks as rcb
    except ImportError:
        return (
            "rpy2 is required for the R bridge.\n"
            "Install: pip install rpy2\n"
            "Also requires a working R installation."
        )

    code = args.strip().strip('"\'')
    if not code:
        return 'Usage: r "<R code>"'

    import io as _io
    output_lines: list[str] = []

    # Capture R output
    def _capture(x):
        output_lines.append(x)

    old_write = rcb.consolewrite_print
    rcb.consolewrite_print = _capture

    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            # Push current dataframe into R as 'data'
            if session.df is not None:
                try:
                    r_df = ro.conversion.py2rpy(session.df.to_pandas())
                    ro.globalenv["data"] = r_df
                except Exception:
                    pass  # non-critical

            # Execute R code
            ro.r(code)

            # Pull 'data' back if it was modified
            try:
                r_data = ro.globalenv.get("data")
                if r_data is not None:
                    import polars as pl
                    pd_df = ro.conversion.rpy2py(r_data)
                    new_df = pl.from_pandas(pd_df)
                    if session.df is None or not new_df.equals(session.df):
                        session.snapshot()
                        session.df = new_df
            except Exception:
                pass

    except Exception as e:
        return friendly_error(e, "R bridge")
    finally:
        rcb.consolewrite_print = old_write

    result = "".join(output_lines).strip()
    return result or "[R code executed — no output]"
