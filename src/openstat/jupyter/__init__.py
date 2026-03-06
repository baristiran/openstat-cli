"""Jupyter notebook integration for OpenStat.

Usage in Jupyter:
    %load_ext openstat
    %ost load data.csv
    %ost summarize

    %%openstat
    load data.csv
    summarize
    ols y ~ x1 + x2
"""


def load_ipython_extension(ipython):
    """Register OpenStat magics when %load_ext openstat is called."""
    from openstat.jupyter.magic import OpenStatMagics
    ipython.register_magics(OpenStatMagics)
