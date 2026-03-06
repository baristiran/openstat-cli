"""Command package — import all command modules to register them."""

# Importing these modules triggers the @command decorators,
# which populate the global registry in base.py.
from openstat.commands import data_cmds   # noqa: F401
from openstat.commands import stat_cmds   # noqa: F401
from openstat.commands import plot_cmds   # noqa: F401
from openstat.commands import report_cmds  # noqa: F401
from openstat.commands import plugin_cmds  # noqa: F401
from openstat.commands import panel_cmds   # noqa: F401
from openstat.commands import iv_cmds      # noqa: F401
from openstat.commands import mixed_cmds   # noqa: F401
from openstat.commands import ts_cmds      # noqa: F401
from openstat.commands import surv_cmds    # noqa: F401
from openstat.commands import mi_cmds      # noqa: F401
from openstat.commands import survey_cmds  # noqa: F401
from openstat.commands import backend_cmds  # noqa: F401
from openstat.commands import discrete_cmds  # noqa: F401
from openstat.commands import causal_cmds    # noqa: F401
from openstat.commands import power_cmds     # noqa: F401
from openstat.commands import factor_cmds    # noqa: F401
from openstat.commands import export_cmds    # noqa: F401
from openstat.commands import nonparam_cmds  # noqa: F401
from openstat.commands import ml_cmds        # noqa: F401
from openstat.commands import cluster_cmds   # noqa: F401
from openstat.commands import manova_cmds    # noqa: F401
from openstat.commands import arch_cmds      # noqa: F401
from openstat.commands import bayes_cmds     # noqa: F401
from openstat.commands import reshape_cmds   # noqa: F401
from openstat.commands import advreg_cmds    # noqa: F401
from openstat.commands import ts_adv_cmds   # noqa: F401
from openstat.commands import influence_cmds # noqa: F401
from openstat.commands import ml_adv_cmds   # noqa: F401
from openstat.commands import esttab_cmds   # noqa: F401
from openstat.commands import string_cmds   # noqa: F401
from openstat.commands import epi_cmds      # noqa: F401
from openstat.commands import dsl_cmds          # noqa: F401
from openstat.commands import resampling_cmds   # noqa: F401
from openstat.commands import model_eval_cmds   # noqa: F401
from openstat.commands import dataquality_cmds  # noqa: F401
from openstat.commands import outreg_cmds       # noqa: F401
from openstat.commands import equiv_tobit_cmds  # noqa: F401
from openstat.commands import viz_extra_cmds    # noqa: F401
from openstat.commands import posthoc_cmds      # noqa: F401
from openstat.commands import sem_cmds          # noqa: F401
from openstat.commands import meta_cmds         # noqa: F401
from openstat.commands import network_cmds      # noqa: F401
from openstat.commands import automodel_cmds    # noqa: F401
from openstat.commands import repro_cmds        # noqa: F401
from openstat.commands import tui_cmds          # noqa: F401
from openstat.commands import i18n_cmds         # noqa: F401
from openstat.commands import mediate_cmds      # noqa: F401
from openstat.commands import validate_cmds     # noqa: F401
from openstat.commands import regex_cmds        # noqa: F401
from openstat.commands import alias_cmds        # noqa: F401
from openstat.commands import pdf_cmds          # noqa: F401
from openstat.commands import watch_cmds        # noqa: F401
from openstat.commands import rbridge_cmds      # noqa: F401
from openstat.commands import nlquery_cmds      # noqa: F401
from openstat.commands import stata_import_cmds # noqa: F401
from openstat.commands import help_cmds         # noqa: F401
from openstat.commands import datetime_cmds     # noqa: F401
from openstat.commands import groupby_cmds      # noqa: F401
from openstat.commands import profile_cmds      # noqa: F401
from openstat.commands import advanced_ml_cmds  # noqa: F401
from openstat.commands import viz_adv_cmds      # noqa: F401
from openstat.commands import pipeline_cmds     # noqa: F401
from openstat.commands import export_extra_cmds # noqa: F401
from openstat.commands import datamanip_cmds    # noqa: F401
from openstat.commands import import_extra_cmds # noqa: F401
from openstat.commands import adv_stat_cmds      # noqa: F401
from openstat.commands import mixture_changepoint_cmds  # noqa: F401
from openstat.commands import dimreduce_cmds     # noqa: F401
from openstat.commands import arules_cmds        # noqa: F401
from openstat.commands import textanalysis_cmds  # noqa: F401
from openstat.commands import ux_cmds            # noqa: F401
from openstat.commands import export_beamer_cmds # noqa: F401

from openstat.commands.base import get_registry

# Public API — the COMMANDS dict used by the REPL
COMMANDS = get_registry()
