# Changelog

All notable changes to OpenStat are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.9.0] – 2025-05-01

### Added
- **Post-hoc tests** (`posthoc`): Tukey HSD, Bonferroni, Scheffé pairwise comparisons after ANOVA
- **Coefficient plot** (`plot coef`): horizontal error-bar plot of regression coefficients with 95% CI
- **Marginal effects plot** (`plot margins`): visualise marginal effects after `margins`
- **Interaction plot** (`plot interaction <y> <x> <moderator>`): shows regression lines by ±1 SD groups of the moderator
- **Real-time session log** (`log using <path>`): streams every command and its output to a file
- **Script runner enhancements**: `foreach`, `forvalues`, `if/else` blocks with variable substitution (`{var}`)
- **Database connectivity** (`sqlload`): load data directly from SQL databases via connection URL (requires `connectorx`)
- **SEM / CFA** (`sem`, `cfa`): structural equation modelling and confirmatory factor analysis via semopy
- **Meta-analysis** (`meta`): fixed-effects (inverse-variance) and random-effects (DerSimonian-Laird), forest and funnel plots
- **Network analysis** (`network`): build, describe, centrality, community detection, and plotting via networkx
- **Auto model selection** (`automodel`): exhaustive subset search (≤8 predictors) or forward stepwise, ranked by AIC/BIC
- **IPTW** (`iptw`): inverse probability of treatment weighting for causal inference; ATE/ATT; covariate balance table
- **Reproducibility** (`session info/save/replay`, `set seed`, `version`): full session management and script replay
- **TUI Dashboard** (`dashboard`): full-screen terminal UI with dataset overview, variable table, model results, and history (requires `textual`)
- **PyPI packaging**: `pyproject.toml` polished for release; new extras `database`, `sem`, `network`, `tui`

### Changed
- `export docx` now includes a Summary Statistics table (N, Mean, SD, Min–Max per numeric column)
- `set` command extended with `set seed <N>` sub-command
- `log` command moved from `report_cmds` to `outreg_cmds` and upgraded to real-time streaming

### Fixed
- `plot coef` crash when `params` is a numpy array (not a pandas Series)
- `} else {` block parsing in script runner
- `automodel` formula normalisation (space-separated predictors now work)
- `semopy.Model` API: `obj=` parameter goes in `fit()`, not `__init__()`
- Duplicate command registration for `set` and `log`

---

## [0.8.0] – 2025-03-01

### Added
- **Bayesian inference** (`bayes`): MCMC sampling, posterior summaries, trace/posterior plots
- **ARCH/GARCH** (`arch`, `garch`): volatility modelling for financial time series
- **MANOVA** (`manova`): multivariate analysis of variance
- **Clustering** (`cluster kmeans`, `cluster hclust`, `discriminant`): k-means, hierarchical, LDA/QDA
- **Advanced ML** (`randomforest`, `gradientboost`, `neuralnet`, `svm`, `knn`): ensemble and deep learning models
- **Influence diagnostics** (`influence`): Cook's distance, DFFITS, leverage plots
- **Advanced regression** (`quantreg`, `truncreg`, `intreg`, `heckman`): quantile, truncated, interval, and selection models
- **Advanced time series** (`vecm`, `var`, `granger`, `threshold`): VAR, VECM, Granger causality, threshold models
- **Epidemiology** (`epi`): risk ratios, odds ratios, attributable risk, Mantel-Haenszel pooling
- **Equivalence testing** (`equiv`, `tobit`): TOST and Tobit censored regression
- **String commands** (`strreplace`, `strsplit`, `strextract`, `strpad`, `strcat`): column string manipulation
- **DSL / macro system** (`define`, `macro`, `eval`): variable macros and expressions
- **Resampling** (`bootstrap`, `jackknife`, `permtest`): resampling-based inference
- **Model evaluation** (`roc`, `calibration`, `confusion`): classification diagnostics
- **Data quality** (`missing`, `duplicates`, `outlier`, `assert`): profiling and validation
- **Reshape** (`reshape wide`, `reshape long`, `pivot`, `unpivot`): data reshaping
- **esttab** (`esttab`): publication-quality coefficient tables (LaTeX, HTML, Markdown)
- **outreg2** (`outreg2`): Word/RTF-compatible regression tables
- **Visualisation extras** (`plot violin`, `plot pairplot`, `plot parallel`, `plot density`): additional plot types

### Changed
- Session now tracks `_last_fit_result` and `_last_fit_kwargs` for bootstrap/esttab integration

---

## [0.7.0] – 2025-01-01

### Added
- **Survey analysis** (`svyset`, `svy: mean`, `svy: total`, `svy: proportion`, `svy: reg`): complex survey design
- **Multiple imputation** (`mi impute`, `mi estimate`): MICE-based imputation with pooled estimates
- **DuckDB backend** (`set backend duckdb`, `sql`): fast in-memory SQL on datasets
- **Web API** (`openstat web`): FastAPI + WebSocket server for browser-based access
- **Jupyter magic** (`%openstat`): run OpenStat commands in Jupyter notebooks
- **Plugin system** (`plugin load/list/unload`): third-party command packages

---

## [0.6.0] – 2024-11-01

### Added
- **Power analysis** (`power`): t-test, ANOVA, chi-square, proportion power and sample size
- **Factor analysis** (`factor`, `pca`, `rotate`): EFA, PCA, varimax/oblimin rotation
- **IV regression** (`ivregress`): two-stage least squares
- **Mixed models** (`mixed`): linear mixed-effects models via statsmodels
- **Panel data** (`xtset`, `xtreg`, `xttest`, `hausman`): fixed/random effects, Hausman test

---

## [0.5.0] – 2024-09-01

### Added
- **Survival analysis** (`stset`, `sts graph`, `stcox`, `streg`, `stsum`): Kaplan-Meier, Cox PH, AFT
- **Time series** (`tsset`, `arima`, `ardl`, `adf`, `kpss`, `forecast`): ARIMA, ARDL, unit-root tests
- **Causal inference** (`pscore`, `teffects`, `did`, `rddesign`, `synth`): propensity score, DiD, RD
- **Discrete choice** (`logit`, `probit`, `ologit`, `oprobit`, `mlogit`, `clogit`): limited dependent variable models
- **Undo/redo** (`undo`, `redo`): step-back for data transformations

---

## [0.4.0] – 2024-07-01

### Added
- **Non-parametric tests** (`kruskal`, `mannwhitney`, `wilcoxon`, `friedman`, `spearman`): rank-based inference
- **Report generation** (`report html`, `report latex`): automated analysis reports
- **Plot diagnostics** (`plot diagnostics`): residuals vs fitted, Q-Q, scale-location

---

## [0.3.0] – 2024-05-01

### Added
- **Data management** (`load`, `save`, `drop`, `keep`, `rename`, `encode`, `decode`, `generate`, `replace`, `sort`, `merge`, `append`, `sample`, `undo`)
- **Descriptive statistics** (`describe`, `summarize`, `tabulate`, `correlate`, `crosstab`, `anova`)
- **Regression** (`ols`, `logit`, `poisson`, `margins`, `predict`, `test`)
- **Plots** (`plot hist`, `plot scatter`, `plot line`, `plot box`, `plot bar`, `plot heatmap`, `plot acf`, `plot pacf`)
- **Output** (`export docx`, `export pptx`)
- **Configuration** (`config show`, `config set`)
- **Script runner** (`run <script.ost>`): execute .ost script files
- Interactive REPL with tab completion, syntax highlighting, and command history

---

## [0.1.0] – 2024-01-01

### Added
- Initial project scaffold
- Basic REPL infrastructure
- Polars DataFrame backend
