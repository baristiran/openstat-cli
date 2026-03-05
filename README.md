<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-brightgreen?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/tests-343%20passed-success?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/powered%20by-Polars%20%7C%20statsmodels-purple?style=for-the-badge" alt="Stack">
</p>

<h1 align="center">OpenStat</h1>

<p align="center">
  <strong>The open-source statistical analysis tool you've been waiting for.</strong><br>
  Load data. Explore. Transform. Model. Plot. Report. All from your terminal.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-why-openstat">Why OpenStat?</a> &bull;
  <a href="#-full-command-reference">Commands</a> &bull;
  <a href="#-statistical-models">Models</a> &bull;
  <a href="#-contributing">Contributing</a>
</p>

---

> **Note:** OpenStat is an independent, community-driven open-source project. It is not affiliated with, endorsed by, or connected to StataCorp LLC or any commercial statistical software vendor.

## Why OpenStat?

**Statistical analysis shouldn't require expensive licenses.** Every researcher, student, data scientist, and curious mind deserves access to professional-grade statistical tools — for free, forever.

OpenStat brings the familiar workflow of commercial statistical packages into your terminal with a clean, intuitive REPL. It's built on the incredible open-source Python ecosystem (Polars, statsmodels, scipy) and designed to be:

- **Accessible** — No licensing fees. No registration. Just `pip install` and go.
- **Familiar** — If you've used Stata, R, or SPSS, you'll feel right at home.
- **Fast** — Powered by [Polars](https://pola.rs/) (not pandas) for blazing-fast data operations.
- **Safe** — No `eval()` anywhere. All user expressions go through a secure whitelist parser.
- **Scriptable** — Write `.ost` scripts for reproducible analysis pipelines.
- **Extensible** — Adding a new command takes 10 lines of code. Seriously.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/baristiran/OpenStat.git
cd OpenStat

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install OpenStat with all dependencies
pip install -e ".[dev]"
```

### Launch the Interactive REPL

```bash
openstat repl
```

```
OpenStat v0.2.0 — Open-source statistical analysis tool
Type help for commands, quit to exit.

openstat> load examples/data.csv
Loaded 50 rows x 7 columns from examples/data.csv

openstat> summarize age income score
┌──────────┬────┬─────────┬─────────┬───────┬─────────┬─────────┬─────────┬─────────┐
│ Variable │ N  │ Mean    │ SD      │ Min   │ P25     │ P50     │ P75     │ Max     │
├──────────┼────┼─────────┼─────────┼───────┼─────────┼─────────┼─────────┼─────────┤
│ age      │ 50 │ 34.6600 │  8.7634 │ 21.00 │ 27.2500 │ 34.0000 │ 42.5000 │ 53.0000 │
│ income   │ 50 │ 49840.0 │ 17547.2 │ 26000 │ 34000.0 │ 47000.0 │ 66000.0 │ 88000.0 │
│ score    │ 50 │  7.4280 │  1.2844 │  4.90 │  6.4750 │  7.5000 │  8.5500 │  9.4000 │
└──────────┴────┴─────────┴─────────┴───────┴─────────┴─────────┴─────────┴─────────┘

openstat> ols score ~ age + income --robust
┌──────────┬────────┬─────────┬───────┬────────┬────────────┬─────────────┐
│ Variable │ Coef   │ Std.Err │ t/z   │ P>|t|  │ [95% CI L] │ [95% CI H]  │
├──────────┼────────┼─────────┼───────┼────────┼────────────┼─────────────┤
│ _cons    │ 2.1435 │ 0.4521  │ 4.741 │ 0.0000 │ 1.2343     │ 3.0527      │
│ age      │ 0.0312 │ 0.0187  │ 1.668 │ 0.1018 │ -0.0066    │ 0.0690      │
│ income   │ 0.0001 │ 0.0000  │ 5.234 │ 0.0000 │ 0.0000     │ 0.0001      │
└──────────┴────────┴─────────┴───────┴────────┴────────────┴─────────────┘
N = 50  |  R² = 0.5481  |  Adj.R² = 0.5289  |  F(2, 47) = 28.52 (p=0.0000)

openstat> predict yhat
Predictions added as 'yhat'. 50 rows x 8 columns.

openstat> quit
Bye!
```

### Run a Script

```bash
# Run an analysis script
openstat run examples/demo.ost

# Strict mode — stop on first error (great for CI/CD)
openstat run examples/demo.ost --strict
```

---

## What's New in v0.2.0

Version 0.2.0 is a massive leap in statistical depth. Here's what's new:

| Feature | What it does | Example |
|---------|-------------|---------|
| **Interaction Terms** | Model interactions between variables | `ols y ~ x1*x2` or `ols y ~ x1:x2` |
| **Cluster-Robust SE** | Standard errors robust to within-group correlation | `ols y ~ x1 + x2 --cluster=region` |
| **Poisson Regression** | Count data modeling with optional exposure offset | `poisson visits ~ age + income --exposure=time` |
| **Negative Binomial** | Overdispersed count data (reports dispersion alpha) | `negbin claims ~ age + gender` |
| **Quantile Regression** | Model any quantile, not just the mean | `quantreg y ~ x1 + x2 tau=0.75` |
| **Marginal Effects** | Average or at-means marginal effects for logit/probit | `margins --at=average` |
| **Bootstrap CI** | Non-parametric confidence intervals via resampling | `bootstrap n=1000 ci=95` |
| **Post-Estimation Diagnostics** | Breusch-Pagan, Ramsey RESET, link test, IC | `estat all` |
| **Model Comparison** | Side-by-side model comparison table | `estimates table` |
| **Multi-Way Interactions** | Three-way and beyond: `x1*x2*x3` auto-expands | Full factorial expansion |

---

## Full Command Reference

### Data Management

| Command | Description | Example |
|---------|-------------|---------|
| `load <path>` | Load CSV, Parquet, Stata (.dta), Excel (.xlsx) | `load survey.csv` |
| `save <path>` | Save data to any supported format | `save results.parquet` |
| `describe` | Show dataset structure (types, nulls) | `describe` |
| `head [N]` | Show first N rows (default: 10) | `head 20` |
| `tail [N]` | Show last N rows | `tail 5` |
| `count` | Row and column count | `count` |
| `merge <path> on <key> [how=...]` | Join with another file | `merge scores.csv on id how=left` |
| `undo` | Undo last data change (multi-level) | `undo` |

### Data Transformation

| Command | Description | Example |
|---------|-------------|---------|
| `filter <expr>` | Filter rows with expressions | `filter age > 30 and income < 50000` |
| `select <cols>` | Keep specific columns | `select age income score` |
| `derive <col> = <expr>` | Create new variables | `derive bmi = weight / (height ** 2)` |
| `dropna [cols]` | Drop missing values | `dropna age income` |
| `fillna <col> <strategy>` | Fill missing values | `fillna income median` |
| `sort <col> [--desc]` | Sort dataset | `sort income --desc` |
| `rename <old> <new>` | Rename a column | `rename income salary` |
| `cast <col> <type>` | Cast column type | `cast age float` |
| `encode <col> [as <new>]` | Label-encode strings | `encode region as region_code` |
| `recode <col> old=new ...` | Recode values | `recode region North=N South=S` |
| `replace <col> <old> <new>` | Replace values | `replace region North Norte` |
| `sample <N\|N%>` | Random sample | `sample 100` or `sample 10%` |
| `duplicates [drop] [cols]` | Find or drop duplicates | `duplicates drop` |
| `unique <col>` | List unique values | `unique region` |
| `lag <col> [N]` | Lag variable (shift down) | `lag price 2` |
| `lead <col> [N]` | Lead variable (shift up) | `lead price` |
| `pivot <val> by <col>` | Reshape to wide format | `pivot score by subject over name` |
| `melt <ids>, <vals>` | Reshape to long format | `melt name, math eng` |

### Descriptive Statistics

| Command | Description | Example |
|---------|-------------|---------|
| `summarize [cols]` | Summary statistics (N, Mean, SD, quartiles) | `summarize age income` |
| `tabulate <col>` | Frequency table (top 50 values) | `tabulate education` |
| `crosstab <row> <col>` | Two-way contingency table with row percentages | `crosstab gender status` |
| `corr [cols]` | Pearson correlation matrix | `corr age income score` |
| `groupby <cols> summarize <aggs>` | Group-by with aggregations | `groupby region summarize mean(income) count()` |

### Statistical Models

| Command | Description | Example |
|---------|-------------|---------|
| `ols y ~ x1 + x2` | OLS linear regression | `ols score ~ age + income --robust` |
| `logit y ~ x1 + x2` | Logistic regression (binary) | `logit employed ~ age + income` |
| `probit y ~ x1 + x2` | Probit regression (binary) | `probit employed ~ age + income` |
| `poisson y ~ x1 + x2` | Poisson regression (counts) | `poisson visits ~ age --exposure=time` |
| `negbin y ~ x1 + x2` | Negative Binomial (overdispersed) | `negbin claims ~ age + gender` |
| `quantreg y ~ x1 + x2` | Quantile regression | `quantreg wage ~ edu + exp tau=0.9` |

**All models support:** `--robust` (heteroscedasticity-robust SE), `--cluster=col` (cluster-robust SE)

**Formula syntax:**
- `y ~ x1 + x2` — standard predictors
- `y ~ x1*x2` — full factorial (expands to `x1 + x2 + x1:x2`)
- `y ~ x1:x2` — interaction term only
- `y ~ x1*x2*x3` — three-way interaction (all combinations)

### Post-Estimation

| Command | Description | Example |
|---------|-------------|---------|
| `predict [name]` | Predicted values from last model | `predict yhat` |
| `residuals [name]` | Residuals + diagnostic plots | `residuals resid` |
| `vif` | Variance Inflation Factor | `vif` |
| `margins [--at=means\|average]` | Marginal effects (logit/probit) | `margins --at=average` |
| `bootstrap [n=N] [ci=N]` | Bootstrap confidence intervals | `bootstrap n=1000 ci=95` |
| `estat <sub>` | Post-estimation diagnostics | `estat all` |
| `estimates table` | Side-by-side model comparison | `estimates table` |
| `stepwise y ~ x1 + ...` | Stepwise variable selection | `stepwise y ~ x1 + x2 + x3 --backward` |
| `latex [path.tex]` | Export model as LaTeX table | `latex results.tex` |

**`estat` subcommands:**
- `estat hettest` — Breusch-Pagan heteroscedasticity test
- `estat ovtest` — Ramsey RESET specification test
- `estat linktest` — Link test for model specification
- `estat ic` — Information criteria (AIC, BIC, Log-Likelihood)
- `estat all` — Run all diagnostics at once

### Hypothesis Tests

| Command | Description | Example |
|---------|-------------|---------|
| `ttest <col>` | One-sample t-test (H0: mean=0) | `ttest score mu=7` |
| `ttest <col> by <group>` | Two-sample Welch t-test | `ttest income by employed` |
| `ttest <col> paired <col2>` | Paired t-test | `ttest before paired after` |
| `chi2 <col1> <col2>` | Chi-square independence test | `chi2 region employed` |
| `anova <col> by <group>` | One-way ANOVA (F-test) | `anova score by region` |

### Visualization

| Command | Description | Example |
|---------|-------------|---------|
| `plot hist <col>` | Histogram | `plot hist age` |
| `plot scatter <y> <x>` | Scatter plot | `plot scatter score income` |
| `plot line <y> <x>` | Line plot | `plot line score age` |
| `plot box <col> [by <g>]` | Box plot (optionally grouped) | `plot box income by region` |
| `plot bar <col> [by <g>]` | Bar chart | `plot bar income by region` |
| `plot heatmap [cols]` | Correlation heatmap | `plot heatmap age income score` |
| `plot diagnostics` | Residual diagnostic plots | `plot diagnostics` |

### Other

| Command | Description | Example |
|---------|-------------|---------|
| `report <path>` | Generate Markdown report | `report analysis.md` |
| `help [cmd]` | Show help (all or specific command) | `help ols` |
| `quit` / `exit` / `q` | Exit REPL | `quit` |

---

## Expression Language

The expression language used by `filter` and `derive` is a **safe, recursive-descent parser** — no Python `eval()` is ever used.

```bash
# Arithmetic
openstat> derive income_k = income / 1000
openstat> derive bmi = weight / (height ** 2)

# Comparisons and boolean logic
openstat> filter age > 30 and income < 50000
openstat> filter not is_null(score) and region == "North"

# Functions
openstat> derive log_income = log(income)
openstat> derive name_upper = upper(name)
openstat> derive score_clean = fill_null(score, 0)
```

**Available functions:**

| Category | Functions |
|----------|----------|
| Math | `log(x)`, `log10(x)`, `sqrt(x)`, `abs(x)`, `exp(x)`, `round(x, n)` |
| String | `upper(x)`, `lower(x)`, `len_chars(x)`, `strip(x)`, `contains(x, "pattern")` |
| Null | `is_null(x)`, `is_not_null(x)`, `fill_null(x, value)` |
| Type | `cast_float(x)`, `cast_int(x)`, `cast_str(x)` |

---

## Statistical Models — In Depth

### Automatic Diagnostics

Every model automatically checks for common problems and warns you:

- **Multicollinearity** — Condition number > 30 triggers a warning
- **Heteroscedasticity** — Breusch-Pagan test; suggests `--robust` if p < 0.05
- **Autocorrelation** — Durbin-Watson statistic far from 2.0
- **Convergence** — Warns if logit/probit MLE did not converge
- **Missing values** — Reports how many observations were dropped
- **Low sample size** — Warns when observation-to-predictor ratio is low

### Interaction Terms

```bash
# Full factorial: automatically expands to x1 + x2 + x1:x2
openstat> ols y ~ x1*x2

# Interaction only
openstat> ols y ~ x1 + x2 + x1:x2

# Three-way interaction (7 terms total)
openstat> ols y ~ x1*x2*x3
```

### Cluster-Robust Standard Errors

```bash
# Clustered at the region level
openstat> ols wage ~ education + experience --cluster=region

# Works with all model types
openstat> logit promoted ~ age + performance --cluster=department
```

### Marginal Effects

After fitting a logit or probit model, compute marginal effects to understand the practical impact:

```bash
openstat> logit employed ~ age + education + income
openstat> margins                    # Average marginal effects (default)
openstat> margins --at=means         # Marginal effects at means
```

### Bootstrap Confidence Intervals

Non-parametric bootstrap for any model — no distributional assumptions needed:

```bash
openstat> ols y ~ x1 + x2
openstat> bootstrap n=1000 ci=95     # 1000 replications, 95% CI
openstat> bootstrap n=5000 ci=99     # More replications, 99% CI
```

Bootstrap uses thread-pool parallelism for speed when n > 100.

### Model Comparison

Run multiple models and compare them side-by-side:

```bash
openstat> ols y ~ x1
openstat> ols y ~ x1 + x2
openstat> ols y ~ x1 + x2 + x1:x2
openstat> estimates table
```

This produces a publication-ready comparison table with coefficients, standard errors, R², AIC, BIC, and more.

---

## File Formats

| Format | Import | Export | Dependency |
|--------|:------:|:------:|------------|
| CSV | Yes | Yes | Built-in |
| Parquet | Yes | Yes | Built-in |
| Stata (.dta) | Yes | Yes | `pip install openstat[stata]` |
| Excel (.xlsx) | Yes | Yes | `pip install openstat[excel]` |

---

## Configuration

Customize OpenStat by creating `~/.openstat/config.toml`:

```toml
[data]
output_dir = "outputs"
csv_separator = ","

[display]
tabulate_limit = 50
head_default = 10

[undo]
max_undo_stack = 20
max_undo_memory_mb = 500

[plotting]
plot_dpi = 150
plot_figsize_w = 8.0
plot_figsize_h = 5.0

[model]
condition_threshold = 30
min_obs_per_predictor = 5
bootstrap_iterations = 1000
```

---

## CLI Options

```bash
openstat repl                     # Interactive REPL
openstat run script.ost           # Run a script
openstat run script.ost --strict  # Stop on first error (exit code 1)
openstat --verbose repl           # Verbose logging (INFO)
openstat --debug repl             # Debug logging (DEBUG)
openstat --version                # Show version
```

Logs are saved to `~/.openstat/logs/openstat.log`.

---

## Aggregation Functions

For use with `groupby ... summarize`:

| Function | Description |
|----------|-------------|
| `mean(col)` | Arithmetic mean |
| `sd(col)` | Standard deviation (sample) |
| `sum(col)` | Sum |
| `min(col)` | Minimum |
| `max(col)` | Maximum |
| `median(col)` | Median |
| `count()` | Row count per group |

---

## Technology Stack

OpenStat is built on best-in-class open-source libraries:

| Component | Library | Why |
|-----------|---------|-----|
| Data Engine | [Polars](https://pola.rs/) | 10-100x faster than pandas, zero-copy, Rust-powered |
| Statistics | [statsmodels](https://www.statsmodels.org/) | Industry-standard OLS, GLM, quantile regression |
| Scientific | [SciPy](https://scipy.org/) | Hypothesis tests, distributions |
| Plotting | [matplotlib](https://matplotlib.org/) | Publication-quality figures |
| CLI Framework | [Typer](https://typer.tiangolo.com/) | Beautiful CLI with zero boilerplate |
| Terminal UI | [Rich](https://github.com/Textualize/rich) | Gorgeous tables and formatting |
| REPL | [prompt-toolkit](https://python-prompt-toolkit.readthedocs.io/) | Tab completion, history, syntax |

---

## Project Structure

```
OpenStat/
├── src/openstat/
│   ├── cli.py              # Typer CLI entry point
│   ├── repl.py             # Interactive REPL with tab completion
│   ├── session.py          # Session state, undo system
│   ├── config.py           # Configuration management (~/.openstat/config.toml)
│   ├── commands/
│   │   ├── base.py         # @command decorator, registry, CommandArgs
│   │   ├── data_cmds.py    # load, filter, select, derive, sort, merge, ...
│   │   ├── stat_cmds.py    # summarize, ols, logit, poisson, margins, ...
│   │   ├── plot_cmds.py    # plot hist/scatter/line/box/bar/heatmap
│   │   └── report_cmds.py  # report, help
│   ├── dsl/
│   │   ├── tokenizer.py    # Safe expression tokenizer
│   │   └── parser.py       # Recursive descent parser (no eval!)
│   ├── stats/
│   │   └── models.py       # OLS, Logit, Probit, Poisson, NegBin, QuantReg, ...
│   ├── plots/
│   │   └── plotter.py      # matplotlib chart generation
│   ├── io/
│   │   └── loader.py       # CSV, Parquet, DTA, Excel loaders
│   └── reporting/
│       └── report.py       # Markdown report generator
├── tests/                  # 343 tests (and growing!)
├── examples/
│   ├── data.csv            # Sample dataset
│   └── demo.ost            # Demo script showcasing all features
├── .github/workflows/
│   └── ci.yml              # GitHub Actions: test on 4 Python versions x 2 OS
├── pyproject.toml
├── LICENSE                 # MIT
├── CONTRIBUTING.md
└── README.md
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run the full test suite
pytest

# Run with coverage
pytest --cov=openstat --cov-report=term-missing

# Run a specific test file
pytest tests/test_v020.py -v

# Lint
pip install ruff
ruff check src/ tests/
```

**Current test status:** 343 tests passed, 0 failures across 11 test files.

---

## Contributing

**We love contributions!** Whether you're fixing a typo, adding a new command, improving documentation, or building an entire new feature — your contribution matters and is deeply appreciated.

OpenStat is built by the community, for the community. Every contribution makes statistical analysis more accessible to researchers, students, and data scientists around the world.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write** your code and tests
4. **Ensure** all tests pass (`pytest`) and lint is clean (`ruff check src/`)
5. **Submit** a Pull Request with a clear description

### What Can You Contribute?

- **New statistical methods** — Panel data, time series, survival analysis, mixed models
- **New commands** — Any data manipulation or analysis command you find useful
- **DSL functions** — Add functions to the expression language
- **Plot types** — New visualization types
- **Documentation** — Tutorials, examples, translations
- **Bug reports** — Found something that doesn't work? Open an issue!
- **Performance** — Found a bottleneck? We'd love a PR!
- **File formats** — Support for more data formats (SAS, SPSS, etc.)

### First-Time Contributors Welcome!

Never contributed to open source before? No problem! Look for issues labeled `good first issue`. We're happy to mentor and guide you through the process. Every expert was once a beginner.

Check out [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup instructions and coding guidelines.

---

## Roadmap

We have big plans for OpenStat. Here's what's coming:

### Completed

- [x] OLS, Logit, Probit regression
- [x] Interaction terms (`x1*x2`, `x1:x2`, multi-way)
- [x] Cluster-robust standard errors
- [x] Poisson & Negative Binomial regression
- [x] Quantile regression
- [x] Marginal effects (average, at-means)
- [x] Bootstrap confidence intervals (parallelized)
- [x] Post-estimation diagnostics (`estat`)
- [x] Model comparison tables (`estimates table`)
- [x] Stepwise variable selection (forward/backward)
- [x] Robust standard errors (HC1)
- [x] Residual diagnostics with plots
- [x] VIF multicollinearity check
- [x] LaTeX table export
- [x] Data joining/merging
- [x] Pivot/melt reshaping
- [x] Safe expression language (no eval)
- [x] Tab completion in REPL
- [x] Configuration file support
- [x] Multi-level undo with memory management
- [x] CI/CD with GitHub Actions

### Planned

- [ ] Panel data / fixed effects / random effects
- [ ] Time series analysis (ARIMA, VAR)
- [ ] Survival analysis (Cox PH, Kaplan-Meier)
- [ ] Mixed / hierarchical linear models
- [ ] Instrumental variables (2SLS, IV)
- [ ] DuckDB / LazyFrame backend for large datasets
- [ ] Plugin / extension system
- [ ] Web-based GUI
- [ ] Jupyter notebook integration
- [ ] SAS (.sas7bdat) and SPSS (.sav) file support
- [ ] Multiple imputation for missing data
- [ ] Survey weighting support

---

## Community

OpenStat is more than code — it's a community of people who believe that statistical tools should be free and open. If you use OpenStat in your research, teaching, or work, we'd love to hear about it!

- **Star this repo** if you find it useful — it helps others discover the project
- **Share** with colleagues, students, and fellow researchers
- **Open issues** for bugs, feature requests, or questions
- **Join the conversation** in GitHub Discussions

---

## Acknowledgements

OpenStat stands on the shoulders of giants. Huge thanks to the maintainers and contributors of:

- [Polars](https://pola.rs/) — for reimagining what a DataFrame library can be
- [statsmodels](https://www.statsmodels.org/) — for bringing professional statistics to Python
- [SciPy](https://scipy.org/) — for decades of scientific computing excellence
- [Rich](https://github.com/Textualize/rich) — for making terminal output beautiful
- [prompt-toolkit](https://python-prompt-toolkit.readthedocs.io/) — for the interactive REPL foundation

And to every researcher, student, and data scientist who believes in open science. This project is for you.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Free as in freedom. Free as in beer. Use it, modify it, share it, sell it — no restrictions.

---

<p align="center">
  <strong>If OpenStat helps your work, give it a star! Every star helps more people discover free statistical tools.</strong>
</p>

<p align="center">
  Made with care for the open-source community.
</p>
