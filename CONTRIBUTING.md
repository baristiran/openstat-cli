# Contributing to OpenStat

Thanks for your interest in contributing! OpenStat is an open-source statistical analysis tool and we welcome contributions of all kinds.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/openstat.git
cd openstat

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode with dev deps
pip install -e ".[dev]"

# Run tests
pytest
```

## Project Structure

```
src/openstat/
├── cli.py              # Typer CLI entry point
├── repl.py             # Interactive REPL with tab completion
├── session.py          # Session state, undo system
├── commands/
│   ├── base.py         # @command decorator, registry
│   ├── data_cmds.py    # load, filter, select, derive, sort, ...
│   ├── stat_cmds.py    # summarize, tabulate, corr, ols, logit, ...
│   ├── plot_cmds.py    # plot hist/scatter/line/box
│   └── report_cmds.py  # report, help
├── dsl/
│   ├── tokenizer.py    # Safe expression tokenizer
│   └── parser.py       # Recursive descent parser (no eval!)
├── stats/
│   └── models.py       # OLS, Logit via statsmodels
├── plots/
│   └── plotter.py      # matplotlib chart generation
├── io/
│   └── loader.py       # CSV, Parquet, DTA, Excel loaders
└── reporting/
    └── report.py       # Markdown report generator
```

## Adding a New Command

1. Pick the right module in `src/openstat/commands/` (or create a new one).
2. Use the `@command` decorator:

```python
from openstat.commands.base import command

@command("mycommand", usage="mycommand <arg>")
def cmd_mycommand(session, args):
    """One-line description shown in help."""
    df = session.require_data()
    # ... your logic ...
    return "Result text shown to user"
```

3. If you created a new module, import it in `src/openstat/commands/__init__.py`.
4. Add tests in `tests/`.

## Adding a DSL Function

To add a new function to the expression language (used by `filter` and `derive`):

1. Edit `src/openstat/dsl/parser.py`
2. Add a case in `_apply_function()`
3. Add a test in `tests/test_parser.py`

## Guidelines

- **No `eval()`** — all user expressions go through the safe parser
- **Snapshot before mutation** — call `session.snapshot()` before modifying `session.df`
- **Return strings** — command handlers return plain text (use `rich_to_str()` for Rich tables)
- **Friendly errors** — use `friendly_error()` to wrap exceptions
- **Test real values** — assert on actual numbers, not just "contains some string"

## Running Tests

```bash
# Full suite
pytest

# Verbose with specific file
pytest tests/test_commands.py -v

# With coverage
pytest --cov=openstat --cov-report=term-missing
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting. Before submitting:

```bash
pip install ruff
ruff check src/ tests/
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Write tests for your changes
3. Ensure all tests pass (`pytest`)
4. Run `ruff check` with no errors
5. Submit a PR with a clear description of what and why

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
