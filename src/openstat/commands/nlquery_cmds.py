"""Natural language query: 'ask' command using OpenAI or Anthropic API."""

from __future__ import annotations

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


def _build_context(session: Session) -> str:
    """Build a compact dataset context string for the LLM."""
    import polars as pl

    if session.df is None:
        return "No dataset loaded."

    df = session.df
    NUMERIC = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

    col_info = []
    for c in df.columns[:30]:  # cap at 30 cols
        dtype = str(df[c].dtype)
        n_miss = df[c].null_count()
        if df[c].dtype in NUMERIC:
            col_data = df[c].drop_nulls()
            if col_data.len() > 0:
                extra = f"mean={col_data.mean():.2f}, sd={col_data.std():.2f}"
            else:
                extra = "all null"
        else:
            n_uniq = df[c].drop_nulls().n_unique()
            extra = f"{n_uniq} unique values"
        miss_str = f", {n_miss} missing" if n_miss else ""
        col_info.append(f"  {c} ({dtype}{miss_str}): {extra}")

    lines = [
        f"Dataset: {session.dataset_name or 'unknown'}",
        f"Shape: {df.height} rows × {df.width} columns",
        "Columns:",
    ] + col_info

    if session.results:
        lines.append("\nLast model: " + session.results[-1].name +
                     " — " + session.results[-1].formula)

    return "\n".join(lines)


def _ask_openai(question: str, context: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        return None  # type: ignore

    client = OpenAI()  # uses OPENAI_API_KEY env var
    resp = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful statistical analysis assistant for OpenStat, "
                    "a Python-based data analysis REPL similar to Stata. "
                    "Answer questions about the dataset concisely. "
                    "When relevant, suggest the exact OpenStat command to use."
                ),
            },
            {
                "role": "user",
                "content": f"Dataset context:\n{context}\n\nQuestion: {question}",
            },
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def _ask_anthropic(question: str, context: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        return None  # type: ignore

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    resp = client.messages.create(
        model=model or "claude-haiku-4-5-20251001",
        max_tokens=500,
        system=(
            "You are a helpful statistical analysis assistant for OpenStat, "
            "a Python-based data analysis REPL similar to Stata. "
            "Answer questions about the dataset concisely. "
            "When relevant, suggest the exact OpenStat command to use."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Dataset context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return resp.content[0].text.strip()


@command("ask", usage='ask "<natural language question>"')
def cmd_ask(session: Session, args: str) -> str:
    """Ask a natural language question about your dataset (requires AI API key).

    Uses OpenAI (OPENAI_API_KEY) or Anthropic (ANTHROPIC_API_KEY), whichever
    is available. The assistant can suggest OpenStat commands, explain results,
    and answer statistical questions.

    Options:
      --provider=openai|anthropic   (default: auto-detect from env)
      --model=<model-name>          (default: gpt-4o-mini / claude-haiku-4-5)

    Examples:
      ask "What's the correlation between income and education?"
      ask "Which variables have the most missing data?"
      ask "What regression model should I use for this binary outcome?"
      ask "Explain the OLS results" --provider=anthropic
    """
    import os

    ca = CommandArgs(args)
    # Question is the rest of args after stripping flags
    question = ca.strip_flags_and_options().strip().strip('"\'')
    if not question:
        return 'Usage: ask "<question>" [--provider=openai|anthropic] [--model=<name>]'

    provider = ca.options.get("provider", "").lower()
    model = ca.options.get("model", "")

    context = _build_context(session)

    # Auto-detect provider
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            return (
                "No AI API key found.\n"
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.\n"
                "Install: pip install openai  OR  pip install anthropic"
            )

    try:
        if provider == "openai":
            result = _ask_openai(question, context, model)
            if result is None:
                return "openai package not installed. Run: pip install openai"
        elif provider == "anthropic":
            result = _ask_anthropic(question, context, model)
            if result is None:
                return "anthropic package not installed. Run: pip install anthropic"
        else:
            return f"Unknown provider: {provider}. Use 'openai' or 'anthropic'."
        return result
    except Exception as e:
        return friendly_error(e, "ask")
