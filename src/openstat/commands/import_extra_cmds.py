"""Extra import commands: URL, clipboard, SPSS syntax translation, REST webhook."""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
from pathlib import Path

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ---------------------------------------------------------------------------
# import url
# ---------------------------------------------------------------------------

@command("import url", usage="import url <url> [--format=csv|json|parquet] [--sep=,]")
def cmd_import_url(session: Session, args: str) -> str:
    """Load data from an HTTP/HTTPS URL.

    Auto-detects format from the URL extension if --format is not supplied.
    Supported formats: csv, json, parquet.

    Example: import url https://example.com/data.csv
    Example: import url https://api.example.com/data.json --format=json
    """
    import polars as pl
    from openstat.io.loader import load_file

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: import url <url> [--format=csv|json|parquet] [--sep=,]"

    url = ca.positional[0]
    fmt = ca.options.get("format", "").lower()
    sep = ca.options.get("sep", ",")

    # Auto-detect format from extension
    if not fmt:
        url_path = url.split("?")[0].lower()
        if url_path.endswith(".parquet"):
            fmt = "parquet"
        elif url_path.endswith(".json") or url_path.endswith(".jsonl"):
            fmt = "json"
        else:
            fmt = "csv"

    # Download to a temporary file
    suffix = f".{fmt}"
    try:
        try:
            import requests
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data_bytes = resp.content
        except ImportError:
            import urllib.request as _urllib
            with _urllib.urlopen(url, timeout=60) as resp:  # noqa: S310
                data_bytes = resp.read()
    except Exception as exc:
        return f"Failed to download '{url}': {exc}"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data_bytes)
            tmp_path = tmp.name

        if fmt == "parquet":
            df = pl.read_parquet(tmp_path)
        elif fmt == "json":
            df = pl.read_json(tmp_path)
        else:
            actual_sep = "\t" if sep in ("\\t", "\t") else sep
            df = pl.read_csv(tmp_path, separator=actual_sep)
    except Exception as exc:
        return friendly_error(exc, "import url")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    session.snapshot()
    session.df = df
    session.dataset_path = url
    session.dataset_name = url.split("/")[-1].split("?")[0] or "url_import"
    session._undo_stack.clear()
    r, c = df.shape
    return f"Loaded {r:,} rows x {c} columns from {url}"


# ---------------------------------------------------------------------------
# import clipboard
# ---------------------------------------------------------------------------

@command("import clipboard", usage="import clipboard [--sep=\\t|,]")
def cmd_import_clipboard(session: Session, args: str) -> str:
    """Load tabular data pasted from a spreadsheet (Excel, Google Sheets, etc.).

    The default separator is a tab character, which is what Excel/Sheets
    copies to the clipboard. Use --sep=, for comma-separated text.

    Example: import clipboard
    Example: import clipboard --sep=,
    """
    import polars as pl

    ca = CommandArgs(args)
    sep_raw = ca.options.get("sep", "\\t")
    sep = "\t" if sep_raw in ("\\t", "\t") else sep_raw

    # Retrieve clipboard contents
    text: str | None = None

    try:
        import pyperclip
        text = pyperclip.paste()
    except ImportError:
        pass

    if text is None:
        # Fallback: platform-specific subprocess
        try:
            import subprocess
            platform = sys.platform
            if platform == "darwin":
                result = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=5)
                text = result.stdout
            elif platform.startswith("linux"):
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode != 0:
                    result = subprocess.run(
                        ["xsel", "--clipboard", "--output"],
                        capture_output=True, text=True, timeout=5,
                    )
                text = result.stdout
            elif platform == "win32":
                result = subprocess.run(
                    ["powershell", "-command", "Get-Clipboard"],
                    capture_output=True, text=True, timeout=5,
                )
                text = result.stdout
        except Exception as exc:
            return (
                f"Could not read clipboard: {exc}\n"
                "Install pyperclip for reliable clipboard support: pip install pyperclip"
            )

    if not text or not text.strip():
        return "Clipboard is empty or contains no text."

    try:
        df = pl.read_csv(
            io.StringIO(text),
            separator=sep,
            infer_schema_length=1000,
        )
    except Exception as exc:
        return friendly_error(exc, "import clipboard")

    session.snapshot()
    session.df = df
    session.dataset_path = "clipboard"
    session.dataset_name = "clipboard"
    session._undo_stack.clear()
    r, c = df.shape
    sep_display = "tab" if sep == "\t" else repr(sep)
    return f"Loaded {r:,} rows x {c} columns from clipboard (sep={sep_display})."


# ---------------------------------------------------------------------------
# import spss
# ---------------------------------------------------------------------------

# Mapping of SPSS commands to OpenStat equivalents.
# Each entry: (regex_pattern, replacement_template | None)
# None means we emit an [untranslated] comment.

_SPSS_RULES: list[tuple[re.Pattern, str | None]] = [
    # GET FILE = 'path/to/data.sav'.
    (
        re.compile(r"^GET\s+FILE\s*=\s*['\"]?([^'\".\s]+\S*?)['\"]?\s*\.?\s*$", re.IGNORECASE),
        r"load \1",
    ),
    # SAVE OUTFILE = 'path'.
    (
        re.compile(r"^SAVE\s+OUTFILE\s*=\s*['\"]?([^'\".\s]+\S*?)['\"]?\s*\.?\s*$", re.IGNORECASE),
        r"save \1",
    ),
    # REGRESSION VARIABLES = y x1 x2 / DEPENDENT y ...
    (
        re.compile(
            r"^REGRESSION\s+.*VARIABLES\s*=\s*([\w\s]+?)(?:\s*/.*)?\.?\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        None,  # complex to auto-translate; emit comment
    ),
    # FREQUENCIES VARIABLES = col1 col2.
    (
        re.compile(r"^FREQUENCIES\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", re.IGNORECASE),
        None,  # -> tabulate (needs single col); emit comment
    ),
    # DESCRIPTIVES VARIABLES = col1 col2.
    (
        re.compile(r"^DESCRIPTIVES\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", re.IGNORECASE),
        None,
    ),
    # CORRELATIONS VARIABLES = col1 col2.
    (
        re.compile(r"^CORRELATIONS\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", re.IGNORECASE),
        None,
    ),
    # COMPUTE newvar = expression.
    (
        re.compile(r"^COMPUTE\s+(\w+)\s*=\s*(.+?)\.?\s*$", re.IGNORECASE),
        r"generate \1 = \2",
    ),
    # SELECT IF (condition).
    (
        re.compile(r"^SELECT\s+IF\s+\((.+?)\)\.?\s*$", re.IGNORECASE),
        None,
    ),
    # RECODE var (old=new) ...
    (
        re.compile(r"^RECODE\s+.+$", re.IGNORECASE),
        None,
    ),
]

# Simple direct substitution rules (SPSS keyword -> OpenStat command prefix)
_SPSS_SIMPLE: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"^FREQUENCIES\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", re.IGNORECASE),
        "tabulate",
    ),
    (
        re.compile(r"^DESCRIPTIVES\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", re.IGNORECASE),
        "summarize",
    ),
    (
        re.compile(r"^CORRELATIONS\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", re.IGNORECASE),
        "correlate",
    ),
]


def _translate_spss_line(line: str) -> str:
    """Translate a single SPSS syntax line to an OpenStat line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("*") or stripped.startswith("/*"):
        # SPSS comment
        return f"# {stripped.lstrip('*').strip()}" if stripped else ""

    # GET FILE
    m = re.match(r"^GET\s+FILE\s*=\s*['\"]?([^'\".\s]+\S*?)['\"]?\s*\.?\s*$", stripped, re.IGNORECASE)
    if m:
        return f"load {m.group(1)}"

    # SAVE OUTFILE
    m = re.match(r"^SAVE\s+OUTFILE\s*=\s*['\"]?([^'\".\s]+\S*?)['\"]?\s*\.?\s*$", stripped, re.IGNORECASE)
    if m:
        return f"save {m.group(1)}"

    # COMPUTE newvar = expr.
    m = re.match(r"^COMPUTE\s+(\w+)\s*=\s*(.+?)\.?\s*$", stripped, re.IGNORECASE)
    if m:
        return f"generate {m.group(1)} = {m.group(2)}"

    # FREQUENCIES VARIABLES = ...
    m = re.match(r"^FREQUENCIES\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", stripped, re.IGNORECASE)
    if m:
        first_col = m.group(1).split()[0]
        rest = m.group(1).split()[1:]
        extra = " ".join(rest)
        note = f"  # [note] original cols: {extra}" if extra else ""
        return f"tabulate {first_col}{note}"

    # DESCRIPTIVES VARIABLES = ...
    m = re.match(r"^DESCRIPTIVES\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", stripped, re.IGNORECASE)
    if m:
        cols = m.group(1).strip()
        return f"summarize {cols}"

    # CORRELATIONS VARIABLES = ...
    m = re.match(r"^CORRELATIONS\s+VARIABLES\s*=\s*([\w\s]+?)\.?\s*$", stripped, re.IGNORECASE)
    if m:
        cols = m.group(1).strip()
        return f"correlate {cols}"

    # REGRESSION — complex, emit annotated comment + best-effort ols
    m = re.match(
        r"^REGRESSION\s+.*?DEPENDENT\s*=?\s*(\w+)\s+.*?ENTER\s+([\w\s]+?)\.?\s*$",
        stripped, re.IGNORECASE | re.DOTALL,
    )
    if m:
        dep = m.group(1)
        indeps = m.group(2).strip()
        return f"ols {dep} {indeps}  # [translated from REGRESSION]"

    # SELECT IF
    m = re.match(r"^SELECT\s+IF\s+\((.+?)\)\.?\s*$", stripped, re.IGNORECASE)
    if m:
        cond = m.group(1)
        return (
            f"# [untranslated] SELECT IF ({cond})\n"
            f"# Manual equivalent: filter {cond}"
        )

    # RECODE
    if re.match(r"^RECODE\b", stripped, re.IGNORECASE):
        return (
            f"# [untranslated] {stripped}\n"
            "# Manual equivalent: recode <col> old=new ..."
        )

    # Anything else
    return f"# [untranslated] {stripped}"


@command("import spss", usage="import spss <syntax.sps> [--out=<script.ost>] [--run]")
def cmd_import_spss(session: Session, args: str) -> str:
    """Translate an SPSS syntax file (.sps) into an OpenStat script (.ost).

    Translated commands:
      GET FILE        -> load
      SAVE OUTFILE    -> save
      COMPUTE         -> generate
      FREQUENCIES     -> tabulate
      DESCRIPTIVES    -> summarize
      CORRELATIONS    -> correlate
      REGRESSION      -> ols  (best-effort)
      SELECT IF       -> filter (emitted as comment with hint)
      RECODE          -> replace (emitted as comment with hint)
    All other lines are kept as [untranslated] comments.

    Use --run to immediately execute the translated script.

    Example: import spss analysis.sps --out=analysis.ost
    """
    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: import spss <syntax.sps> [--out=<script.ost>] [--run]"

    sps_path = ca.positional[0]
    out_path = ca.options.get("out")
    do_run = ca.has_flag("--run")

    try:
        raw = Path(sps_path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"File not found: {sps_path}"
    except Exception as exc:
        return f"Cannot read '{sps_path}': {exc}"

    # SPSS statements can span multiple lines ending with '.'; we join continuation lines.
    # Simple heuristic: lines not ending with '.' that are not blank are joined to the next.
    joined_lines: list[str] = []
    buffer = ""
    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if buffer:
                joined_lines.append(buffer)
                buffer = ""
            joined_lines.append("")
            continue
        buffer = (buffer + " " + stripped).strip() if buffer else stripped
        if buffer.endswith("."):
            joined_lines.append(buffer.rstrip("."))
            buffer = ""
    if buffer:
        joined_lines.append(buffer)

    ost_lines: list[str] = [
        f"# OpenStat script translated from SPSS: {sps_path}",
        "#",
    ]
    untranslated_count = 0
    translated_count = 0

    for line in joined_lines:
        if not line.strip():
            ost_lines.append("")
            continue
        translated = _translate_spss_line(line)
        ost_lines.append(translated)
        if "[untranslated]" in translated:
            untranslated_count += 1
        else:
            translated_count += 1

    ost_text = "\n".join(ost_lines)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(ost_text, encoding="utf-8")
        result_msg = (
            f"Translated '{sps_path}' -> '{out_path}'\n"
            f"  Translated: {translated_count} lines, "
            f"untranslated (kept as comments): {untranslated_count} lines."
        )
    else:
        result_msg = (
            f"# --- Translated output (not saved) ---\n"
            f"{ost_text}\n"
            f"# --- Translated: {translated_count}, untranslated: {untranslated_count} ---"
        )

    if do_run and out_path:
        try:
            from openstat.script_runner import run_script
            run_script(session, Path(out_path))
            result_msg += "\nScript executed."
        except Exception as exc:
            result_msg += f"\nScript execution failed: {exc}"

    return result_msg


# ---------------------------------------------------------------------------
# webhook
# ---------------------------------------------------------------------------

def _extract_json_path(data: object, dotpath: str) -> object:
    """Traverse a dotted path like 'data.records' through nested dicts/lists."""
    parts = dotpath.strip().split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            return None
        if current is None:
            return None
    return current


@command("webhook", usage="webhook <url> [--method=GET|POST] [--params=key:val,key:val]")
def cmd_webhook(session: Session, args: str) -> str:
    """Fetch data from a REST API endpoint and load as a DataFrame.

    Options:
      --method=GET|POST          HTTP method (default GET).
      --params=key:val,key:val   Request parameters (GET query string or POST body).
      --token=<bearer_token>     Authorization: Bearer <token> header.
      --json_path=<dotpath>      Navigate into nested JSON (e.g. data.records).

    The response must be JSON. Arrays of objects are loaded directly;
    nested structures are extracted with --json_path.

    Example: webhook https://api.example.com/records
    Example: webhook https://api.example.com/search --method=POST --params=q:hello,limit:100
    Example: webhook https://api.example.com/v2/items --token=abc123 --json_path=data.items
    """
    import polars as pl

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: webhook <url> [--method=GET|POST] [--params=key:val,key:val]"

    url = ca.positional[0]
    method = ca.options.get("method", "GET").upper()
    token = ca.options.get("token")
    json_path = ca.options.get("json_path")
    params_raw = ca.options.get("params", "")

    if method not in ("GET", "POST"):
        return f"Unsupported method '{method}'. Use GET or POST."

    # Parse params: key:val,key:val
    params: dict[str, str] = {}
    if params_raw:
        for pair in params_raw.split(","):
            pair = pair.strip()
            if ":" not in pair:
                return f"Invalid param '{pair}'. Use key:val format."
            k, v = pair.split(":", 1)
            params[k.strip()] = v.strip()

    # Build headers
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        try:
            import requests
            if method == "GET":
                resp = requests.get(url, params=params or None, headers=headers, timeout=60)
            else:
                resp = requests.post(url, json=params or None, headers=headers, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
        except ImportError:
            # Fall back to urllib for GET without params easily serialisable
            import urllib.request as _urllib
            import urllib.parse as _urlparse
            import json as _json

            full_url = url
            if method == "GET" and params:
                full_url = url + "?" + _urlparse.urlencode(params)

            req = _urllib.Request(full_url, headers=headers)  # noqa: S310
            if method == "POST" and params:
                import json as _json2
                body = _json2.dumps(params).encode()
                req = _urllib.Request(full_url, data=body, headers={**headers, "Content-Type": "application/json"})  # noqa: S310

            with _urllib.urlopen(req, timeout=60) as resp:  # noqa: S310
                payload = _json.loads(resp.read().decode("utf-8"))

    except Exception as exc:
        return f"Request to '{url}' failed: {exc}"

    # Navigate to nested path if requested
    if json_path:
        payload = _extract_json_path(payload, json_path)
        if payload is None:
            return f"json_path '{json_path}' not found in response."

    # Convert to DataFrame
    try:
        if isinstance(payload, list):
            if not payload:
                return "API returned an empty array."
            df = pl.DataFrame(payload)
        elif isinstance(payload, dict):
            # Try to find an array field automatically
            array_fields = [k for k, v in payload.items() if isinstance(v, list)]
            if not array_fields:
                # Treat the dict as a single-row DataFrame
                df = pl.DataFrame([payload])
            elif len(array_fields) == 1:
                df = pl.DataFrame(payload[array_fields[0]])
            else:
                # Multiple arrays — ask user to specify
                return (
                    f"Response contains multiple array fields: {', '.join(array_fields)}.\n"
                    f"Use --json_path=<field> to select one. Example: --json_path={array_fields[0]}"
                )
        else:
            return f"Unexpected JSON type: {type(payload).__name__}. Expected list or object."
    except Exception as exc:
        return friendly_error(exc, "webhook")

    session.snapshot()
    session.df = df
    session.dataset_path = url
    session.dataset_name = url.split("/")[-1].split("?")[0] or "webhook"
    session._undo_stack.clear()
    r, c = df.shape
    return f"Loaded {r:,} rows x {c} columns from {url} [{method}]."
