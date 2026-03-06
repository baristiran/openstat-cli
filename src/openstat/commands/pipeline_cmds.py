"""Pipeline, batch processing, git integration, progress bars, config profiles."""

from __future__ import annotations

import os
from pathlib import Path

from openstat.commands.base import command, CommandArgs, friendly_error
from openstat.session import Session


# ── Pipeline ─────────────────────────────────────────────────────────────────

@command("pipeline", usage="pipeline define|run|list|show|rm <name> [commands...]")
def cmd_pipeline(session: Session, args: str) -> str:
    """Define and run named command pipelines.

    A pipeline is a named sequence of commands that can be run in one go.
    Pipelines are stored in the session and can be saved via 'session save'.

    Sub-commands:
      pipeline define <name> <cmd1> | <cmd2> | ...  — define a pipeline
      pipeline run <name>                            — run a defined pipeline
      pipeline list                                  — list all pipelines
      pipeline show <name>                           — show pipeline steps
      pipeline rm <name>                             — remove a pipeline

    Examples:
      pipeline define clean "drop id" | "rename old_name new_name" | "generate x2 = x*x"
      pipeline run clean
      pipeline define model_run "ols y x1 x2" | "plot coef" | "export pdf"
      pipeline run model_run
    """
    _PIPELINES: dict = getattr(session, "_pipelines", {})
    if not hasattr(session, "_pipelines"):
        session._pipelines = _PIPELINES  # type: ignore[attr-defined]

    tokens = args.strip().split(None, 2)
    if not tokens:
        return "Usage: pipeline define|run|list|show|rm <name> [steps]"
    subcmd = tokens[0].lower()

    if subcmd == "list":
        if not _PIPELINES:
            return "No pipelines defined. Use: pipeline define <name> <cmd> | <cmd>"
        lines = ["Defined pipelines:"]
        for name, steps in _PIPELINES.items():
            lines.append(f"  {name} ({len(steps)} steps)")
        return "\n".join(lines)

    if subcmd == "rm":
        name = tokens[1] if len(tokens) > 1 else ""
        if not name:
            return "Usage: pipeline rm <name>"
        if name in _PIPELINES:
            del _PIPELINES[name]
            return f"Pipeline '{name}' removed."
        return f"Pipeline '{name}' not found."

    if subcmd == "show":
        name = tokens[1] if len(tokens) > 1 else ""
        if name not in _PIPELINES:
            return f"Pipeline '{name}' not found."
        steps = _PIPELINES[name]
        lines = [f"Pipeline '{name}' ({len(steps)} steps):"]
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)

    if subcmd == "define":
        if len(tokens) < 3:
            return "Usage: pipeline define <name> <cmd1> | <cmd2> | ..."
        name = tokens[1]
        rest = tokens[2]
        steps = [s.strip().strip('"\'') for s in rest.split("|")]
        steps = [s for s in steps if s]
        _PIPELINES[name] = steps
        return f"Pipeline '{name}' defined with {len(steps)} steps."

    if subcmd == "run":
        name = tokens[1] if len(tokens) > 1 else ""
        if not name:
            return "Usage: pipeline run <name>"
        if name not in _PIPELINES:
            return f"Pipeline '{name}' not found. Use 'pipeline list' to see available."
        steps = _PIPELINES[name]
        from openstat.commands.base import run_command
        results = [f"Running pipeline '{name}' ({len(steps)} steps):", ""]
        for i, step in enumerate(steps, 1):
            results.append(f"[{i}/{len(steps)}] {step}")
            try:
                out = run_command(session, step)
                if out:
                    results.append(f"  → {out[:200]}")
            except Exception as exc:
                results.append(f"  → ERROR: {exc}")
                results.append(f"Pipeline stopped at step {i}.")
                break
        results.append("")
        results.append("Pipeline complete.")
        return "\n".join(results)

    return f"Unknown pipeline sub-command: {subcmd}"


# ── Batch ────────────────────────────────────────────────────────────────────

@command("batch", usage="batch <script1.ost> [script2.ost ...] [--stop-on-error]")
def cmd_batch(session: Session, args: str) -> str:
    """Run multiple script files in sequence.

    Options:
      --stop-on-error   halt if any script fails (default: continue)
      --fresh           start each script with a fresh session

    Examples:
      batch clean.ost model.ost report.ost
      batch *.ost --stop-on-error
    """
    import glob as _glob

    ca = CommandArgs(args)
    if not ca.positional:
        return "Usage: batch <script1.ost> [script2.ost ...] [--stop-on-error]"

    stop_on_error = "--stop-on-error" in args
    fresh = "--fresh" in args

    # Expand glob patterns
    files = []
    for pat in ca.positional:
        matched = sorted(_glob.glob(pat))
        if matched:
            files.extend(matched)
        else:
            files.append(pat)

    if not files:
        return "No script files found."

    from openstat.script_runner import run_script_advanced
    results = [f"Batch: {len(files)} scripts", ""]
    errors = 0

    for i, fpath in enumerate(files, 1):
        if not Path(fpath).exists():
            msg = f"[{i}/{len(files)}] SKIP {fpath} — file not found"
            results.append(msg)
            errors += 1
            if stop_on_error:
                break
            continue

        results.append(f"[{i}/{len(files)}] Running: {fpath}")
        try:
            run_sess = Session() if fresh else session
            run_script_advanced(fpath, run_sess)
            results.append(f"  → OK")
        except Exception as exc:
            results.append(f"  → ERROR: {exc}")
            errors += 1
            if stop_on_error:
                results.append("Stopped due to error (--stop-on-error).")
                break

    results += ["", f"Batch complete. {len(files) - errors}/{len(files)} scripts succeeded."]
    return "\n".join(results)


# ── Git integration ──────────────────────────────────────────────────────────

@command("git", usage="git init|status|add|commit|log|diff [args]")
def cmd_git(session: Session, args: str) -> str:
    """Git version control for scripts and outputs (requires git).

    Runs git commands in the current working directory.
    Useful for versioning analysis scripts.

    Sub-commands:
      git init                    — initialize a git repository
      git status                  — show working tree status
      git add <file> [file2 ...]  — stage files
      git commit "<message>"      — commit staged changes
      git log [--n=10]            — show recent commits
      git diff [<file>]           — show unstaged changes
      git tag <name>              — create a tag for this analysis version

    Examples:
      git init
      git add analysis.ost outputs/results.pdf
      git commit "Add OLS regression results"
      git log --n=5
    """
    tokens = args.strip().split(None, 1)
    if not tokens:
        return "Usage: git init|status|add|commit|log|diff [args]"

    subcmd = tokens[0].lower()
    rest = tokens[1].strip() if len(tokens) > 1 else ""

    try:
        import subprocess

        def _run(cmd_parts):
            result = subprocess.run(
                cmd_parts, capture_output=True, text=True, cwd=os.getcwd()
            )
            out = (result.stdout + result.stderr).strip()
            return out or "(no output)"

        if subcmd == "init":
            return _run(["git", "init"])
        elif subcmd == "status":
            return _run(["git", "status", "--short"])
        elif subcmd == "add":
            files = rest.split() if rest else ["."]
            return _run(["git", "add"] + files)
        elif subcmd == "commit":
            msg = rest.strip().strip('"\'') or "OpenStat analysis update"
            return _run(["git", "commit", "-m", msg])
        elif subcmd == "log":
            n = 10
            if "--n=" in rest:
                try:
                    n = int(rest.split("--n=")[1].split()[0])
                except Exception:
                    pass
            return _run(["git", "log", f"--oneline", f"-{n}"])
        elif subcmd == "diff":
            file_arg = [rest] if rest else []
            return _run(["git", "diff"] + file_arg)
        elif subcmd == "tag":
            tag = rest.strip().strip('"\'')
            if not tag:
                return "Usage: git tag <name>"
            return _run(["git", "tag", tag])
        else:
            # Pass through to git directly
            return _run(["git", subcmd] + (rest.split() if rest else []))
    except FileNotFoundError:
        return "git not found. Install git: https://git-scm.com/"
    except Exception as e:
        return friendly_error(e, "git")


# ── Config profiles ───────────────────────────────────────────────────────────

@command("profile config", usage="profile config save|load|list|rm <name>")
def cmd_config_profile(session: Session, args: str) -> str:
    """Save and load configuration profiles.

    A profile saves the current config (output dir, plot size, etc.)
    with a name so you can switch between setups.

    Examples:
      profile config save presentation
      profile config save publication
      profile config load presentation
      profile config list
      profile config rm presentation
    """
    import json
    from openstat.config import get_config

    PROFILE_DIR = Path.home() / ".openstat" / "profiles"
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    tokens = args.strip().split()
    if not tokens:
        return "Usage: profile config save|load|list|rm <name>"

    subcmd = tokens[0].lower()
    name = tokens[1] if len(tokens) > 1 else ""

    if subcmd == "list":
        profiles = list(PROFILE_DIR.glob("*.json"))
        if not profiles:
            return "No saved profiles. Use: profile config save <name>"
        lines = ["Saved config profiles:"]
        for p in profiles:
            lines.append(f"  {p.stem}")
        return "\n".join(lines)

    if not name:
        return f"Usage: profile config {subcmd} <name>"

    profile_path = PROFILE_DIR / f"{name}.json"

    if subcmd == "save":
        cfg = get_config()
        data = {
            "output_dir": str(cfg.output_dir),
            "plot_figsize_w": cfg.plot_figsize_w,
            "plot_figsize_h": cfg.plot_figsize_h,
            "plot_dpi": cfg.plot_dpi,
            "csv_separator": cfg.csv_separator,
            "max_undo_stack": cfg.max_undo_stack,
        }
        profile_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return f"Profile '{name}' saved to {profile_path}"

    elif subcmd == "load":
        if not profile_path.exists():
            return f"Profile '{name}' not found. Use 'profile config list' to see available."
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        cfg = get_config()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return f"Profile '{name}' loaded: {data}"

    elif subcmd == "rm":
        if not profile_path.exists():
            return f"Profile '{name}' not found."
        profile_path.unlink()
        return f"Profile '{name}' removed."

    else:
        return f"Unknown sub-command: {subcmd}. Use save, load, list, or rm."
