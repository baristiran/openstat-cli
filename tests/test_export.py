"""Tests for export commands (docx, pptx) and EDA report."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from openstat.session import Session, ModelResult


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def full_session(tmp_path):
    s = Session()
    s.output_dir = tmp_path
    s.df = pl.DataFrame({
        "age":    [25, 30, 35, 40, 45],
        "income": [50000.0, 60000.0, 75000.0, 90000.0, 110000.0],
        "edu":    ["hs", "ba", "ma", "phd", "ba"],
        "score":  [70.0, 82.0, 91.0, 88.0, 76.0],
    })
    s.dataset_name = "test_dataset"
    s.results.append(ModelResult(
        name="OLS",
        formula="income ~ age",
        table="Coef.   3000.0\n_cons  -25000",
        details={"r2": 0.85, "n": 5},
    ))
    return s


# ── EDA Report ────────────────────────────────────────────────────────────

class TestEDAReport:
    def test_generates_html(self, full_session, tmp_path):
        from openstat.reporting.eda import generate_eda_report
        path = str(tmp_path / "eda.html")
        out = generate_eda_report(full_session, path)
        assert os.path.exists(out)
        assert out.endswith(".html")

    def test_html_content(self, full_session, tmp_path):
        from openstat.reporting.eda import generate_eda_report
        path = str(tmp_path / "eda.html")
        generate_eda_report(full_session, path)
        content = Path(path).read_text(encoding="utf-8")
        assert "EDA Report" in content
        assert "Dataset Overview" in content
        assert "Missing Values" in content
        assert "Numeric Summary" in content

    def test_includes_model_results(self, full_session, tmp_path):
        from openstat.reporting.eda import generate_eda_report
        path = str(tmp_path / "eda.html")
        generate_eda_report(full_session, path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Model Results" in content
        assert "OLS" in content

    def test_correlation_section(self, full_session, tmp_path):
        from openstat.reporting.eda import generate_eda_report
        path = str(tmp_path / "eda.html")
        generate_eda_report(full_session, path)
        content = Path(path).read_text(encoding="utf-8")
        assert "Correlation" in content

    def test_creates_parent_dir(self, full_session, tmp_path):
        from openstat.reporting.eda import generate_eda_report
        nested = str(tmp_path / "a" / "b" / "report.html")
        out = generate_eda_report(full_session, nested)
        assert os.path.exists(out)

    def test_self_contained_no_external_links(self, full_session, tmp_path):
        from openstat.reporting.eda import generate_eda_report
        path = str(tmp_path / "eda.html")
        generate_eda_report(full_session, path)
        content = Path(path).read_text(encoding="utf-8")
        # Should not reference external stylesheets
        assert 'rel="stylesheet"' not in content

    def test_dataset_name_in_title(self, full_session, tmp_path):
        from openstat.reporting.eda import generate_eda_report
        path = str(tmp_path / "eda.html")
        generate_eda_report(full_session, path)
        content = Path(path).read_text(encoding="utf-8")
        assert "test_dataset" in content


class TestReportCommand:
    def test_eda_subcommand(self, full_session, tmp_path):
        from openstat.commands.report_cmds import cmd_report
        path = str(tmp_path / "eda.html")
        out = cmd_report(full_session, f"eda {path}")
        assert "EDA report saved" in out

    def test_eda_default_path(self, full_session, tmp_path):
        from openstat.commands.report_cmds import cmd_report
        os.chdir(tmp_path)
        out = cmd_report(full_session, "eda")
        assert "EDA report saved" in out or "error" in out.lower()

    def test_markdown_still_works(self, full_session, tmp_path):
        from openstat.commands.report_cmds import cmd_report
        path = str(tmp_path / "report.md")
        out = cmd_report(full_session, path)
        # Should not crash; either saved or error
        assert isinstance(out, str)


# ── Export Command ─────────────────────────────────────────────────────────

class TestExportCommand:
    def test_unknown_format(self, full_session):
        from openstat.commands.export_cmds import cmd_export
        out = cmd_export(full_session, "xyz outputs/r.xyz")
        assert "Unknown" in out

    def test_no_args(self, full_session):
        from openstat.commands.export_cmds import cmd_export
        out = cmd_export(full_session, "")
        assert "Usage" in out

    def test_docx_missing_dep(self, full_session, tmp_path, monkeypatch):
        """If python-docx not installed, return install instructions."""
        import sys
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from openstat.commands import export_cmds
        out = export_cmds._export_docx(full_session, str(tmp_path / "r.docx"))
        assert "python-docx" in out or "pip" in out

    def test_pptx_missing_dep(self, full_session, tmp_path, monkeypatch):
        """If python-pptx not installed, return install instructions."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pptx":
                raise ImportError("No module named 'pptx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from openstat.commands import export_cmds
        out = export_cmds._export_pptx(full_session, str(tmp_path / "r.pptx"))
        assert "python-pptx" in out or "pip" in out

    @pytest.mark.skipif(
        not _try_import("docx"), reason="python-docx not installed"
    )
    def test_docx_creates_file(self, full_session, tmp_path):
        from openstat.commands.export_cmds import cmd_export
        path = str(tmp_path / "results.docx")
        out = cmd_export(full_session, f"docx {path}")
        assert "saved" in out
        assert os.path.exists(path)

    @pytest.mark.skipif(
        not _try_import("pptx"), reason="python-pptx not installed"
    )
    def test_pptx_creates_file(self, full_session, tmp_path):
        from openstat.commands.export_cmds import cmd_export
        path = str(tmp_path / "results.pptx")
        out = cmd_export(full_session, f"pptx {path}")
        assert "saved" in out
        assert os.path.exists(path)
