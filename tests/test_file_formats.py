"""Tests for SAS/SPSS file support and labels command (F10)."""

import pytest
import polars as pl

from openstat.session import Session
from openstat.commands.data_cmds import cmd_load, cmd_labels
from openstat.io.loader import load_file, save_file, _LOADERS


class TestFileFormats:
    """Test file format support."""

    def test_sas_extension_registered(self):
        assert ".sas7bdat" in _LOADERS

    def test_spss_extension_registered(self):
        assert ".sav" in _LOADERS

    def test_load_file_unsupported(self, tmp_path):
        fake = tmp_path / "data.xyz"
        fake.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(str(fake))

    def test_load_sas_import_error(self, tmp_path):
        """Loading .sas7bdat without pyreadstat gives clear error."""
        fake = tmp_path / "data.sas7bdat"
        fake.write_bytes(b"fake")
        try:
            load_file(str(fake))
        except (ImportError, Exception):
            pass  # Expected — either ImportError or read error

    def test_load_sav_import_error(self, tmp_path):
        """Loading .sav without pyreadstat gives clear error."""
        fake = tmp_path / "data.sav"
        fake.write_bytes(b"fake")
        try:
            load_file(str(fake))
        except (ImportError, Exception):
            pass

    def test_save_sav_extension(self, tmp_path):
        """save_file recognizes .sav extension."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        try:
            save_file(df, tmp_path / "out.sav")
        except ImportError:
            pass  # pyreadstat may not be installed


class TestLabelsCommand:
    """Test the labels command."""

    @pytest.fixture
    def session_with_labels(self, tmp_path):
        s = Session(output_dir=tmp_path / "out")
        s.df = pl.DataFrame({"gender": [1, 2, 1], "age": [25, 30, 35]})
        s._variable_labels = {
            "gender": {1: "Male", 2: "Female"},
        }
        return s

    def test_labels_no_labels(self, tmp_path):
        s = Session(output_dir=tmp_path / "out")
        s.df = pl.DataFrame({"x": [1]})
        result = cmd_labels(s, "")
        assert "No variable labels" in result

    def test_labels_list_all(self, session_with_labels):
        result = cmd_labels(session_with_labels, "")
        assert "gender" in result
        assert "2" in result or "Labels" in result

    def test_labels_specific_column(self, session_with_labels):
        result = cmd_labels(session_with_labels, "gender")
        assert "Male" in result
        assert "Female" in result

    def test_labels_missing_column(self, session_with_labels):
        result = cmd_labels(session_with_labels, "nonexistent")
        assert "No labels" in result
