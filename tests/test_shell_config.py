"""config.load_config / save_config: corrupt files are rescued, saves are atomic (SHELL-13)."""
from __future__ import annotations

import glob
import json
import os

import pytest

import config


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(config, "_last_backup", None)
    monkeypatch.setattr(config, "_last_load_error", None)


def _write_raw(text: str) -> None:
    with open(config.CONFIG_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_missing_file_gives_defaults_without_error():
    cfg = config.load_config()
    assert cfg["library_root"] == ""
    assert cfg["near_duplicate_threshold"] == config.NEAR_DUPLICATE_THRESHOLD
    assert config.last_load_error() is None


def test_corrupt_file_is_backed_up_before_the_next_save():
    original = '{"metadata_api_key": "SECRET", "theme": "pearl", '   # truncated
    _write_raw(original)

    cfg = config.load_config()
    assert cfg["library_root"] == ""                  # defaults
    assert "could not be read" in (config.last_load_error() or "")

    cfg["library_root"] = "/music"
    config.save_config(cfg)                            # the routine save that used to wipe it

    backups = glob.glob(config.CONFIG_PATH + ".corrupt-*")
    assert len(backups) == 1
    assert open(backups[0], encoding="utf-8").read() == original
    assert config.last_backup_path() == backups[0]
    assert json.load(open(config.CONFIG_PATH, encoding="utf-8"))["library_root"] == "/music"

    # A readable file is never backed up again.
    config.save_config(config.load_config())
    assert len(glob.glob(config.CONFIG_PATH + ".corrupt-*")) == 1
    assert config.last_load_error() is None


def test_non_object_json_counts_as_corrupt():
    _write_raw("[1, 2, 3]")
    assert config.load_config()["library_root"] == ""
    assert config.last_load_error()
    config.save_config({"library_root": "x"})
    assert len(glob.glob(config.CONFIG_PATH + ".corrupt-*")) == 1


def test_failed_save_leaves_the_old_file_intact():
    config.save_config({"metadata_api_key": "SECRET"})
    with pytest.raises(TypeError):
        config.save_config({"metadata_api_key": "NEW", "bad": object()})
    assert json.load(open(config.CONFIG_PATH, encoding="utf-8")) == {"metadata_api_key": "SECRET"}
    leftovers = [p for p in os.listdir(os.path.dirname(config.CONFIG_PATH)) if p.endswith(".tmp")]
    assert leftovers == []


def test_save_goes_through_os_replace(monkeypatch):
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((src, dst))
        return real(src, dst)

    monkeypatch.setattr(config.os, "replace", spy)
    config.save_config({"a": 1})
    assert len(seen) == 1 and seen[0][1] == config.CONFIG_PATH
    assert os.path.dirname(seen[0][0]) == os.path.dirname(os.path.abspath(config.CONFIG_PATH))
