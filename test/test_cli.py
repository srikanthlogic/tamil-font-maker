"""Tests for the stage CLI surface (pipeline.cli)."""
import pytest

from pipeline.cli import main


def test_init_creates_layout_and_config(tmp_path):
    p = tmp_path / "proj"
    assert main(["init", str(p), "--family", "My Font"]) == 0
    for d in ("sheets", "samples", "crops", "out"):
        assert (p / d).is_dir()
    assert 'family = "My Font"' in (p / "config.toml").read_text()
    # idempotent: a second init must not clobber an edited config
    (p / "config.toml").write_text('family = "Edited"\n')
    main(["init", str(p), "--family", "My Font"])
    assert 'family = "Edited"' in (p / "config.toml").read_text()


def test_ingest_extract_rejects_malformed_crop(tmp_path):
    p = tmp_path / "proj"
    main(["init", str(p)])
    with pytest.raises(SystemExit):
        main(["ingest-extract", str(p), "g_u0B95"])   # missing =path
