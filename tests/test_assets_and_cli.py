import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from churn_prediction.assets import obtain
from churn_prediction.paths import DEFAULT_CONFIG


def test_verified_assets_do_not_require_network(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    path.write_bytes(b"existing asset")

    def fail_request(*args, **kwargs):
        raise AssertionError("Verified assets must not trigger downloads")

    monkeypatch.setattr("churn_prediction.assets.requests.get", fail_request)
    assert "verified" in obtain(
        "https://example.invalid/asset", path, hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_bad_download_does_not_replace_existing_asset(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    path.write_bytes(b"keep this")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"invalid replacement"

    monkeypatch.setattr("churn_prediction.assets.requests.get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="checksum"):
        obtain("https://example.invalid/asset", path, "0" * 64)
    assert path.read_bytes() == b"keep this"
    assert not path.with_name("asset.part").exists()


def test_installed_cli_resolves_defaults_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "churn_prediction", "run", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--include-controls" in result.stdout
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from churn_prediction.paths import DEFAULT_CONFIG; print(DEFAULT_CONFIG)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()) == DEFAULT_CONFIG


def test_run_refuses_to_overwrite_previous_outputs(tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_text("previous results", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "churn_prediction", "run", "--output", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "not empty" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "previous results"
