from __future__ import annotations

from f1_tire_model.config import Settings
from f1_tire_model.datasets.manifest import is_built, load_manifest, mark_built


def test_is_built_false_when_no_manifest_exists(tmp_path):
    settings = Settings(cache_dir=tmp_path / "cache", dataset_output_dir=tmp_path / "output")
    assert is_built(2024, 1, "R", settings=settings) is False


def test_mark_built_persists_across_loads(tmp_path):
    settings = Settings(cache_dir=tmp_path / "cache", dataset_output_dir=tmp_path / "output")

    mark_built(2024, 1, "R", settings=settings)

    assert is_built(2024, 1, "R", settings=settings) is True
    assert is_built(2024, 2, "R", settings=settings) is False  # different round, not built


def test_mark_built_is_idempotent(tmp_path):
    settings = Settings(cache_dir=tmp_path / "cache", dataset_output_dir=tmp_path / "output")

    mark_built(2024, 1, "R", settings=settings)
    mark_built(2024, 1, "R", settings=settings)

    assert len(load_manifest(settings=settings)) == 1
