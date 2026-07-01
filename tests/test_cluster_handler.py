"""Tests for Kubernetes cluster handler path selection helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import cluster_handler


def test_split_csv_paths_resolves_relative_paths() -> None:
    """Relative log paths should resolve under the app directory."""

    paths = cluster_handler.split_csv_paths("data/a.txt, /tmp/b.txt", Path("/app"))

    assert paths == [Path("/app/data/a.txt"), Path("/tmp/b.txt")]


def test_path_for_rank_returns_matching_input() -> None:
    """Each Kubernetes completion index should select one file."""

    paths = [Path("wifi_logs.txt"), Path("wifi_logs-1.txt")]

    assert cluster_handler.path_for_rank(paths, 0, "logs") == Path("wifi_logs.txt")
    assert cluster_handler.path_for_rank(paths, 1, "logs") == Path("wifi_logs-1.txt")


def test_path_for_rank_rejects_missing_rank() -> None:
    """A job should fail clearly when WORLD_SIZE exceeds configured file count."""

    with pytest.raises(ValueError, match="rank 2"):
        cluster_handler.path_for_rank([Path("wifi_logs.txt")], 2, "logs")
