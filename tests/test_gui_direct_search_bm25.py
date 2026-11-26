from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import GUI_direct_search


def _sort_combo(mode):
    combo = MagicMock()
    combo.currentText.return_value = mode
    return combo


@pytest.mark.parametrize(
    "mode, expected_finite",
    [
        ("Distance ↑", [0.0, 0.9]),   # ascending: smallest distance first
        ("Distance ↓", [0.9, 0.0]),   # descending: largest distance first
    ],
)
def test_apply_direct_search_view_sorts_finite_then_missing(mode, expected_finite):
    rows = [
        {"file_name": "a", "distance": 0.0},
        {"file_name": "b", "distance": 0.9},
        {"file_name": "c"},                # missing distance key
        {"file_name": "d", "distance": None},
        {"file_name": "e", "distance": float("inf")},
        {"file_name": "f", "distance": float("nan")},
    ]
    captured = {}

    obj = SimpleNamespace(
        _direct_results=rows,
        direct_filter_input=None,
        direct_sort_combo=_sort_combo(mode),
        on_search_results=lambda view: captured.update(view=view),
    )

    GUI_direct_search.DirectSearchMixin.apply_direct_search_view(obj)

    view = captured["view"]
    assert len(view) == 6
    finite_distances = [r.get("distance") for r in view[:2]]
    assert finite_distances == expected_finite
    missing = view[2:]
    assert [row["file_name"] for row in missing] == ["c", "d", "e", "f"]


def test_stale_direct_completion_cannot_replace_current_results():
    displayed = []
    old_worker = object()
    current_worker = object()
    window = SimpleNamespace(
        sender=lambda: old_worker,
        _direct_worker=current_worker,
        _direct_results=[{"chunk_id": "current"}],
        _direct_rewrite_status="",
        apply_direct_search_view=lambda: displayed.append(True),
        statusBar=MagicMock(),
    )

    GUI_direct_search.DirectSearchMixin._on_direct_worker_results(
        window, [{"chunk_id": "previous"}]
    )

    assert window._direct_results == [{"chunk_id": "current"}]
    assert displayed == []
