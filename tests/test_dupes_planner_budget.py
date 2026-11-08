"""The planner's runtime limits must not silently empty a large library's plan.

The grouping timeout (120 s) was measured from the start of the whole plan
build, which includes reading every file's tags. A first scan of a large
library spends longer than that just reading tags, so grouping "timed out" at
its first bucket and the scan reported (almost) no duplicates, with only a
plan-level review flag to say why. The limits now apply to grouping itself
and every limit that cuts a plan short is also written to the scan log.
"""
from __future__ import annotations

import duplicate_consolidation as dc
from _helpers.dupes_plans import write

TAGS = {"artist": "Artist", "title": "Song", "album": "Album"}


def _tracks(tmp_path, count):
    out = []
    for idx in range(count):
        path = write(tmp_path / f"Album{idx:02d}" / "Song.mp3")
        tags = dict(TAGS, title=f"Song {idx // 2}")  # pairs of duplicates
        out.append({"path": str(path), "fingerprint": f"{idx // 2} 1 2 3 4", "ext": ".mp3", "tags": tags})
    return out


def test_slow_tag_reading_does_not_use_up_the_grouping_budget(tmp_path, monkeypatch):
    clock = [0.0]
    real_normalize = dc._normalize_track

    def slow_normalize(raw, **kwargs):
        clock[0] += 10.0  # ten seconds per file on a slow disk
        return real_normalize(raw, **kwargs)

    monkeypatch.setattr(dc, "_now", lambda: clock[0])
    monkeypatch.setattr(dc, "_normalize_track", slow_normalize)
    logs = []
    plan = dc.build_consolidation_plan(_tracks(tmp_path, 20), timeout_sec=120.0, log_callback=logs.append)

    assert len(plan.groups) == 10
    assert not any("timed out" in flag for flag in plan.review_flags)


def test_limits_that_cut_the_plan_short_are_logged(tmp_path):
    logs = []
    plan = dc.build_consolidation_plan(_tracks(tmp_path, 6), max_candidates=4, log_callback=logs.append)
    assert any("Truncated candidate set" in flag for flag in plan.review_flags)
    assert any("Truncated candidate set" in line for line in logs)


def test_exactly_max_candidates_is_not_called_truncated(tmp_path):
    plan = dc.build_consolidation_plan(_tracks(tmp_path, 4), max_candidates=4)
    assert len(plan.groups) == 2
    assert not any("Truncated" in flag for flag in plan.review_flags)
