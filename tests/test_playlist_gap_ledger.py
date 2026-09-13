"""The ledger — what makes run two cheaper than run one."""
from __future__ import annotations

import pytest

from playlist_gap_ledger import (
    DECIDED_AUTO,
    DECIDED_USER,
    DECIDED_VERIFY,
    Ledger,
    ledger_path_for,
)
from playlist_gap_types import Verdict, WantedTrack


@pytest.fixture
def ledger(tmp_path):
    instance = Ledger(ledger_path_for(str(tmp_path)))
    yield instance
    instance.close()


def wanted(row_id, display="Song", **kwargs):
    return WantedTrack(row_id=row_id, display=display, **kwargs)


# ── what changed since last time ─────────────────────────────────────────────


def test_first_run_reports_everything_as_new(ledger):
    diff = ledger.sync_wanted("src", [wanted("a"), wanted("b")])
    assert diff.added == ["a", "b"]
    assert diff.new_count == 2


def test_second_run_reports_only_genuinely_new_rows(ledger):
    """The answer the user actually wants: "what's new since I last checked?"."""
    ledger.sync_wanted("src", [wanted("a"), wanted("b")])
    diff = ledger.sync_wanted("src", [wanted("a"), wanted("b"), wanted("c")])
    assert diff.added == ["c"]
    assert set(diff.unchanged) == {"a", "b"}


def test_rows_removed_upstream_are_reported(ledger):
    ledger.sync_wanted("src", [wanted("a"), wanted("b")])
    diff = ledger.sync_wanted("src", [wanted("a")])
    assert diff.removed_upstream == ["b"]


def test_a_row_returning_upstream_is_not_double_counted(ledger):
    ledger.sync_wanted("src", [wanted("a"), wanted("b")])
    ledger.sync_wanted("src", [wanted("a")])
    diff = ledger.sync_wanted("src", [wanted("a"), wanted("b")])
    assert diff.added == ["b"]


def test_sources_are_isolated(ledger):
    ledger.sync_wanted("one", [wanted("a")])
    assert ledger.sync_wanted("two", [wanted("a")]).added == ["a"]


# ── decision authority ───────────────────────────────────────────────────────


def test_a_human_decision_beats_the_matcher(ledger):
    ledger.record("a", Verdict.OWNED, by=DECIDED_USER, note="I have it")
    assert ledger.record("a", Verdict.MISSING, by=DECIDED_AUTO) is False
    assert ledger.decision_for("a").verdict is Verdict.OWNED


def test_verification_beats_the_matcher_but_not_a_human(ledger):
    ledger.record("a", Verdict.OWNED, by=DECIDED_AUTO)
    assert ledger.record("a", Verdict.MISSING, by=DECIDED_VERIFY) is True
    ledger.record("a", Verdict.OWNED, by=DECIDED_USER)
    assert ledger.record("a", Verdict.MISSING, by=DECIDED_VERIFY) is False


def test_a_human_can_change_their_own_mind(ledger):
    ledger.record("a", Verdict.OWNED, by=DECIDED_USER)
    ledger.record("a", Verdict.MISSING, by=DECIDED_USER)
    assert ledger.decision_for("a").verdict is Verdict.MISSING


def test_only_authoritative_decisions_are_replayed(ledger):
    """Replaying the matcher's own past guesses would freeze in its mistakes."""
    ledger.record("auto", Verdict.OWNED, by=DECIDED_AUTO)
    ledger.record("user", Verdict.IGNORED, by=DECIDED_USER)
    replayed = ledger.decisions()
    assert "user" in replayed and "auto" not in replayed
    assert "auto" in ledger.decisions(human_only=False)


def test_never_want_survives_a_rerun(ledger):
    ledger.record("a", Verdict.IGNORED, by=DECIDED_USER, note="skit")
    ledger.sync_wanted("src", [wanted("a")])
    assert ledger.decisions()["a"][0] is Verdict.IGNORED


def test_bulk_record(ledger):
    written = ledger.record_many(
        [("a", Verdict.OWNED), ("b", Verdict.OWNED)], note="feat-only difference"
    )
    assert written == 2
    assert ledger.decision_for("b").note == "feat-only difference"


def test_forget_removes_a_decision(ledger):
    ledger.record("a", Verdict.OWNED, by=DECIDED_USER)
    ledger.forget("a")
    assert ledger.decision_for("a") is None


# ── export and verification bookkeeping ──────────────────────────────────────


def test_exported_rows_become_outstanding(ledger):
    ledger.mark_exported(["a", "b"])
    assert set(ledger.outstanding()) == {"a", "b"}


def test_verifying_clears_outstanding(ledger):
    ledger.mark_exported(["a"])
    ledger.mark_verified("a", "correct", "/m/a.opus")
    assert ledger.outstanding() == []


def test_reopen_puts_a_row_back_on_the_missing_list(ledger):
    """The wrong-version fix: a download that was the wrong recording must not
    leave the row marked satisfied, or it is never asked about again."""
    ledger.record("a", Verdict.OWNED, by=DECIDED_AUTO)
    ledger.mark_exported(["a"])
    ledger.reopen("a", "wrong version arrived")
    assert ledger.decision_for("a").verdict is Verdict.MISSING
    assert ledger.outstanding() == []


def test_re_exporting_a_reopened_row_makes_it_pending_again(ledger):
    ledger.mark_exported(["a"])
    ledger.reopen("a", "wrong version")
    ledger.mark_exported(["a"])
    assert ledger.outstanding() == ["a"]


# ── identity migration ───────────────────────────────────────────────────────


def test_decision_moves_when_a_row_gains_a_real_id(ledger):
    """A CSV row gets a hashed id; the same track read later through a source
    with video ids gets a real one. The user's answer should follow."""
    ledger.record("s:hash", Verdict.IGNORED, by=DECIDED_USER, note="never want")
    assert ledger.migrate_row_id("s:hash", "yt:abc") is True
    assert ledger.decision_for("yt:abc").verdict is Verdict.IGNORED
    assert ledger.decision_for("s:hash") is None


def test_migration_never_clobbers_an_existing_decision(ledger):
    ledger.record("s:hash", Verdict.OWNED, by=DECIDED_USER)
    ledger.record("yt:abc", Verdict.MISSING, by=DECIDED_USER)
    assert ledger.migrate_row_id("s:hash", "yt:abc") is False
    assert ledger.decision_for("yt:abc").verdict is Verdict.MISSING


def test_migration_of_an_unknown_row_is_a_no_op(ledger):
    assert ledger.migrate_row_id("s:nothing", "yt:abc") is False


# ── durability ───────────────────────────────────────────────────────────────


def test_decisions_survive_reopening_the_database(tmp_path):
    path = ledger_path_for(str(tmp_path))
    first = Ledger(path)
    first.record("a", Verdict.IGNORED, by=DECIDED_USER)
    first.close()

    second = Ledger(path)
    try:
        assert second.decision_for("a").verdict is Verdict.IGNORED
    finally:
        second.close()


def test_reinitializing_is_safe(tmp_path):
    """Schema creation is additive and idempotent — the ledger holds the one
    thing this feature cannot regenerate."""
    path = ledger_path_for(str(tmp_path))
    for _ in range(3):
        instance = Ledger(path)
        instance.record("a", Verdict.OWNED, by=DECIDED_USER)
        instance.close()
    final = Ledger(path)
    try:
        assert final.decision_for("a") is not None
    finally:
        final.close()


def test_ledger_path_is_under_docs(tmp_path):
    assert ledger_path_for(str(tmp_path)).endswith("playlist_gap.sqlite3")
    assert "Docs" in ledger_path_for(str(tmp_path))


def test_source_round_trip(ledger):
    ledger.upsert_source("id1", "Liked videos", "csv", "/x/y.csv")
    ledger.upsert_source("id1", "Renamed", "csv", "/x/y.csv")
    rows = ledger.sources()
    assert len(rows) == 1 and rows[0]["name"] == "Renamed"
    assert ledger.last_run_at("id1") is None
    ledger.touch_source("id1")
    assert ledger.last_run_at("id1") is not None
