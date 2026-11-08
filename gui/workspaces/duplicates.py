"""Duplicate Finder workspace — scan → preview → execute."""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes import motion
from gui.themes.glass import apply_font, current_material, font, paint_surface, qcolor
from gui.themes.manager import get_manager
from gui.themes.tokens import RADIUS, SPACE
from gui.widgets.dupes_parts import (
    GlyphLabel, PathLabel, TileGrid, ToneBadge, hero_title, paint_tone_pill, pill_font,
    tone_pill_size,
)
from gui.widgets.glass import ElidedLabel, GlassPanel
from gui.widgets.glass_parts import EmptyState
from gui.widgets.glyphs import set_glyph
from gui.workspaces.base import WorkspaceBase
from utils.reserved_folders import iter_library_files

_AUDIO_EXTS = {'.flac', '.m4a', '.aac', '.mp3', '.wav', '.ogg', '.opus'}

# How many individual fingerprint failures to log before summarising.
_MAX_LOGGED_FAILURES = 20

# Mirrors ExecutionConfig.operation_limit's default in the executor.
_EXECUTOR_OPERATION_LIMIT = 20_000

_DISPOSITION_CHOICES: list[tuple[str, str | None]] = [
    ("Default (global choice)", None),
    ("Keep loser in place (retain)", "retain"),
    ("Quarantine", "quarantine"),
    ("Delete permanently", "delete"),
]

_DELETE_CONFIRM_WORD = "DELETE"


def _missing_fingerprint_tools() -> list[str]:
    """External tools the scan needs (fpcalc + ffmpeg) that are not on PATH."""
    import chromaprint_utils
    return chromaprint_utils.missing_tools()


def _config_thresholds() -> tuple[float, float, float]:
    """(exact, near, mixed-codec boost) as saved by Settings / config defaults.

    Keys are the ones the Settings dialog writes (``exact_duplicate_threshold``,
    ``near_duplicate_threshold``) plus ``mixed_codec_threshold_boost``.
    """
    from config import (
        EXACT_DUPLICATE_THRESHOLD,
        MIXED_CODEC_THRESHOLD_BOOST,
        NEAR_DUPLICATE_THRESHOLD,
        load_config,
    )

    cfg = load_config()

    def _num(key: str, default: float) -> float:
        try:
            return float(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    return (
        _num("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD),
        _num("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD),
        _num("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST),
    )


def _gather_tracks_for_plan(
    library_path: str,
    db_path: str,
    log_callback,
    progress_callback,
    cancel_event: threading.Event,
    *,
    compute_missing: bool = True,
) -> list[dict]:
    """Walk ``library_path``, fingerprint files, and return a list of track dicts.

    Fingerprints come from ``db_path`` when cached; otherwise they are computed
    with ``chromaprint_utils.fingerprint_fpcalc`` (unless ``compute_missing``
    is False, e.g. because fpcalc is not installed) and stored back.
    """
    from fingerprint_cache import (
        ensure_fingerprint_cache,
        flush_fingerprint_writes,
        get_cached_fingerprint_metadata,
        store_fingerprint,
    )
    import chromaprint_utils
    from utils.audio_metadata_reader import read_tags

    ensure_fingerprint_cache(db_path)

    audio_paths = sorted(str(p) for p in iter_library_files(library_path, _AUDIO_EXTS))

    total = len(audio_paths)
    tracks: list[dict] = []
    failures = 0
    skipped_uncached = 0
    computed = 0

    def _failure(message: str) -> None:
        nonlocal failures
        failures += 1
        if failures <= _MAX_LOGGED_FAILURES:
            log_callback(message)

    for i, path in enumerate(audio_paths):
        if cancel_event.is_set():
            break
        pct = int((i + 1) / total * 60)

        fp = None
        cached_meta: dict | None = None
        try:
            fp, cached_meta = get_cached_fingerprint_metadata(path, db_path)
        except Exception:
            pass

        if not fp:
            if not compute_missing:
                skipped_uncached += 1
                progress_callback(pct)
                continue
            try:
                # Returns the fingerprint string (no duration).
                fp = chromaprint_utils.fingerprint_fpcalc(path)
            except chromaprint_utils.FingerprintError as exc:
                if str(exc).startswith("Required tool"):
                    # fpcalc/ffmpeg vanished mid-scan: say so once, not per file.
                    compute_missing = False
                    skipped_uncached += 1
                    log_callback(
                        f"! {exc}. Remaining files can only use fingerprints already in the cache."
                    )
                else:
                    _failure(f"! Fingerprint failed: {Path(path).name}: {exc}")
                progress_callback(pct)
                continue
            except Exception as exc:  # noqa: BLE001
                _failure(f"! Fingerprint failed: {Path(path).name}: {exc}")
                progress_callback(pct)
                continue
            if not fp:
                _failure(f"! No fingerprint (too short or silent?): {Path(path).name}")
                progress_callback(pct)
                continue
            computed += 1
            # Store newly computed fingerprint
            try:
                tags: dict = {}
                try:
                    tags = dict(read_tags(path) or {})
                except Exception:
                    pass
                ext_val = os.path.splitext(path)[1].lower()
                store_fingerprint(path, db_path, None, fp, ext=ext_val, tags=tags)
                cached_meta = {'ext': ext_val, 'tags': tags}
            except Exception:
                pass

        meta = cached_meta or {}
        ext_val = meta.get('ext') or os.path.splitext(path)[1].lower()
        if isinstance(ext_val, str) and ext_val and not ext_val.startswith('.'):
            ext_val = f'.{ext_val}'

        tracks.append({
            'path': path,
            'fingerprint': fp,
            'ext': ext_val,
            'bitrate': int(meta.get('bitrate') or 0),
            'sample_rate': int(meta.get('sample_rate') or 0),
            'bit_depth': int(meta.get('bit_depth') or 0),
            'channels': int(meta.get('channels') or 0),
            'codec': str(meta.get('codec') or ''),
            'container': str(meta.get('container') or ''),
            'tags': meta.get('tags') if isinstance(meta.get('tags'), dict) else {},
            'artwork': meta.get('artwork') if isinstance(meta.get('artwork'), list) else [],
            # Keeps the plan builder's metadata cache in <library>/Docs rather
            # than in <common folder of the tracks>/Docs (an album folder).
            'discovery': {'scan_roots': [library_path]},
        })
        progress_callback(pct)

    if failures > _MAX_LOGGED_FAILURES:
        log_callback(f"… {failures - _MAX_LOGGED_FAILURES} more fingerprint failure(s) not shown.")
    if skipped_uncached:
        log_callback(
            f"{skipped_uncached} file(s) skipped: not in the fingerprint cache and fpcalc is unavailable."
        )
    if computed:
        try:
            flush_fingerprint_writes(db_path, timeout=10)
        except Exception:
            pass
    return tracks


#: Executor ``skip_reason`` values that mean "left in place on purpose", with
#: the words the summary uses for them.
_SAFETY_REFUSALS = {
    "winner_missing": "winner missing",
    "same_file": "same file as its winner",
    "changed_since_preview": "changed since the preview",
    "no_snapshot": "not in the preview",
    "no_winner": "no winner in the plan",
    "winner_unreadable": "winner unreadable",
}


def _tally_execution(result) -> dict:
    """What an ``ExecutionResult`` did, counted once for the summary text
    (``summarize_execution``) and the page's stat tiles
    (``execution_counts``)."""
    actions = list(getattr(result, "actions", []) or [])
    blocked = [a for a in actions if a.step == "preflight" and a.status in ("blocked", "failed")]
    cleanup = [a for a in actions if a.step == "loser_cleanup"]
    # Safety refusals: the executor left the loser untouched on purpose (its
    # winner vanished, it is the winner through a link, or it changed since the
    # preview). They are recorded as "failed" but are not I/O errors.
    refused = [a for a in cleanup if a.status == "failed"
               and a.metadata.get("skip_reason") in _SAFETY_REFUSALS]
    refused_ids = {id(a) for a in refused}
    return {
        "actions": actions,
        "blocked": blocked,
        "quarantined": sum(
            1 for a in cleanup if a.status == "success" and a.metadata.get("disposition") == "quarantine"
        ),
        "deleted": sum(
            1 for a in cleanup if a.status == "success" and a.metadata.get("disposition") == "delete"
        ),
        "retained": sum(1 for a in cleanup if a.metadata.get("skip_reason") == "retain"),
        "refused": refused,
        "not_attempted": sum(1 for a in actions if a.metadata.get("skip_reason") == "not_attempted"),
        "cancelled": any(a.status == "cancelled" for a in actions),
        "failures": [a for a in actions if a.status == "failed" and id(a) not in refused_ids],
    }


def execution_counts(result) -> dict[str, object]:
    """The numbers behind ``summarize_execution``'s sentence, for stat tiles:
    quarantined, deleted, retained, refused (left in place for safety, with
    ``refusal_reasons``), not_attempted, errors, plus ``blocked`` (a
    preflight refusal: nothing was changed) and ``cancelled``."""
    tally = _tally_execution(result)
    return {
        "blocked": bool(tally["blocked"]),
        "quarantined": tally["quarantined"],
        "deleted": tally["deleted"],
        "retained": tally["retained"],
        "refused": len(tally["refused"]),
        "refusal_reasons": sorted({_SAFETY_REFUSALS[a.metadata["skip_reason"]] for a in tally["refused"]}),
        "not_attempted": tally["not_attempted"],
        "errors": 0 if tally["blocked"] else len(tally["failures"]),
        "cancelled": tally["cancelled"],
    }


def summarize_execution(result) -> tuple[bool, str]:
    """(success, one-line message) describing an ``ExecutionResult`` honestly.

    Preflight refusals used to surface only as "Execution completed with
    errors." — their reason lived in the report JSON.
    """
    tally = _tally_execution(result)
    blocked = tally["blocked"]
    if blocked:
        details: list[str] = []
        for action in blocked:
            text = action.detail
            if action.target not in ("plan", "execution"):
                text = f"{os.path.basename(action.target)}: {action.detail}"
            if text not in details:
                details.append(text)
        more = f" (+{len(details) - 3} more)" if len(details) > 3 else ""
        return False, "Execution blocked — nothing was changed. " + "; ".join(details[:3]) + more

    quarantined, deleted, retained = tally["quarantined"], tally["deleted"], tally["retained"]
    counts = f"{quarantined} quarantined, {deleted} deleted, {retained} kept in place"
    refused = tally["refused"]
    not_attempted = tally["not_attempted"]
    if refused:
        reasons = sorted({_SAFETY_REFUSALS[a.metadata["skip_reason"]] for a in refused})
        counts += f", {len(refused)} left in place for safety ({'; '.join(reasons)})"
    if not_attempted:
        counts += f", {not_attempted} not attempted"

    if tally["cancelled"]:
        return False, f"Execution cancelled at a safe point ({counts} before stopping). See the report."
    failures = tally["failures"]
    if failures:
        return False, (
            f"Execution finished with {len(failures)} error(s): {failures[0].detail} ({counts}). "
            "Some changes may have been applied — see the report."
        )
    if refused or not getattr(result, "success", False):
        return False, f"Execution finished: {counts}. See the report for each file left in place."
    return True, f"Execution complete: {counts}."


def _latest_execution_report(library_path: str) -> str:
    if not library_path:
        return ""
    root = Path(library_path) / "Docs" / "duplicate_execution_reports"
    if not root.is_dir():
        return ""
    try:
        reports = sorted(root.rglob("execution_report.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return ""
    return str(reports[0]) if reports else ""


# ── Workers ───────────────────────────────────────────────────────────────────

class DupeBuildWorker(QtCore.QThread):
    """Fingerprint library, build consolidation plan, and write the previews."""
    progress = Signal(int)
    log_line = Signal(str)
    groups_ready = Signal(list)     # list of GroupPlan
    done = Signal(bool, str)

    def __init__(
        self,
        library_path: str,
        exact_threshold: float,
        near_threshold: float,
        codec_boost: float,
        compute_missing: bool = True,
    ) -> None:
        super().__init__()
        self.library_path = library_path
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
        self.codec_boost = codec_boost
        self.compute_missing = compute_missing
        self._cancel_event = threading.Event()
        # Populated during run(); read by workspace after done
        self.plan = None
        self.preview_html_path: str = ""

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_consolidation import (
                build_consolidation_plan,
                export_consolidation_preview,
                export_consolidation_preview_html,
            )
        except ImportError as exc:
            self.done.emit(False, f"Import error: {exc}")
            return

        try:
            def _log(msg: str) -> None:
                self.log_line.emit(msg)

            docs = Path(self.library_path) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            db_path = str(docs / ".duplicate_fingerprints.db")

            _log("Stage 1: fingerprinting library…")
            tracks = _gather_tracks_for_plan(
                self.library_path,
                db_path,
                log_callback=_log,
                progress_callback=lambda pct: self.progress.emit(pct),
                cancel_event=self._cancel_event,
                compute_missing=self.compute_missing,
            )

            if self._cancel_event.is_set():
                self.done.emit(False, "Cancelled.")
                return

            if not tracks:
                self.done.emit(False, "No fingerprinted tracks found.")
                return

            _log(f"Stage 2: building consolidation plan for {len(tracks)} tracks…")
            self.progress.emit(62)
            last_pct = [62]

            def _plan_progress(current: int, total: int, _msg: str) -> None:
                pct = 62 + int(23 * current / total) if total else 62
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    self.progress.emit(pct)

            plan = build_consolidation_plan(
                tracks,
                exact_duplicate_threshold=self.exact_threshold,
                near_duplicate_threshold=self.near_threshold,
                mixed_codec_threshold_boost=self.codec_boost,
                cancel_event=self._cancel_event,
                progress_callback=_plan_progress,
                log_callback=_log,
            )
            if self._cancel_event.is_set():
                self.done.emit(False, "Cancelled.")
                return
            self.plan = plan
            self.progress.emit(85)

            _log("Stage 3: writing preview…")
            html_path = str(docs / "duplicate_preview.html")
            try:
                export_consolidation_preview_html(plan, html_path)
                self.preview_html_path = html_path
                _log(f"Preview written → {html_path}")
            except Exception as exc:
                _log(f"Warning: could not write HTML preview: {exc}")
            # The JSON plan is what Duplicates → Pair review reads.
            try:
                export_consolidation_preview(plan, str(docs / "duplicate_preview.json"))
            except Exception as exc:
                _log(f"Warning: could not write JSON preview: {exc}")

            self.progress.emit(100)
            self.groups_ready.emit(plan.groups)
            self.done.emit(True, f"Found {len(plan.groups)} duplicate group(s).")

        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


class DupeExecuteWorker(QtCore.QThread):
    """Execute a consolidation plan (dispositions are already on the plan)."""
    log_line = Signal(str)
    done = Signal(bool, str, str)   # success, message, report_html_path

    def __init__(
        self,
        plan,
        library_path: str,
        *,
        allow_review_required: bool = False,
        allow_deletion: bool = False,
        confirm_operation_overage: bool = False,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.library_path = library_path
        self.allow_review_required = allow_review_required
        self.allow_deletion = allow_deletion
        self.confirm_operation_overage = confirm_operation_overage
        self._cancel_event = threading.Event()
        # Set before ``done`` is emitted; read by the workspace's slot.
        self.counts: dict[str, object] | None = None

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_consolidation_executor import ExecutionConfig, execute_consolidation_plan
        except ImportError as exc:
            self.done.emit(False, f"Import error: {exc}", "")
            return
        try:
            docs = Path(self.library_path) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            reports_dir = str(docs / "duplicate_execution_reports")

            config = ExecutionConfig(
                library_root=self.library_path,
                reports_dir=reports_dir,
                quarantine_dir=str(Path(self.library_path) / "Quarantine"),
                quarantine_flatten=True,
                retain_losers=False,
                allow_review_required=self.allow_review_required,
                # The executor's deletion double-gate: both flags, set only
                # after the user typed the confirmation word.
                allow_deletion=self.allow_deletion,
                confirm_deletion=self.allow_deletion,
                confirm_operation_overage=self.confirm_operation_overage,
                cancel_event=self._cancel_event,
                log_callback=lambda msg: self.log_line.emit(msg),
            )

            result = execute_consolidation_plan(self.plan, config)
            report_html = result.report_paths.get("html_report", "")
            self.counts = execution_counts(result)
            ok, message = summarize_execution(result)
            self.done.emit(ok, message, str(report_html or ""))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc), "")


# ── Group rows ────────────────────────────────────────────────────────────────

_LOSSLESS_EXTS = {'.flac', '.wav', '.alac', '.ape', '.aiff', '.aif'}

#: How each disposition is named on a loser's tag pill, and its tone.
_DISPOSITION_TAGS: dict[str, tuple[str, str]] = {
    "quarantine": ("Quarantine", ""),
    "delete": ("Delete", "danger"),
    "retain": ("Keep in place", "accent"),
}
_KEEP_TAG = ("Keep", "success")
_MAX_LOSER_LINES = 3


def _format_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"  # pragma: no cover - unreachable


def quality_pills(group, path: str) -> list[tuple[str, str]]:
    """``(text, tone)`` pills for one file of a group: its format (lossless in
    the success tone) and, when known, its bitrate."""
    quality = (getattr(group, "track_quality", None) or {}).get(path) or {}
    ext = os.path.splitext(path)[1].lower()
    label = ext.lstrip(".").upper() or str(quality.get("container") or "?").upper()
    lossless = quality.get("is_lossless")
    if lossless is None:
        lossless = ext in _LOSSLESS_EXTS
    pills = [(label, "success" if lossless else "")]
    try:
        bitrate = int(quality.get("bitrate") or 0)
    except (TypeError, ValueError):
        bitrate = 0
    if bitrate > 0:
        kbps = bitrate // 1000 if bitrate >= 1000 else bitrate
        pills.append((f"{kbps:,} kbps", ""))
    return pills


class _GroupRowDelegate(QtWidgets.QStyledItemDelegate):
    """Paints one duplicate group as a glass row: the copy that is kept on
    the first line, then each loser with its disposition, each with its
    format and bitrate pills on the right. Row height follows the number of
    losers (at most three lines, then "+ N more")."""

    ROW_GAP = 6
    PAD_X = 12
    PAD_Y = 7
    WINNER_LINE = 26
    LOSER_LINE = 22

    def __init__(self, workspace: "DuplicatesWorkspace", parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._ws = workspace

    def _group(self, index: QtCore.QModelIndex):
        i = index.data(QtCore.Qt.ItemDataRole.UserRole)
        groups = self._ws._groups
        return groups[i] if isinstance(i, int) and 0 <= i < len(groups) else None

    @classmethod
    def row_height(cls, losers: int) -> int:
        lines = min(losers, _MAX_LOSER_LINES) + (1 if losers > _MAX_LOSER_LINES else 0)
        return cls.ROW_GAP + 2 * cls.PAD_Y + cls.WINNER_LINE + lines * cls.LOSER_LINE

    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: N802
        group = self._group(index)
        return QtCore.QSize(max(option.rect.width(), 200),
                            self.row_height(len(group.losers) if group else 0))

    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        group = self._group(index)
        if group is None:
            return
        t = get_manager().current
        m = current_material()
        state = option.state
        selected = bool(state & QtWidgets.QStyle.StateFlag.State_Selected)
        hovered = bool(state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        rect = QtCore.QRectF(option.rect).adjusted(0, self.ROW_GAP / 2, 0, -self.ROW_GAP / 2)
        fill = m.selection if selected else (m.pressed if hovered else m.well.fill)
        paint_surface(painter, rect, m.well, radius=RADIUS.sm, fill=fill)

        primary = qcolor(m.selection_text if selected else t.text_primary)
        muted = qcolor(m.selection_text, 0.78) if selected else qcolor(t.text_secondary)
        pfont = pill_font()
        tags = [_KEEP_TAG[0]] + [label for label, _tone in _DISPOSITION_TAGS.values()]
        tag_w = max(tone_pill_size(text, pfont).width() for text in tags)

        left = int(rect.left()) + self.PAD_X
        right = int(rect.right()) - self.PAD_X
        y = int(rect.top()) + self.PAD_Y
        extra = [("Review", "warning")] if group.review_flags else []
        self._line(painter, left, right, y, self.WINNER_LINE, _KEEP_TAG, tag_w,
                   Path(group.winner_path).name, font("body", weight=600), primary,
                   quality_pills(group, group.winner_path) + extra, pfont)
        y += self.WINNER_LINE
        disposition = self._ws._group_disposition(group)
        tag = _DISPOSITION_TAGS.get(disposition, _DISPOSITION_TAGS["quarantine"])
        for loser in group.losers[:_MAX_LOSER_LINES]:
            self._line(painter, left, right, y, self.LOSER_LINE, tag, tag_w,
                       Path(loser).name, font("label"), muted, quality_pills(group, loser), pfont)
            y += self.LOSER_LINE
        more = len(group.losers) - _MAX_LOSER_LINES
        if more > 0:
            painter.save()
            painter.setFont(font("caption"))
            painter.setPen(muted)
            painter.drawText(QtCore.QRect(left + tag_w + 10, y, max(0, right - left - tag_w - 10),
                                          self.LOSER_LINE),
                             int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                             f"+ {more} more (all listed below when selected)")
            painter.restore()

    @staticmethod
    def _line(painter: QtGui.QPainter, left: int, right: int, top: int, height: int,
              tag: tuple[str, str], tag_w: int, name: str, name_font: QtGui.QFont,
              colour: QtGui.QColor, pills: list[tuple[str, str]], pfont: QtGui.QFont) -> None:
        pill_h = tone_pill_size("X", pfont).height()
        pill_y = top + (height - pill_h) // 2
        tag_size = tone_pill_size(tag[0], pfont)
        paint_tone_pill(painter, QtCore.QRect(left, pill_y, tag_size.width(), pill_h), tag[0], tag[1], pfont)
        x_right = right
        for text, tone in reversed(pills):
            w = tone_pill_size(text, pfont).width()
            x_right -= w
            paint_tone_pill(painter, QtCore.QRect(x_right, pill_y, w, pill_h), text, tone, pfont)
            x_right -= 6
        name_left = left + tag_w + 10
        width = max(0, x_right - 6 - name_left)
        painter.save()
        painter.setFont(name_font)
        painter.setPen(colour)
        fm = QtGui.QFontMetrics(name_font)
        painter.drawText(QtCore.QRect(name_left, top, width, height),
                         int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                         fm.elidedText(name, QtCore.Qt.TextElideMode.ElideMiddle, width))
        painter.restore()


# ── Workspace ─────────────────────────────────────────────────────────────────

class DuplicatesWorkspace(WorkspaceBase):
    """Duplicate Finder — scan, review groups, preview and execute."""

    _STEP_SCAN, _STEP_REVIEW, _STEP_EXECUTE = range(3)

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._groups: list = []
        self._plan = None
        self._plan_executed = False
        self._group_overrides: dict[str, str] = {}
        self._worker: DupeBuildWorker | None = None
        self._exec_worker: DupeExecuteWorker | None = None
        self._scan_running = False
        self._exec_running = False
        self._preview_html_path: str = ""
        self._last_report_path: str = ""
        self._config_snapshot: tuple[float, float, float] | None = None
        self._step_labels: list[QtWidgets.QLabel] = []
        self._build_ui()
        self._load_thresholds_from_config(force=True)
        self._last_report_path = _latest_execution_report(library_path)
        self._show_empty(*self._initial_empty_text(library_path))
        self._refresh_buttons()

    @staticmethod
    def _initial_empty_text(library_path: str) -> tuple[str, str]:
        if not library_path:
            return ("No library selected",
                    "Choose a library folder first; Find duplicates then scans it.")
        return ("No scan yet",
                "Find duplicates fingerprints the library and lists each group of copies here, "
                "with the copy it keeps.")

    def _build_ui(self) -> None:
        cl = self.content_layout

        # ── Hero: the library, the scan and its options ───────────────────
        hero = self._make_card()
        hero.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(hero)
        hl.setContentsMargins(16, 16, 16, 16)
        hl.setSpacing(SPACE.md)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(SPACE.md)
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(SPACE.xxs)
        title = hero_title("Scan your library")
        titles.addWidget(title)
        titles.addWidget(self._make_subtitle(
            "Fingerprints every track and groups copies of the same recording, keeping the "
            "best one. Nothing moves until you execute."
        ))
        top.addLayout(titles, 1)
        self._scan_btn = self._make_primary_button("Find duplicates")
        set_glyph(self._scan_btn, "search", role="inverse")
        self._scan_btn.setToolTip("Fingerprint the library and build a duplicate plan (read-only).")
        self._scan_btn.clicked.connect(self._on_scan)
        top.addWidget(self._scan_btn, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        hl.addLayout(top)

        lib_row = GlassPanel("well", radius=RADIUS.sm)
        lib_row.setObjectName("dupLibraryRow")
        lrl = QtWidgets.QHBoxLayout(lib_row)
        lrl.setContentsMargins(12, 8, 12, 8)
        lrl.setSpacing(SPACE.sm)
        lrl.addWidget(GlyphLabel("folder"), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._library_lbl = PathLabel(self._library_path, "No library selected")
        self._library_lbl.setObjectName("dupLibraryPath")
        lrl.addWidget(self._library_lbl, 1)
        hl.addWidget(lib_row)

        # Scan options: the three thresholds, label above field.
        opts = QtWidgets.QHBoxLayout()
        opts.setSpacing(SPACE.lg)
        self._exact_spin = QtWidgets.QDoubleSpinBox()
        self._exact_spin.setRange(0.0, 1.0)
        self._exact_spin.setSingleStep(0.005)
        self._exact_spin.setDecimals(3)
        self._exact_spin.setToolTip(
            "Fingerprint distance ≤ this value → exact duplicate.\n"
            "Starts from the value saved in Settings."
        )

        self._near_spin = QtWidgets.QDoubleSpinBox()
        self._near_spin.setRange(0.0, 1.0)
        self._near_spin.setSingleStep(0.01)
        self._near_spin.setDecimals(3)
        self._near_spin.setToolTip(
            "Fingerprint distance ≤ this value → near duplicate.\n"
            "Starts from the value saved in Settings."
        )

        self._boost_spin = QtWidgets.QDoubleSpinBox()
        self._boost_spin.setRange(0.0, 0.5)
        self._boost_spin.setSingleStep(0.005)
        self._boost_spin.setDecimals(3)
        self._boost_spin.setToolTip(
            "Extra distance allowance when comparing lossless vs. lossy files."
        )
        for caption, spin in (("Exact duplicate", self._exact_spin),
                              ("Near duplicate", self._near_spin),
                              ("Mixed-codec boost", self._boost_spin)):
            field = QtWidgets.QVBoxLayout()
            field.setSpacing(SPACE.xxs)
            lbl = QtWidgets.QLabel(caption)
            lbl.setObjectName("mutedLabel")
            apply_font(lbl, "caption")
            lbl.setToolTip(spin.toolTip())
            lbl.setBuddy(spin)
            spin.setMinimumWidth(104)
            spin.setAccessibleName(f"{caption} threshold")
            field.addWidget(lbl)
            field.addWidget(spin)
            opts.addLayout(field)
        opts.addStretch(1)
        hl.addLayout(opts)

        # Status: the step the flow is at, progress, and Cancel while busy.
        status = QtWidgets.QHBoxLayout()
        status.setSpacing(SPACE.sm)
        stepper = QtWidgets.QFrame()
        stepper.setObjectName("dupStepRow")
        sl = QtWidgets.QHBoxLayout(stepper)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(SPACE.xs)
        for i, step in enumerate(["Scan", "Review", "Execute"]):
            lbl = QtWidgets.QLabel(step)
            lbl.setObjectName("stepActive" if i == 0 else "stepInactive")
            lbl.setToolTip("Where the duplicate workflow is: scan, review the groups, execute the plan.")
            self._step_labels.append(lbl)
            sl.addWidget(lbl)
        status.addWidget(stepper, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setObjectName("statusHint")
        self._prog_status.setWordWrap(True)
        self._prog_status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                        QtWidgets.QSizePolicy.Policy.Preferred)
        status.addWidget(self._prog_status, 1)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        set_glyph(self._cancel_btn, "close")
        self._cancel_btn.setToolTip("Stop the scan, or stop the execution at the next safe point.")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        status.addWidget(self._cancel_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        hl.addLayout(status)

        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setAccessibleName("Duplicate scan progress")
        hl.addWidget(self._prog_bar)
        cl.addWidget(hero)

        # ── Results: summary, groups, the selected group ──────────────────
        results = self._make_card()
        results.setObjectName("workspaceCard")
        rl = QtWidgets.QVBoxLayout(results)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(SPACE.md)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(SPACE.sm)
        head.addWidget(self._make_card_title("Duplicate groups"))
        head.addStretch(1)
        self._fp_status_lbl = QtWidgets.QLabel("")
        self._fp_status_lbl.setObjectName("mutedLabel")
        apply_font(self._fp_status_lbl, "caption")
        head.addWidget(self._fp_status_lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        rl.addLayout(head)

        self._results_stack = QtWidgets.QStackedWidget()
        self._empty = EmptyState(
            "No scan yet",
            "Find duplicates fingerprints the library and lists each group of copies here, "
            "with the copy it keeps.",
        )
        self._results_stack.addWidget(self._empty)

        page = QtWidgets.QWidget()
        self._results_page = page
        pl = QtWidgets.QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(SPACE.sm)
        self._plan_stats = TileGrid([
            ("groups", "Groups"),
            ("files", "Duplicate files"),
            ("space", "Space reclaimable"),
            ("flagged", "Flagged for review"),
        ])
        self._plan_stats.tile("space").setToolTip(
            "Size of the copies that leave the library (quarantined or deleted). "
            "Quarantined files still use disk space until you empty Quarantine/."
        )
        self._plan_stats.tile("flagged").setToolTip(
            "Groups the planner wants you to look at. Execute lists them and asks before running them."
        )
        pl.addWidget(self._plan_stats)
        self._outcome_stats = TileGrid([
            ("quarantined", "Quarantined"),
            ("deleted", "Deleted"),
            ("refused", "Left in place for safety"),
            ("not_attempted", "Not attempted"),
            ("errors", "Errors"),
        ])
        self._outcome_stats.tile("refused").setToolTip(
            "Copies the executor left untouched on purpose: their winner was missing or "
            "unreadable, or the file changed since the preview."
        )
        self._outcome_stats.setVisible(False)
        pl.addWidget(self._outcome_stats)
        self._summary_reason = QtWidgets.QLabel("")
        self._summary_reason.setObjectName("mutedLabel")
        self._summary_reason.setWordWrap(True)
        apply_font(self._summary_reason, "label")
        pl.addWidget(self._summary_reason)

        self._groups_table = QtWidgets.QTreeWidget()
        self._groups_table.setObjectName("dupGroupList")
        self._groups_table.setColumnCount(1)
        self._groups_table.setHeaderHidden(True)
        self._groups_table.setRootIsDecorated(False)
        self._groups_table.setIndentation(0)
        self._groups_table.setUniformRowHeights(False)
        self._groups_table.setAlternatingRowColors(False)
        self._groups_table.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._groups_table.setMouseTracking(True)
        self._groups_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._groups_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._groups_table.setAccessibleName("Duplicate groups")
        self._groups_table.setItemDelegate(_GroupRowDelegate(self, self._groups_table))
        self._groups_table.currentItemChanged.connect(self._on_group_selected)
        pl.addWidget(self._groups_table)

        inspector = GlassPanel("well", radius=RADIUS.sm)
        inspector.setObjectName("dupGroupInspector")
        il = QtWidgets.QHBoxLayout(inspector)
        il.setContentsMargins(12, 6, 8, 6)
        il.setSpacing(SPACE.sm)
        self._selected_lbl = ElidedLabel("Select a group to choose what happens to its copies.")
        self._selected_lbl.setObjectName("mutedLabel")
        il.addWidget(self._selected_lbl, 1)
        disp_lbl = QtWidgets.QLabel("Losers")
        disp_lbl.setObjectName("mutedLabel")
        apply_font(disp_lbl, "caption")
        il.addWidget(disp_lbl)
        self._disposition_combo = QtWidgets.QComboBox()
        for label, value in _DISPOSITION_CHOICES:
            self._disposition_combo.addItem(label, value)
        self._disposition_combo.setEnabled(False)
        self._disposition_combo.setToolTip(
            "What happens to this group's losers. \"Default\" follows the "
            "Quarantine / Delete choice below."
        )
        self._disposition_combo.setAccessibleName("Disposition for the selected group")
        disp_lbl.setBuddy(self._disposition_combo)
        self._disposition_combo.currentIndexChanged.connect(self._on_disposition_changed)
        il.addWidget(self._disposition_combo)
        pl.addWidget(inspector)

        self._group_details = QtWidgets.QPlainTextEdit()
        self._group_details.setObjectName("logText")
        self._group_details.setReadOnly(True)
        self._group_details.setFixedHeight(96)
        self._group_details.setPlaceholderText("Select a group to see its files and review flags…")
        self._group_details.setAccessibleName("Selected group details")
        pl.addWidget(self._group_details)
        self._results_stack.addWidget(page)
        rl.addWidget(self._results_stack)
        cl.addWidget(results)

        # ── Apply the plan ────────────────────────────────────────────────
        apply_card = self._make_card()
        apply_card.setObjectName("actionCard")
        al = QtWidgets.QVBoxLayout(apply_card)
        al.setContentsMargins(16, 16, 16, 16)
        al.setSpacing(SPACE.sm)
        al.addWidget(self._make_card_title("Apply the plan"))
        al.addWidget(self._make_subtitle(
            "Execute shows exactly what will change and asks first. Every action is logged to Docs/."
        ))

        self._quarantine_rb = QtWidgets.QRadioButton("Quarantine duplicates (safe default)")
        self._quarantine_rb.setChecked(True)
        self._quarantine_rb.setToolTip("Losers are moved to the library's Quarantine/ folder.")
        self._delete_rb = QtWidgets.QRadioButton("Delete losers permanently")
        self._delete_rb.setObjectName("dupDeleteChoice")
        self._delete_rb.setToolTip(
            "Losers are deleted, not moved. Execute asks you to type DELETE to confirm."
        )
        self._quarantine_rb.toggled.connect(self._on_global_disposition_changed)
        al.addWidget(self._quarantine_rb)
        delete_row = QtWidgets.QHBoxLayout()
        delete_row.setSpacing(SPACE.sm)
        delete_row.addWidget(self._delete_rb)
        self._delete_badge = ToneBadge("Cannot be undone", "danger")
        self._delete_badge.setToolTip(self._delete_rb.toolTip())
        delete_row.addWidget(self._delete_badge, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        delete_row.addStretch(1)
        al.addLayout(delete_row)

        self._playlists_cb = QtWidgets.QCheckBox("Update playlists after execution")
        # The executor always rewrites playlist entries that point at a moved
        # loser (it would otherwise leave broken entries), so this is not a
        # choice: show it as always on rather than as a dead checkbox.
        self._playlists_cb.setChecked(True)
        self._playlists_cb.setEnabled(False)
        self._playlists_cb.setToolTip(
            "Always on: playlists that reference a duplicate that is moved or deleted are "
            "rewritten to point at the kept copy (backups are saved with the execution report)."
        )
        al.addWidget(self._playlists_cb)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(SPACE.sm)
        self._preview_btn = QtWidgets.QPushButton("Preview")
        set_glyph(self._preview_btn, "external")
        self._preview_btn.setToolTip("Open the plan (Docs/duplicate_preview.html) in your browser.")
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview)

        self._report_btn = QtWidgets.QPushButton("Report")
        set_glyph(self._report_btn, "file")
        self._report_btn.setEnabled(False)
        self._report_btn.setToolTip("Open the most recent execution report.")
        self._report_btn.clicked.connect(self._on_open_report)

        self._execute_btn = QtWidgets.QPushButton("Execute…")
        self._execute_btn.setEnabled(False)
        self._execute_btn.setObjectName("dangerBtn")
        set_glyph(self._execute_btn, "check", role="inverse")
        self._execute_btn.setToolTip("Confirm, then quarantine (or delete) the losers as previewed.")
        self._execute_btn.clicked.connect(self._on_execute)

        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._report_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._execute_btn)
        al.addLayout(btn_row)
        cl.addWidget(apply_card)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.setSpacing(SPACE.sm)
        log_layout.addWidget(self._make_card_title("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setObjectName("logText")
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(120)
        self._log_area.setPlaceholderText("Scan and execution messages appear here.")
        self._log_area.setAccessibleName("Duplicate finder log")
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── State helpers ─────────────────────────────────────────────────────

    def _status(self, text: str, kind: str = "info") -> None:
        tokens = get_manager().current
        colour = {
            "busy": tokens.warning,
            "ok": tokens.success,
            "warn": tokens.warning,
            "error": tokens.danger,
        }.get(kind, tokens.text_secondary)
        self.status_changed.emit(text, colour)

    def _set_step(self, index: int) -> None:
        for i, lbl in enumerate(self._step_labels):
            name = "stepActive" if i == index else "stepInactive"
            if lbl.objectName() != name:
                lbl.setObjectName(name)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

    def _current_step(self) -> int:
        for i, lbl in enumerate(self._step_labels):
            if lbl.objectName() == "stepActive":
                return i
        return 0

    def _refresh_buttons(self) -> None:
        busy = self._scan_running or self._exec_running
        self._scan_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._cancel_btn.setVisible(busy)
        self._prog_bar.setVisible(busy)
        self._preview_btn.setEnabled(
            not busy and bool(self._preview_html_path) and Path(self._preview_html_path).exists()
        )
        self._execute_btn.setEnabled(
            not busy and self._plan is not None and bool(self._groups) and not self._plan_executed
        )
        self._report_btn.setEnabled(
            not self._exec_running and bool(self._last_report_path) and Path(self._last_report_path).exists()
        )

    def _load_thresholds_from_config(self, force: bool = False) -> None:
        """Use the thresholds saved in Settings; re-read when they changed."""
        try:
            values = _config_thresholds()
        except Exception:  # noqa: BLE001 - a broken config must not break the page
            return
        if not force and values == self._config_snapshot:
            return
        self._config_snapshot = values
        exact, near, boost = values
        self._exact_spin.setValue(exact)
        self._near_spin.setValue(near)
        self._boost_spin.setValue(boost)

    def on_settings_saved(self) -> None:
        # Follow thresholds saved in Settings (only reloads when they changed,
        # so a value the user typed here survives an unrelated save).
        self._load_thresholds_from_config(force=False)

    def _global_disposition(self) -> str:
        return "delete" if self._delete_rb.isChecked() else "quarantine"

    def _group_disposition(self, group) -> str:
        """What Execute will do with ``group``'s losers (its override, else the global choice)."""
        return self._group_overrides.get(group.group_id) or self._global_disposition()

    # ── Results view ──────────────────────────────────────────────────────

    def _show_empty(self, title: str, message: str) -> None:
        self._empty.set_text(title, message)
        self._results_stack.setCurrentWidget(self._empty)

    def _show_results(self) -> None:
        if self._results_stack.currentWidget() is not self._results_page:
            self._results_stack.setCurrentWidget(self._results_page)
            motion.reveal(self._results_page)

    def _fit_list_height(self) -> None:
        """The group list is as tall as its rows, up to a cap (then it scrolls)."""
        total = sum(_GroupRowDelegate.row_height(len(g.losers)) for g in self._groups[:40])
        self._groups_table.setFixedHeight(max(_GroupRowDelegate.row_height(1), min(total + 2, 420)))

    def _refresh_plan_stats(self) -> None:
        groups = self._groups
        files = sum(len(g.losers) for g in groups)
        flagged = sum(1 for g in groups if g.review_flags)
        leaving = 0
        for group in groups:
            if self._group_disposition(group) == "retain":
                continue
            state = getattr(group, "library_state", None) or {}
            for loser in group.losers:
                try:
                    leaving += int((state.get(loser) or {}).get("size") or 0)
                except (TypeError, ValueError):
                    pass
        stats = self._plan_stats
        stats.tile("groups").set_value(f"{len(groups):,}")
        stats.tile("files").set_value(f"{files:,}")
        stats.tile("space").set_value(_format_bytes(leaving))
        stats.tile("flagged").set_value(f"{flagged:,}", tone="warning" if flagged else "")
        if self._plan_executed:
            return
        if flagged:
            reason = (f"{flagged} group(s) are flagged for review: Execute lists them and asks "
                      "before running them. Nothing has moved yet.")
        else:
            reason = "Nothing has moved yet. Preview opens the plan; Execute asks before it changes anything."
        self._summary_reason.setText(reason)

    def _show_outcome(self, counts: dict | None, message: str) -> None:
        """Fill the execution tiles from ``DupeExecuteWorker.counts``."""
        tiles = self._outcome_stats
        if counts is None:
            for key in tiles.keys():
                tiles.tile(key).set_value("–", tone="")
        else:
            tones = {"refused": "warning", "not_attempted": "warning", "errors": "danger"}
            for key in tiles.keys():
                value = int(counts.get(key) or 0)
                tiles.tile(key).set_value(f"{value:,}", tone=tones.get(key, "") if value else "")
        tiles.setVisible(True)
        self._summary_reason.setText(message)

    @Slot(bool)
    def _on_global_disposition_changed(self, _checked: bool = False) -> None:
        """The Quarantine / Delete choice applies to every group without an
        override: repaint the rows, the totals and the selected group."""
        self._groups_table.viewport().update()
        if self._groups:
            self._refresh_plan_stats()
        item = self._groups_table.currentItem()
        if item is not None:
            index = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(index, int) and 0 <= index < len(self._groups):
                self._group_details.setPlainText(self._group_detail_text(self._groups[index]))

    def _effective_dispositions(self) -> dict[str, str]:
        if self._plan is None:
            return {}
        default = self._global_disposition()
        out: dict[str, str] = {}
        for group in self._plan.groups:
            disposition = self._group_overrides.get(group.group_id) or default
            for loser in group.losers:
                out[loser] = disposition
        return out

    def _apply_dispositions(self, effective: dict[str, str]) -> bool:
        """Write the chosen dispositions into the plan; True if anything changed.

        The plan's signature is refreshed and the preview re-exported so the
        preview always shows exactly what Execute will do.
        """
        previous = {g.group_id: dict(g.loser_disposition) for g in self._plan.groups}
        changed = False
        for group in self._plan.groups:
            wanted = {loser: effective.get(loser, "quarantine") for loser in group.losers}
            if dict(group.loser_disposition) != wanted:
                group.loser_disposition = wanted
                changed = True
        if not changed:
            return False
        self._plan.refresh_plan_signature()
        try:
            from duplicate_consolidation import (
                export_consolidation_preview,
                export_consolidation_preview_html,
            )
            docs = Path(self._library_path) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            html_path = str(docs / "duplicate_preview.html")
            export_consolidation_preview_html(self._plan, html_path)
            export_consolidation_preview(self._plan, str(docs / "duplicate_preview.json"))
        except Exception:
            # Keep plan and preview in step: undo, so a retry re-exports.
            for group in self._plan.groups:
                group.loser_disposition = previous[group.group_id]
            self._plan.refresh_plan_signature()
            raise
        self._preview_html_path = html_path
        self._log_line("Preview updated with your disposition choices.")
        return True

    def _group_detail_text(self, group) -> str:
        override = self._group_overrides.get(group.group_id)
        default = self._global_disposition()
        lines = [f"Winner: {group.winner_path}"]
        for loser in group.losers:
            disposition = override or default
            lines.append(f"  Loser ({disposition}): {loser}")
        if override:
            lines.append(f"Disposition override: {override}")
        if group.review_flags:
            lines.append(f"Review flags: {', '.join(group.review_flags)}")
        return "\n".join(lines)

    def _log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        if self._scan_running or self._exec_running:
            return

        compute_missing = True
        missing = _missing_fingerprint_tools()
        if missing:
            names = " and ".join(missing)
            reply = QtWidgets.QMessageBox.question(
                self,
                "Fingerprinting tools not found",
                f"{names} could not be found on PATH.\n\n"
                "Find duplicates fingerprints audio with fpcalc (Chromaprint — "
                "https://acoustid.org/chromaprint) and decodes it with FFmpeg "
                "(https://ffmpeg.org). Install them, make sure they are on PATH, "
                "then scan again.\n\n"
                "Scan anyway, using only fingerprints already in this library's cache? "
                "Files that were never fingerprinted will be skipped.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                self._prog_status.setText(f"Not scanned: {names} not found.")
                self._log(f"Duplicate scan not started: {names} not found on PATH.", "warn")
                return
            compute_missing = False

        self._log_area.clear()
        self._groups_table.clear()
        self._group_details.clear()
        self._groups = []
        self._plan = None
        self._plan_executed = False
        self._group_overrides.clear()
        self._fp_status_lbl.setText("")
        self._outcome_stats.setVisible(False)
        self._show_empty("Scanning…", "Groups appear here when the scan finishes.")
        self._prog_bar.setValue(0)
        self._prog_status.setText("Scanning…")
        self._set_step(self._STEP_SCAN)
        self._log("Starting duplicate scan…", "info")
        self._status("Scanning…", "busy")
        if not compute_missing:
            self._log_line("fpcalc/ffmpeg unavailable: using cached fingerprints only.")

        self._worker = DupeBuildWorker(
            library_path=self._library_path,
            exact_threshold=self._exact_spin.value(),
            near_threshold=self._near_spin.value(),
            codec_boost=self._boost_spin.value(),
            compute_missing=compute_missing,
        )
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.groups_ready.connect(self._populate_groups)
        self._worker.done.connect(self._on_scan_finished)
        self._scan_running = True
        self._refresh_buttons()
        self._start_worker(self._worker, "Duplicate scan")

    @Slot()
    def _on_cancel(self) -> None:
        if self._scan_running and self._worker is not None:
            self._worker.cancel()
            self._prog_status.setText("Cancelling…")
            self._cancel_btn.setEnabled(False)
        if self._exec_running and self._exec_worker is not None:
            self._exec_worker.cancel()
            self._prog_status.setText("Cancelling at the next safe point…")
            self._cancel_btn.setEnabled(False)

    @Slot()
    def _on_preview(self) -> None:
        if self._preview_html_path and Path(self._preview_html_path).exists():
            webbrowser.open(Path(self._preview_html_path).resolve().as_uri())
        else:
            QtWidgets.QMessageBox.information(self, "No Preview", "Preview not yet generated — run a scan first.")

    def _review_gate(self) -> tuple[bool, bool]:
        """(proceed, allow_review_required). Shows why groups are flagged."""
        plan = self._plan
        flagged = [(i, g) for i, g in enumerate(plan.groups, 1) if g.review_flags]
        if not flagged and not plan.review_flags:
            return True, False
        lines = [
            f"{len(flagged)} of {len(plan.groups)} duplicate group(s) are flagged for review:",
            "",
        ]
        for index, group in flagged[:8]:
            lines.append(f"  #{index} {Path(group.winner_path).name} — {'; '.join(group.review_flags)}")
        if len(flagged) > 8:
            lines.append(f"  … and {len(flagged) - 8} more (see the preview)")
        if plan.review_flags:
            lines.append("")
            lines.append("Plan-wide flags:")
            lines.extend(f"  • {flag}" for flag in plan.review_flags[:5])
        lines += [
            "",
            "The executor refuses a plan with review flags unless you explicitly override.",
            "",
            "Yes — execute ALL groups exactly as previewed, including the flagged ones.",
            "No — stop here; nothing will be moved or changed.",
        ]
        reply = QtWidgets.QMessageBox.question(
            self,
            "Groups flagged for review",
            "\n".join(lines),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return False, False
        return True, True

    def _planned_operation_count(self) -> int:
        plan = self._plan
        losers = sum(len(g.losers) for g in plan.groups)
        artwork = sum(len(g.artwork) for g in plan.groups)
        tags = sum(1 for g in plan.groups if g.metadata_changes)
        rewrites: set[str] = set()
        for group in plan.groups:
            rewrites.update(group.playlist_rewrites)
        return losers + artwork + tags + len(rewrites)

    @Slot()
    def _on_execute(self) -> None:
        if self._plan is None or not self._groups:
            QtWidgets.QMessageBox.warning(self, "No Plan", "No consolidation plan — run a scan first.")
            return
        if self._plan_executed:
            QtWidgets.QMessageBox.information(
                self, "Plan already executed", "This plan has been executed. Scan again to build a fresh plan."
            )
            return
        if self._scan_running or self._exec_running:
            return
        if self._library_writes_blocked("Duplicate execution"):
            return

        proceed, allow_review = self._review_gate()
        if not proceed:
            msg = "Nothing was moved: the plan has groups flagged for review."
            self._prog_status.setText(msg)
            self._log(msg, "warn")
            self._status("Not executed", "warn")
            return

        effective = self._effective_dispositions()
        try:
            self._apply_dispositions(effective)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(
                self, "Preview not updated",
                f"Could not update the preview with your disposition choices:\n{exc}\n\n"
                "Nothing was changed.",
            )
            return

        quarantine = sum(1 for d in effective.values() if d == "quarantine")
        delete = sum(1 for d in effective.values() if d == "delete")
        retain = sum(1 for d in effective.values() if d == "retain")
        tag_updates = sum(1 for g in self._plan.groups if g.metadata_changes)
        artwork = sum(len(g.artwork) for g in self._plan.groups)
        operations = self._planned_operation_count()
        overage = operations > _EXECUTOR_OPERATION_LIMIT

        lines = [f"Execute the duplicate plan for {len(self._plan.groups)} group(s)?", ""]
        if quarantine:
            lines.append(f"• {quarantine} duplicate file(s) will be moved to Quarantine/.")
        if delete:
            lines.append(f"• {delete} duplicate file(s) will be PERMANENTLY DELETED.")
        if retain:
            lines.append(f"• {retain} duplicate file(s) will stay where they are.")
        if tag_updates or artwork:
            lines.append(
                f"• Tags will be updated on {tag_updates} kept file(s); "
                f"{artwork} artwork change(s) will be applied."
            )
        lines.append("• Playlists that reference a moved file are rewritten to the kept copy (backups kept).")
        if overage:
            lines.append(
                f"• This is a large plan: {operations} operations, above the usual "
                f"{_EXECUTOR_OPERATION_LIMIT:,} safety limit. Confirming runs all of them."
            )
        lines += [
            "",
            "The preview (Docs/duplicate_preview.html) shows exactly this plan. "
            "Every action is logged under Docs/duplicate_execution_reports/.",
        ]
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute", "\n".join(lines),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        allow_deletion = False
        if delete:
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                "Confirm permanent deletion",
                f"{delete} file(s) will be permanently deleted. They will not go to "
                "Quarantine and cannot be recovered.\n\n"
                f"Type {_DELETE_CONFIRM_WORD} to confirm:",
            )
            if not ok or text.strip() != _DELETE_CONFIRM_WORD:
                msg = "Deletion not confirmed — nothing was changed."
                self._prog_status.setText(msg)
                self._log(msg, "warn")
                return
            allow_deletion = True

        self._prog_status.setText("Executing…")
        self._prog_bar.setValue(0)
        self._set_step(self._STEP_EXECUTE)
        self._status("Executing…", "busy")

        self._exec_worker = DupeExecuteWorker(
            plan=self._plan,
            library_path=self._library_path,
            allow_review_required=allow_review,
            allow_deletion=allow_deletion,
            confirm_operation_overage=overage,
        )
        self._exec_worker.log_line.connect(self._on_log_line)
        self._exec_worker.done.connect(self._on_execute_finished)
        self._exec_running = True
        self._refresh_buttons()
        self._start_worker(self._exec_worker, "Duplicate execution", library_write=True)

    @Slot()
    def _on_open_report(self) -> None:
        report = self._last_report_path
        if not report or not Path(report).exists():
            report = _latest_execution_report(self._library_path)
        if report and Path(report).exists():
            self._last_report_path = report
            webbrowser.open(Path(report).resolve().as_uri())
            return
        QtWidgets.QMessageBox.information(
            self, "No Report",
            "No execution report yet — reports are written when you Execute. "
            "Use Preview to see the plan.",
        )

    @Slot(object, object)
    def _on_group_selected(self, current, previous) -> None:  # noqa: ANN001
        if current is None:
            self._group_details.clear()
            self._disposition_combo.setEnabled(False)
            self._selected_lbl.setText("Select a group to choose what happens to its copies.")
            return
        index = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        group = self._groups[index] if isinstance(index, int) and 0 <= index < len(self._groups) else None
        if group is None:
            self._group_details.setPlainText(f"Group: {current.text(0)}")
            self._disposition_combo.setEnabled(False)
            return
        self._selected_lbl.setText(
            f"Group {index + 1} of {len(self._groups)} · {Path(group.winner_path).name}"
        )
        self._selected_lbl.setToolTip(group.winner_path)
        override = self._group_overrides.get(group.group_id)
        combo_index = next(
            (i for i, (_label, value) in enumerate(_DISPOSITION_CHOICES) if value == override), 0
        )
        self._disposition_combo.blockSignals(True)
        self._disposition_combo.setCurrentIndex(combo_index)
        self._disposition_combo.blockSignals(False)
        self._disposition_combo.setEnabled(not self._plan_executed and not self._exec_running)
        self._group_details.setPlainText(self._group_detail_text(group))

    @Slot(int)
    def _on_disposition_changed(self, combo_index: int) -> None:
        item = self._groups_table.currentItem()
        if item is None:
            return
        index = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or not (0 <= index < len(self._groups)):
            return
        group = self._groups[index]
        value = self._disposition_combo.itemData(combo_index)
        if value:
            self._group_overrides[group.group_id] = value
        else:
            self._group_overrides.pop(group.group_id, None)
        self._group_details.setPlainText(self._group_detail_text(group))
        self._groups_table.viewport().update()
        self._refresh_plan_stats()

    @Slot(int)
    def _on_scan_progress(self, value: int) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return
        self._prog_bar.setValue(value)

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._worker and sender is not self._exec_worker:
            return  # a job from before a library change
        self._log_area.appendPlainText(line)

    @Slot(bool, str)
    def _on_scan_finished(self, success: bool, message: str) -> None:
        worker = self.sender()
        if worker is not None and worker is not self._worker:
            return  # stale scan (library changed meanwhile); its results are discarded
        self._scan_running = False
        if self._worker is not None:
            self._preview_html_path = self._worker.preview_html_path
            if success:
                self._plan = self._worker.plan

        self._prog_status.setText(message)

        if success:
            self._log(message, "ok")
            self._status("Scan complete", "ok")
            self._set_step(self._STEP_REVIEW if self._groups else self._STEP_SCAN)
            if not self._groups:
                self._show_empty("No duplicates found",
                                 "Every track in this library is unique at these thresholds.")
        elif message == "Cancelled.":
            self._log(message, "warn")
            self._status("Cancelled", "warn")
            self._plan = None
            self._show_empty("Scan cancelled", "Nothing was changed. Find duplicates starts a fresh scan.")
        else:
            self._log(message, "error")
            self._status("Error", "error")
            self._plan = None
            self._show_empty("No groups", "The scan did not finish: the status above and the log say why.")
        self._refresh_buttons()

    @Slot(bool, str, str)
    def _on_execute_finished(self, success: bool, message: str, report_path: str) -> None:
        worker = self.sender()
        if worker is not None and worker is not self._exec_worker:
            # Execution of the previous library's plan, stopped by a library change.
            where = f" Report: {report_path}" if report_path else ""
            self._log(f"Previous library's duplicate execution ended: {message}{where}", "warn")
            return
        self._exec_running = False
        self._plan_executed = True
        self._prog_status.setText(f"{message} Scan again to build a fresh plan.")
        counts = getattr(worker if worker is not None else self._exec_worker, "counts", None)
        self._show_outcome(counts, message)

        if success:
            self._log(message, "ok")
            self._status("Execution complete", "ok")
            self._prog_bar.setValue(100)
        else:
            self._log(message, "error")
            self._status("Not executed" if message.startswith("Execution blocked") else "Error", "error")

        if report_path and Path(report_path).exists():
            self._last_report_path = report_path
            webbrowser.open(Path(report_path).resolve().as_uri())

        self._disposition_combo.setEnabled(False)
        self._refresh_buttons()

    # ── Helpers ───────────────────────────────────────────────────────────

    @Slot(list)
    def _populate_groups(self, groups: list) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return
        self._groups = list(groups)
        self._groups_table.clear()
        items = []
        for i, group in enumerate(self._groups):
            item = QtWidgets.QTreeWidgetItem([Path(group.winner_path).name])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, i)
            tip = [f"Keeps: {group.winner_path}"]
            tip += [f"Loser: {loser}" for loser in group.losers]
            if group.review_flags:
                tip.append("Flagged for review: " + "; ".join(group.review_flags))
            item.setToolTip(0, "\n".join(tip))
            items.append(item)
        self._groups_table.addTopLevelItems(items)
        flagged = sum(1 for g in self._groups if g.review_flags)
        text = f"{len(self._groups):,} group(s)"
        if flagged:
            text += f" · {flagged:,} flagged for review"
        self._fp_status_lbl.setText(text)
        self._outcome_stats.setVisible(False)
        self._refresh_plan_stats()
        if self._groups:
            self._fit_list_height()
            self._show_results()

    def _abandon_running_jobs(self, reason: str, *, keep_confirmed_writes: bool = False) -> None:
        """Detach running jobs from the page; their late results are ignored.

        The scan is read-only and is always cancelled. A confirmed execution is
        cancelled unless ``keep_confirmed_writes`` — on a library change it
        finishes the plan the user confirmed (for the previous library), like
        every other page's confirmed writes; the library-write lock keeps a new
        execution from starting meanwhile.
        """
        if self._scan_running and self._worker is not None:
            self._worker.cancel()
            self._log(f"{reason}: cancelled the running duplicate scan.", "warn")
        if self._exec_running and self._exec_worker is not None:
            if keep_confirmed_writes:
                self._log(
                    f"{reason}: the confirmed duplicate execution for the previous library "
                    "keeps running to completion; its report goes to that library's Docs/ folder.",
                    "warn",
                )
            else:
                self._exec_worker.cancel()
                self._log(
                    f"{reason}: asked the running duplicate execution to stop at the next safe "
                    "point. Its report is still written to the previous library's Docs/ folder.",
                    "warn",
                )
        self._worker = None
        self._exec_worker = None
        self._scan_running = False
        self._exec_running = False

    def _on_library_changed(self, path: str) -> None:
        self._abandon_running_jobs("Library changed", keep_confirmed_writes=True)
        self._library_lbl.setText(path)
        self._groups_table.clear()
        self._group_details.clear()
        self._log_area.clear()
        self._fp_status_lbl.setText("")
        self._outcome_stats.setVisible(False)
        self._show_empty(*self._initial_empty_text(path))
        self._groups = []
        self._plan = None
        self._plan_executed = False
        self._group_overrides.clear()
        self._preview_html_path = ""
        self._last_report_path = _latest_execution_report(path)
        self._prog_bar.setValue(0)
        self._prog_status.setText("Idle")
        self._disposition_combo.setEnabled(False)
        self._set_step(self._STEP_SCAN)
        self._refresh_buttons()
