"""Duplicate Pair Review dialog — Qt port of DuplicatePairReviewTool."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from gui.compat import QtCore, QtGui, QtWidgets, Slot

# ── Audio extensions (same set as simple_duplicate_finder) ─────────────────
_SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}

_SNAPSHOT_FILENAME = "duplicate_pair_review.json"
_PREVIEW_FILENAME  = "duplicate_preview.json"
_COVER_SIZE        = 120


# ── Data types ───────────────────────���───────────────────────────���─────────

@dataclass
class _Pair:
    left_path:     str
    right_path:    str
    winner_path:   str | None = None
    source:        str = "plan"
    manual_winner: str | None = None


# ── Pair-building helpers (self-contained, no main_gui dependency) ──────────

def _collect_filename_pairs(library_path: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"\s*\(\d+\)$")
    candidates: dict[str, list[str]] = {}
    for dirpath, _dirs, files in os.walk(library_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in _SUPPORTED_EXTS:
                continue
            stem = os.path.splitext(filename)[0].lower()
            normalized = pattern.sub("", stem).strip()
            if not normalized:
                continue
            candidates.setdefault(normalized, []).append(
                os.path.join(dirpath, filename)
            )
    pairs: list[tuple[str, str]] = []
    for items in candidates.values():
        if len(items) < 2:
            continue
        sorted_items = sorted(items, key=str.lower)
        anchor = sorted_items[0]
        for other in sorted_items[1:]:
            pairs.append((anchor, other))
    return pairs


def _append_pair(
    pairs: list[_Pair],
    seen:  set[tuple[str, str]],
    left:  str,
    right: str,
    winner_path: str | None,
    source: str,
    manual_winner: str | None = None,
) -> None:
    if left == right:
        return
    key = tuple(sorted((left, right)))
    if key in seen:
        return
    seen.add(key)
    pairs.append(_Pair(left_path=left, right_path=right, winner_path=winner_path,
                       source=source, manual_winner=manual_winner))


def _build_pairs_from_plan(plan, library_path: str) -> list[_Pair]:
    """Build pair list from a ConsolidationPlan object."""
    pairs: list[_Pair] = []
    seen:  set[tuple[str, str]] = set()
    for group in plan.groups:
        if not group.losers:
            continue
        for loser in group.losers:
            _append_pair(pairs, seen, group.winner_path, loser,
                         group.winner_path, "plan")
    for left, right in _collect_filename_pairs(library_path):
        _append_pair(pairs, seen, left, right, None, "filename")
    return pairs


def _load_pairs_from_snapshot(library_path: str) -> list[_Pair] | None:
    snap_path = os.path.join(library_path, "Docs", _SNAPSHOT_FILENAME)
    if not os.path.exists(snap_path):
        return None
    try:
        with open(snap_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        return None
    pairs: list[_Pair] = []
    seen:  set[tuple[str, str]] = set()
    for item in raw_pairs:
        if not isinstance(item, Mapping):
            continue
        left  = item.get("left_path")
        right = item.get("right_path")
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        winner       = item.get("winner_path")
        manual_w     = item.get("manual_winner")
        source       = item.get("source") if isinstance(item.get("source"), str) else "snapshot"
        _append_pair(pairs, seen, left, right,
                     winner if isinstance(winner, str) else None,
                     source,
                     manual_w if isinstance(manual_w, str) else None)
    return pairs or None


def _load_pairs_from_preview(library_path: str) -> list[_Pair] | None:
    preview_path = os.path.join(library_path, "Docs", _PREVIEW_FILENAME)
    try:
        with open(preview_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "plan" not in payload:
        return None
    try:
        from duplicate_consolidation import consolidation_plan_from_dict
        plan = consolidation_plan_from_dict(payload["plan"])
    except (ImportError, ValueError):
        return None
    return _build_pairs_from_plan(plan, library_path)


def load_pairs(library_path: str) -> list[_Pair] | None:
    """Return pairs from snapshot → preview fallback. None if neither exists."""
    pairs = _load_pairs_from_snapshot(library_path)
    if pairs:
        return pairs
    return _load_pairs_from_preview(library_path)


# ── Track-panel widget ───────────────────────���──────────────────────────────

class _TrackPanel(QtWidgets.QFrame):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)

        self._header_lbl = QtWidgets.QLabel(f"<b>{title}</b>")
        layout.addWidget(self._header_lbl)

        self._cover_lbl = QtWidgets.QLabel()
        self._cover_lbl.setFixedSize(_COVER_SIZE, _COVER_SIZE)
        self._cover_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._cover_lbl.setStyleSheet("background: #333; border-radius: 4px;")
        layout.addWidget(self._cover_lbl, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        self._winner_lbl = QtWidgets.QLabel()
        self._winner_lbl.setObjectName("sectionSubtitle")
        layout.addWidget(self._winner_lbl)

        self._title_lbl = QtWidgets.QLabel()
        self._title_lbl.setWordWrap(True)
        font = self._title_lbl.font()
        font.setBold(True)
        self._title_lbl.setFont(font)
        layout.addWidget(self._title_lbl)

        self._meta_lbl = QtWidgets.QLabel()
        self._meta_lbl.setWordWrap(True)
        self._meta_lbl.setObjectName("sectionSubtitle")
        layout.addWidget(self._meta_lbl)

        self._path_lbl = QtWidgets.QLabel()
        self._path_lbl.setWordWrap(True)
        self._path_lbl.setObjectName("mutedLabel")
        layout.addWidget(self._path_lbl)

        layout.addStretch(1)

    def load(
        self,
        tags: dict,
        path: str,
        *,
        is_winner: bool,
        has_winner: bool,
        cover_bytes: bytes | None,
    ) -> None:
        # Cover art
        pm = QtGui.QPixmap(_COVER_SIZE, _COVER_SIZE)
        pm.fill(QtGui.QColor("#333333"))
        if cover_bytes:
            raw = QtGui.QPixmap()
            if raw.loadFromData(cover_bytes):
                raw = raw.scaled(
                    _COVER_SIZE, _COVER_SIZE,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                pm = raw
        self._cover_lbl.setPixmap(pm)

        # Winner badge
        if not has_winner:
            self._winner_lbl.setText("Winner: not set")
            self._winner_lbl.setStyleSheet("color: #888;")
        elif is_winner:
            self._winner_lbl.setText("\u2705 Winner (kept)")
            self._winner_lbl.setStyleSheet("color: #22c55e;")
        else:
            self._winner_lbl.setText("Not winner")
            self._winner_lbl.setStyleSheet("color: #888;")

        # Title
        artist = tags.get("artist") or ""
        title  = tags.get("title")  or ""
        if artist or title:
            display = f"{artist} \u2013 {title}" if artist and title else (artist or title)
        else:
            display = os.path.basename(path)
        self._title_lbl.setText(str(display))

        # Metadata
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "unknown"
        lines = [f"Format: {ext}"]
        for field_name, tag_key in [
            ("Artist", "artist"), ("Title", "title"), ("Album", "album"),
            ("Year",   "year"),   ("Track", "track"), ("Genre", "genre"),
        ]:
            val = tags.get(tag_key)
            if val:
                lines.append(f"{field_name}: {val}")
        self._meta_lbl.setText("\n".join(str(l) for l in lines))
        self._path_lbl.setText(path)

    def clear(self) -> None:
        pm = QtGui.QPixmap(_COVER_SIZE, _COVER_SIZE)
        pm.fill(QtGui.QColor("#333333"))
        self._cover_lbl.setPixmap(pm)
        self._winner_lbl.setText("")
        self._title_lbl.setText("")
        self._meta_lbl.setText("")
        self._path_lbl.setText("")


# ── Main dialog ──────────────────────────���─────────────────────────────���────

class PairReviewDialog(QtWidgets.QDialog):
    """Review duplicate pairs one-by-one, mirroring the Tkinter DuplicatePairReviewTool."""

    def __init__(
        self,
        library_path: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Duplicate Pair Review")
        self.setMinimumSize(760, 580)
        self.resize(880, 640)

        self._library_path = library_path
        self._pairs: list[_Pair] = []
        self._index = 0

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(8)

        # Header
        title_lbl = QtWidgets.QLabel("<b>Duplicate Pair Review</b>")
        root.addWidget(title_lbl)
        desc = QtWidgets.QLabel(
            "Review paired duplicates. <b>Yes</b> deletes the inferior file (prefers FLAC). "
            "<b>No</b> skips. <b>Tag</b> appends the pair to Docs/duplicate_pair_tags.txt."
        )
        desc.setWordWrap(True)
        desc.setObjectName("sectionSubtitle")
        root.addWidget(desc)

        # Progress
        self._progress_lbl = QtWidgets.QLabel("")
        root.addWidget(self._progress_lbl)

        # Two-panel comparison
        panels_row = QtWidgets.QHBoxLayout()
        self._left_panel  = _TrackPanel("Left Track")
        self._right_panel = _TrackPanel("Right Track")
        panels_row.addWidget(self._left_panel,  1)
        panels_row.addWidget(self._right_panel, 1)
        root.addLayout(panels_row)

        # Status
        self._status_lbl = QtWidgets.QLabel("Ready")
        self._status_lbl.setObjectName("sectionSubtitle")
        root.addWidget(self._status_lbl)

        # Navigation row
        nav_row = QtWidgets.QHBoxLayout()
        self._prev_btn   = QtWidgets.QPushButton("\u25c4 Prev")
        self._next_btn   = QtWidgets.QPushButton("Next \u25ba")
        self._switch_btn = QtWidgets.QPushButton("Switch")
        self._mp3_btn    = QtWidgets.QPushButton("Prefer MP3")
        self._prev_btn.clicked.connect(self._go_previous)
        self._next_btn.clicked.connect(self._go_next)
        self._switch_btn.clicked.connect(self._handle_switch)
        self._mp3_btn.clicked.connect(self._handle_prefer_mp3)
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        nav_row.addWidget(self._switch_btn)
        nav_row.addWidget(self._mp3_btn)
        nav_row.addSpacing(16)
        nav_row.addWidget(QtWidgets.QLabel("Jump to:"))
        self._jump_edit = QtWidgets.QLineEdit()
        self._jump_edit.setFixedWidth(54)
        self._jump_edit.returnPressed.connect(self._jump_to_pair)
        nav_row.addWidget(self._jump_edit)
        go_btn = QtWidgets.QPushButton("Go")
        go_btn.clicked.connect(self._jump_to_pair)
        nav_row.addWidget(go_btn)
        nav_row.addStretch(1)
        root.addLayout(nav_row)

        # Action row
        action_row = QtWidgets.QHBoxLayout()
        self._yes_btn = QtWidgets.QPushButton("Yes — delete inferior")
        self._yes_btn.setObjectName("primaryBtn")
        self._no_btn  = QtWidgets.QPushButton("No — skip")
        self._tag_btn = QtWidgets.QPushButton("Tag")
        close_btn     = QtWidgets.QPushButton("Close")
        self._yes_btn.clicked.connect(self._handle_yes)
        self._no_btn.clicked.connect(self._handle_no)
        self._tag_btn.clicked.connect(self._handle_tag)
        close_btn.clicked.connect(self.close)
        action_row.addWidget(self._yes_btn)
        action_row.addWidget(self._no_btn)
        action_row.addWidget(self._tag_btn)
        action_row.addStretch(1)
        action_row.addWidget(close_btn)
        root.addLayout(action_row)

        # Keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Return),    self, self._handle_yes)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Backspace), self, self._handle_no)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left),      self, self._go_previous)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right),     self, self._go_next)

        self._reload_pairs()

    # ── Pair loading ──────────────────────��────────────────────���────────────

    def _reload_pairs(self) -> None:
        pairs = load_pairs(self._library_path)
        if pairs is None:
            self._pairs = []
        else:
            self._pairs = pairs
        self._index = 0
        self._load_pair()

    # ── Navigation ───────────────────────���───────────────────────���─────────

    @Slot()
    def _go_previous(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._load_pair()

    @Slot()
    def _go_next(self) -> None:
        if self._index < len(self._pairs) - 1:
            self._index += 1
            self._load_pair()

    @Slot()
    def _jump_to_pair(self) -> None:
        raw = self._jump_edit.text().strip()
        if not raw:
            return
        try:
            idx = int(raw) - 1
        except ValueError:
            self._set_status("Invalid jump value.")
            return
        if idx < 0 or idx >= len(self._pairs):
            self._set_status("Jump out of range.")
            return
        self._index = idx
        self._load_pair()

    def _advance(self) -> None:
        self._index += 1
        self._load_pair()

    # ── Display ──────────────────────��──────────────────────────────────────

    def _load_pair(self) -> None:
        if not self._pairs:
            self._progress_lbl.setText("No paired duplicates found.")
            self._set_status("Idle")
            self._left_panel.clear()
            self._right_panel.clear()
            self._set_buttons_enabled(False)
            return

        # Skip missing files
        while self._index < len(self._pairs):
            pair = self._pairs[self._index]
            if os.path.exists(pair.left_path) and os.path.exists(pair.right_path):
                break
            self._set_status("Skipped missing files.")
            self._index += 1

        if self._index >= len(self._pairs):
            self._progress_lbl.setText("All pairs reviewed.")
            self._set_status("Done")
            self._left_panel.clear()
            self._right_panel.clear()
            self._set_buttons_enabled(False)
            return

        pair = self._pairs[self._index]
        self._progress_lbl.setText(f"Pair {self._index + 1} of {len(self._pairs)}")
        self._set_status("Ready")
        self._set_buttons_enabled(True)
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < len(self._pairs) - 1)

        winner_path = pair.manual_winner or pair.winner_path or self._infer_winner(pair)
        is_left_winner  = winner_path == pair.left_path  if winner_path else False
        is_right_winner = winner_path == pair.right_path if winner_path else False

        left_tags,  left_cover  = self._read_track(pair.left_path)
        right_tags, right_cover = self._read_track(pair.right_path)

        self._left_panel.load(
            left_tags, pair.left_path,
            is_winner=is_left_winner, has_winner=bool(winner_path),
            cover_bytes=left_cover,
        )
        self._right_panel.load(
            right_tags, pair.right_path,
            is_winner=is_right_winner, has_winner=bool(winner_path),
            cover_bytes=right_cover,
        )

    @staticmethod
    def _infer_winner(pair: _Pair) -> str | None:
        left_ext  = os.path.splitext(pair.left_path)[1].lower()
        right_ext = os.path.splitext(pair.right_path)[1].lower()
        if left_ext == ".flac" and right_ext != ".flac":
            return pair.left_path
        if right_ext == ".flac" and left_ext != ".flac":
            return pair.right_path
        return None

    @staticmethod
    def _read_track(path: str) -> tuple[dict, bytes | None]:
        try:
            from utils.audio_metadata_reader import read_metadata
            tags, covers, _err, _hint = read_metadata(path, include_cover=True)
            cover = covers[0] if covers else None
        except Exception:  # noqa: BLE001
            tags, cover = {}, None
        return tags, cover

    # ── Action handlers ─────────────────────────────��───────────────────────

    @Slot()
    def _handle_switch(self) -> None:
        if not self._pairs or self._index >= len(self._pairs):
            return
        pair = self._pairs[self._index]
        current_winner = pair.manual_winner or pair.winner_path
        pair.left_path, pair.right_path = pair.right_path, pair.left_path
        if current_winner == pair.right_path:      # was left before swap
            pair.manual_winner = pair.left_path    # now the old-right, new-left
        elif current_winner == pair.left_path:     # was right before swap
            pair.manual_winner = pair.right_path
        else:
            pair.manual_winner = pair.left_path
        self._load_pair()

    @Slot()
    def _handle_prefer_mp3(self) -> None:
        if not self._pairs or self._index >= len(self._pairs):
            return
        pair = self._pairs[self._index]
        left_ext  = os.path.splitext(pair.left_path)[1].lower()
        right_ext = os.path.splitext(pair.right_path)[1].lower()
        if {left_ext, right_ext} != {".m4a", ".mp3"}:
            self._set_status("No MP3/M4A pair to override.")
            return
        pair.manual_winner = pair.left_path if left_ext == ".mp3" else pair.right_path
        self._set_status("MP3 preferred over M4A.")
        self._load_pair()

    @Slot()
    def _handle_yes(self) -> None:
        if not self._pairs or self._index >= len(self._pairs):
            return
        pair = self._pairs[self._index]
        if not os.path.exists(pair.left_path) or not os.path.exists(pair.right_path):
            self._set_status("Missing files; skipped.")
            self._advance()
            return
        winner_path = pair.manual_winner or pair.winner_path
        delete_path = self._pick_inferior(pair.left_path, pair.right_path, winner_path)
        if not delete_path:
            self._set_status("No FLAC in pair; nothing deleted.")
            self._advance()
            return
        kept_path = pair.right_path if delete_path == pair.left_path else pair.left_path
        try:
            from utils.path_helpers import ensure_long_path
            long_del = ensure_long_path(delete_path)
            if not os.path.exists(long_del):
                self._set_status("Already removed.")
                self._advance()
                return
            os.remove(long_del)
            self._update_playlists(delete_path, kept_path)
            self._set_status(f"Deleted {os.path.basename(delete_path)}")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Delete Failed", str(exc))
            self._set_status("Delete failed")
        self._advance()

    @Slot()
    def _handle_no(self) -> None:
        if not self._pairs or self._index >= len(self._pairs):
            return
        pair = self._pairs[self._index]
        if not os.path.exists(pair.left_path) or not os.path.exists(pair.right_path):
            self._set_status("Missing files; skipped.")
        else:
            self._set_status("Skipped")
        self._advance()

    @Slot()
    def _handle_tag(self) -> None:
        if not self._pairs or self._index >= len(self._pairs):
            return
        pair = self._pairs[self._index]
        if not os.path.exists(pair.left_path) or not os.path.exists(pair.right_path):
            self._set_status("Missing files; skipped.")
            self._advance()
            return
        docs_dir = os.path.join(self._library_path, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        tag_path = os.path.join(docs_dir, "duplicate_pair_tags.txt")
        left_name  = self._display_name(pair.left_path)
        right_name = self._display_name(pair.right_path)
        try:
            with open(tag_path, "a", encoding="utf-8") as fh:
                fh.write(f"{left_name} | {right_name}\n")
            self._set_status("Tagged pair added")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Tag Failed", str(exc))
            self._set_status("Tag failed")
            return
        self._advance()

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_inferior(left: str, right: str, winner: str | None) -> str | None:
        if winner:
            if winner == left:
                return right
            if winner == right:
                return left
        le = os.path.splitext(left)[1].lower()
        re_ = os.path.splitext(right)[1].lower()
        if le == ".flac" and re_ != ".flac":
            return right
        if re_ == ".flac" and le != ".flac":
            return left
        return None

    def _update_playlists(self, deleted: str, kept: str) -> None:
        playlists_dir = os.path.join(self._library_path, "Playlists")
        if not os.path.isdir(playlists_dir):
            return
        norm_del = os.path.normcase(os.path.normpath(deleted))
        found = False
        for dirpath, _dirs, files in os.walk(playlists_dir):
            for fname in files:
                if not fname.lower().endswith((".m3u", ".m3u8")):
                    continue
                pl_path = os.path.join(dirpath, fname)
                try:
                    with open(pl_path, "r", encoding="utf-8") as fh:
                        lines = [ln.rstrip("\n") for ln in fh]
                except OSError:
                    continue
                for line in lines:
                    if os.path.normcase(os.path.normpath(
                        os.path.join(dirpath, line)
                    )) == norm_del:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            try:
                from playlist_generator import update_playlists
                update_playlists({deleted: kept})
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _display_name(path: str) -> str:
        try:
            from utils.audio_metadata_reader import read_tags
            tags = read_tags(path)
            artist = tags.get("artist") or ""
            title  = tags.get("title")  or ""
            if artist or title:
                return f"{artist} \u2013 {title}" if artist and title else (str(artist) or str(title))
        except Exception:  # noqa: BLE001
            pass
        return os.path.basename(path)

    def _set_status(self, text: str) -> None:
        self._status_lbl.setText(text)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self._yes_btn, self._no_btn, self._tag_btn,
                    self._switch_btn, self._mp3_btn):
            btn.setEnabled(enabled)
        if not enabled:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
