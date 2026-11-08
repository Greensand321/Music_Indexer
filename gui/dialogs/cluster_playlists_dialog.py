"""Save cluster playlists — the preview (docs/specs/cluster_playlists.md, CP-39).

Shows one row per cluster of the analysis: whether it is saved, its colour,
its (editable) name, how many tracks it has and how many are missing, and
what Save would do with it (New, Replace or Skip and why). Every option or
checkbox change, and every *committed* name edit, re-plans on the GUI thread
with the pure ``cluster_playlist_export.plan_cluster_playlists``; nothing is
written here. After ``exec()`` returns Accepted the caller reads
:meth:`ClusterPlaylistsDialog.plan` and writes exactly that plan.

Theming: object names and theme roles only (no colours, font sizes or inline
style sheets; see tests/test_gui_hygiene_ratchets.py).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Iterable

import cluster_playlist_export as cpe
from gui.compat import QtCore, QtGui, QtWidgets

_COL_SAVE, _COL_COLOUR, _COL_NAME, _COL_TRACKS, _COL_MISSING, _COL_ACTION = range(6)
_HEADERS = ("Save", "", "Name", "Tracks", "Missing", "Action")
_ACTION_TEXT = {cpe.ACTION_NEW: "New", cpe.ACTION_REPLACE: "Replace", cpe.ACTION_SKIP: "Skip"}
_ROLE_CLUSTER = QtCore.Qt.ItemDataRole.UserRole
_MISSING_TOOLTIP_LIMIT = 20


def _created_text(run: dict) -> str:
    try:
        return datetime.fromisoformat(str(run.get("created") or "")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""


class ClusterPlaylistsDialog(QtWidgets.QDialog):
    """Modal preview of the cluster playlists Save would write."""

    def __init__(
        self,
        summary: cpe.RunSummary,
        preselect: Iterable[int],
        colour_for: Callable[[int], QtGui.QColor | None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("clusterPlaylistsDialog")
        self.setWindowTitle("Save cluster playlists")
        self.setModal(True)
        self.setMinimumSize(780, 480)
        self._summary = summary
        self._colour_for = colour_for
        self._selected: set[int] = {int(c) for c in preselect if int(c) in summary.clusters}
        self._names: dict[int, str] = {}
        self._plan: cpe.ClusterPlaylistPlan | None = None
        self._populating = False
        self._build_ui()
        self._replan()

    # ── Public ────────────────────────────────────────────────────────────

    def plan(self) -> cpe.ClusterPlaylistPlan:
        """The plan the dialog currently shows (what Save writes)."""
        assert self._plan is not None
        return self._plan

    def selected_clusters(self) -> set[int]:
        return set(self._selected)

    def order(self) -> str:
        return str(self._order_combo.currentData())

    def min_size(self) -> int:
        return int(self._min_spin.value())

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        s = self._summary
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Save a playlist per cluster")
        title.setObjectName("cardTitle")
        root.addWidget(title)

        run = s.run
        n_clusters = len(s.clusters)
        bits = [cpe.method_label(run), f"{n_clusters} cluster{'s' if n_clusters != 1 else ''}",
                f"{int(run.get('track_count') or 0):,} tracks"]
        created = _created_text(run)
        if created:
            bits.append(f"analysed {created}")
        header_lines = [" · ".join(bits), f"Destination: {s.run_folder_rel}"]
        if s.downsampled:
            header_lines.append(
                f"The map shows {s.map_points:,} of {s.map_total:,} tracks; "
                "the playlists include all of them."
            )
        self._header = QtWidgets.QLabel("\n".join(header_lines))
        self._header.setObjectName("clusterPlaylistsHeader")
        self._header.setWordWrap(True)
        self._header.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._header)

        self._warnings = QtWidgets.QLabel("\n".join(f"⚠ {w}" for w in s.warnings))
        self._warnings.setObjectName("statusHint")
        self._warnings.setWordWrap(True)
        self._warnings.setVisible(bool(s.warnings))
        root.addWidget(self._warnings)

        # ── Options ────────────────────────────────────────────────────────
        options = QtWidgets.QHBoxLayout()
        options.setSpacing(8)
        options.addWidget(QtWidgets.QLabel("Order:"))
        self._order_combo = QtWidgets.QComboBox()
        self._order_combo.setObjectName("clusterPlaylistsOrder")
        self._order_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        for key, text in cpe.ORDERS.items():
            self._order_combo.addItem(text, key)
        # Room for the longest label plus the theme's padding and arrow.
        longest = max(cpe.ORDERS.values(), key=len)
        self._order_combo.setMinimumWidth(self._order_combo.fontMetrics().horizontalAdvance(longest) + 64)
        self._order_combo.setCurrentIndex(self._order_combo.findData(cpe.DEFAULT_ORDER))
        self._order_combo.setToolTip(
            "Closest to the centre first puts each cluster's most typical tracks first; "
            "library order follows artist, album and track."
        )
        self._order_combo.currentIndexChanged.connect(self._on_option_changed)
        options.addWidget(self._order_combo)
        options.addSpacing(16)
        options.addWidget(QtWidgets.QLabel("Minimum tracks:"))
        self._min_spin = QtWidgets.QSpinBox()
        self._min_spin.setObjectName("clusterPlaylistsMinSize")
        self._min_spin.setRange(*cpe.MIN_SIZE_RANGE)
        self._min_spin.setValue(cpe.DEFAULT_MIN_SIZE)
        self._min_spin.setMinimumWidth(self._min_spin.fontMetrics().horizontalAdvance("500") + 48)
        self._min_spin.setToolTip(
            "Clusters with fewer tracks present than this are not saved. This only "
            "affects the playlists, not the analysis."
        )
        self._min_spin.valueChanged.connect(self._on_option_changed)
        options.addWidget(self._min_spin)
        options.addStretch(1)
        root.addLayout(options)

        # ── Table ──────────────────────────────────────────────────────────
        self.table = QtWidgets.QTableWidget(0, len(_HEADERS))
        self.table.setObjectName("clusterPlaylistsTable")
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(_COL_NAME, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table, 1)

        self._left_unchanged = QtWidgets.QLabel("")
        self._left_unchanged.setObjectName("statusHint")
        self._left_unchanged.setWordWrap(True)
        self._left_unchanged.setVisible(False)
        root.addWidget(self._left_unchanged)

        note = QtWidgets.QLabel(
            "Double-click a name to change it. Nothing is written until you click Save; "
            "a playlist that is replaced is backed up beside it first, and every write "
            "is logged to Docs/playlist_writes.log."
        )
        note.setObjectName("notesHint")
        note.setWordWrap(True)
        root.addWidget(note)

        # ── Buttons ────────────────────────────────────────────────────────
        buttons = QtWidgets.QHBoxLayout()
        self._all_btn = QtWidgets.QPushButton("Select all")
        self._all_btn.clicked.connect(self._select_all)
        self._none_btn = QtWidgets.QPushButton("Select none")
        self._none_btn.clicked.connect(self._select_none)
        self._open_btn = QtWidgets.QPushButton("Open folder")
        self._open_btn.setToolTip(s.run_folder)
        self._open_btn.clicked.connect(self._open_folder)
        self._open_btn.setVisible(bool(s.run_folder) and os.path.isdir(s.run_folder))
        for b in (self._all_btn, self._none_btn, self._open_btn):
            b.setAutoDefault(False)
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.setObjectName("primaryBtn")
        self.save_btn.setAutoDefault(False)
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setDefault(True)   # Enter must not write files
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.cancel_btn)
        root.addLayout(buttons)

    # ── Planning ──────────────────────────────────────────────────────────

    def _replan(self) -> None:
        self._plan = cpe.plan_cluster_playlists(
            self._summary,
            self._selected,
            order=self.order(),
            min_size=self.min_size(),
            names=self._names,
        )
        self._populate()

    def _populate(self) -> None:
        plan = self._plan
        self._populating = True
        try:
            # The rows are always the run's clusters in the same order, so the
            # items are created once and then updated in place: replacing an
            # item from inside its own itemChanged would delete it mid-signal.
            if self.table.rowCount() != len(plan.rows):
                self.table.setRowCount(len(plan.rows))
                for r in range(len(plan.rows)):
                    for c in range(len(_HEADERS)):
                        self.table.setItem(r, c, QtWidgets.QTableWidgetItem())
            for r, row in enumerate(plan.rows):
                self._fill_row(r, row)
        finally:
            self._populating = False

        n = len(plan.items)
        k = plan.replace_count
        label = f"Save {n} playlist{'s' if n != 1 else ''}"
        if k:
            label += f" ({k} replace)"
        self.save_btn.setText(label)
        self.save_btn.setEnabled(n > 0 and plan.ok)
        if not plan.ok:
            self.save_btn.setToolTip("; ".join(plan.errors))
        elif n == 0:
            self.save_btn.setToolTip("Tick at least one cluster to save")
        else:
            self.save_btn.setToolTip(f"Write {n} playlist{'s' if n != 1 else ''} to {self._summary.run_folder_rel}")

        if plan.left_unchanged:
            shown = ", ".join(plan.left_unchanged[:5])
            more = len(plan.left_unchanged) - 5
            if more > 0:
                shown += f" and {more} more"
            self._left_unchanged.setText(
                f"Left unchanged in this folder (not part of this save): {shown}"
            )
        self._left_unchanged.setVisible(bool(plan.left_unchanged))

    def _fill_row(self, r: int, row: cpe.PlanRow) -> None:
        cid = row.cluster_id
        noise = cid == cpe.NOISE_LABEL
        eligible = not noise and not (
            row.action == cpe.ACTION_SKIP and row.reason != cpe.NOT_SELECTED_REASON
        )

        enabled = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter

        save = self.table.item(r, _COL_SAVE)
        save.setData(_ROLE_CLUSTER, cid)
        if noise:
            # Noise is never a playlist (CP-19): no checkbox at all.
            save.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable)
            save.setToolTip("Unclustered tracks are never saved as a playlist.")
        elif eligible:
            save.setFlags(enabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            save.setCheckState(
                QtCore.Qt.CheckState.Checked if cid in self._selected
                else QtCore.Qt.CheckState.Unchecked
            )
            save.setToolTip("Save this cluster as a playlist")
        else:
            # Too small / path too long: unticked and locked, the reason as tooltip.
            save.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            save.setCheckState(QtCore.Qt.CheckState.Unchecked)
            save.setToolTip(row.reason)

        colour = self.table.item(r, _COL_COLOUR)
        colour.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        if colour.icon().isNull():
            qcolour = self._colour(cid)
            if qcolour is not None:
                pix = QtGui.QPixmap(12, 12)
                pix.fill(qcolour)
                colour.setIcon(QtGui.QIcon(pix))

        name = self.table.item(r, _COL_NAME)
        name.setData(_ROLE_CLUSTER, cid)
        if noise:
            name.setText(f"Unclustered ({row.tracks}) — not saved")
            name.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable)
            name.setToolTip("Tracks the analysis put in no cluster are never saved as a playlist.")
        else:
            name.setText(row.name)
            name.setFlags(enabled | QtCore.Qt.ItemFlag.ItemIsEditable)
            name.setToolTip(f"{self._summary.clusters[cid].label} · double-click to rename")

        tracks = self.table.item(r, _COL_TRACKS)
        tracks.setText(str(row.tracks))
        tracks.setFlags(enabled)
        tracks.setTextAlignment(right)

        missing = self.table.item(r, _COL_MISSING)
        missing.setText("" if noise else str(row.missing))
        missing.setFlags(enabled)
        missing.setTextAlignment(right)
        tip = ""
        if not noise and row.missing:
            paths = self._summary.clusters[cid].missing
            tip = "Left out (missing, or in a reserved folder):\n" + "\n".join(paths[:_MISSING_TOOLTIP_LIMIT])
            if len(paths) > _MISSING_TOOLTIP_LIMIT:
                tip += f"\n… and {len(paths) - _MISSING_TOOLTIP_LIMIT} more"
        missing.setToolTip(tip)

        text = _ACTION_TEXT.get(row.action, row.action)
        if noise:
            text = "—"
        elif row.reason:
            text = f"{text} · {row.reason}"
        action = self.table.item(r, _COL_ACTION)
        action.setText(text)
        action.setFlags(enabled)
        if row.action == cpe.ACTION_REPLACE:
            action.setToolTip("A playlist with this name is already in the folder; it is backed up first.")
        else:
            action.setToolTip(row.reason)

    def _colour(self, cluster_id: int) -> QtGui.QColor | None:
        if self._colour_for is None:
            return None
        try:
            colour = self._colour_for(cluster_id)
        except Exception:  # noqa: BLE001 - a colour is decoration only
            return None
        return QtGui.QColor(colour) if colour is not None else None

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_option_changed(self, *_args) -> None:
        self._replan()

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._populating:
            return
        cid = item.data(_ROLE_CLUSTER)
        if cid is None or int(cid) == cpe.NOISE_LABEL:
            return
        cid = int(cid)
        if item.column() == _COL_SAVE:
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                self._selected.add(cid)
            else:
                self._selected.discard(cid)
        elif item.column() == _COL_NAME:
            self._names[cid] = item.text()
        else:
            return
        # Items are updated in place (never replaced), so re-planning inside
        # the signal is safe.
        self._replan()

    def _select_all(self) -> None:
        self._selected = set(self._summary.clusters)
        self._replan()

    def _select_none(self) -> None:
        self._selected.clear()
        self._replan()

    def _open_folder(self) -> None:
        folder = self._summary.run_folder
        if folder and os.path.isdir(folder):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))
