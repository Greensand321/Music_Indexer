"""Multi-step clustering configuration wizard dialog.

Also home to the two small controls the Sound Map's Analyse page shares with
the wizard, so both look and read the same:

``FEATURE_CHOICES``
    The six audio feature blocks in ``clustered_playlists.FEATURE_ORDER``
    with a plain-English label and a tooltip (a test pins the order to the
    engine's, without importing the engine into the GUI at start-up).
``PillToggle``
    A checkable pill (a check glyph and the selection tint when on, a faint
    well when off), painted from the active material.
"""
from __future__ import annotations

from typing import Dict

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import apply_font, current_material, font, paint_surface, qcolor

#: Shown on the controls the clustering engine does not implement yet. They
#: stay visible (the redesign decides their fate) but cannot be changed, so the
#: wizard no longer promises behaviour that never happens.
_NOT_AVAILABLE = "Not available yet — the clustering engine does not do this."

#: (engine key, label, what it listens for) in the engine's FEATURE_ORDER.
FEATURE_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("mfcc", "Timbre", "Timbre, the colour of the sound (MFCC)"),
    ("tempo", "Tempo", "Beats per minute"),
    ("chroma", "Harmony", "Harmonic content and key (chroma)"),
    ("spectral", "Brightness", "How bright the sound is (spectral centroid)"),
    ("energy", "Energy", "Loudness and how much it varies"),
    ("onset_rate", "Rhythm density", "How busy the rhythm is (onsets per second; slow to compute)"),
)
FEATURE_LABELS: Dict[str, str] = {key: label for key, label, _tip in FEATURE_CHOICES}

#: The wizard's steps, as the step pills name them.
STEP_NAMES = ("Features", "Scaling", "Algorithm", "Post-processing", "Output")

#: The "Presets" on the features step.
FEATURE_PRESETS = (
    ("Fast", {"tempo": True, "mfcc": False, "chroma": False, "spectral": False, "energy": True, "onset_rate": False}),
    ("Balanced", {"tempo": True, "mfcc": True, "chroma": False, "spectral": False, "energy": True, "onset_rate": False}),
    ("Complete", {"tempo": True, "mfcc": True, "chroma": True, "spectral": True, "energy": True, "onset_rate": False}),
)


def features_text(features: Dict[str, bool]) -> str:
    """"timbre, tempo and energy" for the ticked feature blocks."""
    names = [label.lower() for key, label, _tip in FEATURE_CHOICES if features.get(key)]
    if not names:
        return "no features"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


class PillToggle(QtWidgets.QAbstractButton):
    """A checkable pill: on = check glyph on the selection tint with an
    accent rim; off = a faint well. Hover, focus and disabled states are
    painted too. Behaves like a check box (``toggled``, ``isChecked``)."""

    HEIGHT = 34
    _PAD = 14
    _GLYPH = 14
    _GAP = 6

    def __init__(self, text: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setFont(font("label", weight=600))
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self._hover = False

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        fm = QtGui.QFontMetrics(self.font())
        w = fm.horizontalAdvance(self.text()) + 2 * self._PAD + self._GLYPH + self._GAP
        return QtCore.QSize(w, self.HEIGHT)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.sizeHint()

    def event(self, event: QtCore.QEvent) -> bool:  # noqa: D401
        if event.type() in (QtCore.QEvent.Type.HoverEnter, QtCore.QEvent.Type.HoverLeave):
            self._hover = event.type() == QtCore.QEvent.Type.HoverEnter
            self.update()
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager
        from gui.widgets.glyphs import paint_glyph

        t = get_manager().current
        m = current_material()
        on = self.isChecked()
        enabled = self.isEnabled()
        rect = QtCore.QRectF(self.rect())
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            if not enabled:
                p.setOpacity(0.5)
            if on:
                rim = qcolor(t.accent)
                rim.setAlphaF(0.55)
                paint_surface(p, rect, m.well, radius=rect.height() / 2, fill=qcolor(m.selection))
                body = rect.adjusted(0.5, 0.5, -0.5, -0.5)
                p.setPen(QtGui.QPen(rim, 1.0))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(body, body.height() / 2, body.height() / 2)
            else:
                paint_surface(p, rect, m.well, radius=rect.height() / 2)
            if self._hover and enabled:
                path = QtGui.QPainterPath()
                path.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
                p.fillPath(path, qcolor(m.hover))
            fm = QtGui.QFontMetrics(self.font())
            text_w = fm.horizontalAdvance(self.text())
            total = self._GLYPH + self._GAP + text_w
            left = rect.left() + max(self._PAD, (rect.width() - total) / 2)
            glyph = QtCore.QRectF(left, rect.center().y() - self._GLYPH / 2, self._GLYPH, self._GLYPH)
            if on:
                paint_glyph(p, glyph, "check", qcolor(t.accent if m.is_glass else m.selection_text), weight=2.2)
            else:
                paint_glyph(p, glyph, "plus", qcolor(t.text_muted), weight=1.8)
            p.setFont(self.font())
            p.setPen(qcolor(m.selection_text if on else t.text_secondary))
            text_rect = QtCore.QRectF(glyph.right() + self._GAP, rect.top(),
                                      max(0.0, rect.right() - glyph.right() - self._GAP - self._PAD / 2),
                                      rect.height())
            p.drawText(text_rect, int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                       fm.elidedText(self.text(), QtCore.Qt.TextElideMode.ElideRight, int(text_rect.width())))
            if self.hasFocus():
                p.setOpacity(1.0)
                ring = rect.adjusted(1, 1, -1, -1)
                p.setPen(QtGui.QPen(QtGui.QColor(t.input_focus), 1.5))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(ring, ring.height() / 2, ring.height() / 2)
        finally:
            p.end()


def feature_toggles(parent_layout: QtWidgets.QGridLayout, columns: int = 3) -> Dict[str, PillToggle]:
    """One ``PillToggle`` per feature block, laid out ``columns`` per row in
    ``parent_layout`` (a grid). Returns them keyed by engine name."""
    toggles: Dict[str, PillToggle] = {}
    for i, (key, label, tip) in enumerate(FEATURE_CHOICES):
        toggle = PillToggle(label)
        toggle.setObjectName(f"featureToggle_{key}")
        toggle.setToolTip(tip)
        toggle.setAccessibleName(f"Listen for {label.lower()}")
        parent_layout.addWidget(toggle, i // columns, i % columns)
        toggles[key] = toggle
    for c in range(columns):
        parent_layout.setColumnStretch(c, 1)
    return toggles


class _StepPills(QtWidgets.QWidget):
    """The wizard's progress as a compact row of pills: done steps carry a
    check, the current one is raised, the rest are muted."""

    HEIGHT = 30

    def __init__(self, names: tuple[str, ...], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._names = names
        self._current = 0
        self.setFont(font("caption", weight=600))
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_current(self, index: int) -> None:
        self._current = index
        self.setAccessibleName(f"Step {index + 1} of {len(self._names)}: {self._names[index]}")
        self.update()

    def current(self) -> int:
        return self._current

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        fm = QtGui.QFontMetrics(self.font())
        w = sum(fm.horizontalAdvance(n) + 30 for n in self._names) + 4 * (len(self._names) - 1)
        return QtCore.QSize(min(w, 420), self.HEIGHT)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.minimumSizeHint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager
        from gui.widgets.glyphs import paint_glyph

        t = get_manager().current
        m = current_material()
        n = len(self._names)
        gap = 4.0
        w = (self.width() - gap * (n - 1)) / n
        fm = QtGui.QFontMetrics(self.font())
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setFont(self.font())
            for i, name in enumerate(self._names):
                rect = QtCore.QRectF(i * (w + gap), 0, w, self.height())
                if i == self._current:
                    paint_surface(p, rect, m.raised, radius=rect.height() / 2)
                    colour = qcolor(t.text_primary)
                else:
                    paint_surface(p, rect, m.well, radius=rect.height() / 2)
                    colour = qcolor(t.text_secondary if i < self._current else t.text_muted)
                label = fm.elidedText(name, QtCore.Qt.TextElideMode.ElideRight, int(rect.width() - 14 - (16 if i < self._current else 0)))
                tw = fm.horizontalAdvance(label)
                x = rect.center().x() - (tw + (16 if i < self._current else 0)) / 2
                if i < self._current:
                    paint_glyph(p, QtCore.QRectF(x, rect.center().y() - 6, 12, 12), "check",
                                qcolor(t.accent), weight=2.4)
                    x += 16
                p.setPen(colour)
                p.drawText(QtCore.QRectF(x, rect.top(), tw + 2, rect.height()),
                           int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter), label)
        finally:
            p.end()


class ClusteringWizardDialog(QtWidgets.QDialog):
    """Multi-step wizard for clustering configuration.

    Steps:
    1. Feature Selection (which features to use)
    2. Normalization (scaling)
    3. Algorithm Selection (K-Means vs HDBSCAN)
    4. Post-Processing (not implemented by the engine yet; shown disabled)
    5. Output (what a run produces; "Offer to save a playlist per cluster"
       sets the run-config key ``create_playlists`` — in-memory only, never
       saved to ~/.soundvault_config.json)

    Every page's controls are read when the user moves *off* the page and
    again in :meth:`get_config`, so no choice is lost whichever way the user
    navigates. :meth:`set_config` pre-fills every page (the Analyse page
    opens the wizard on its current settings).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Clustering Configuration Wizard")
        self.setMinimumWidth(620)
        self.setMinimumHeight(480)

        # Configuration storage
        self.config = {
            "features": {
                "tempo": True,
                "mfcc": True,
                "chroma": True,
                "spectral": True,
                "energy": True,
                "onset_rate": False,
            },
            "normalization": "standard",
            "dimensionality_reduction": "none",
            "algorithm": "kmeans",
            "k": 8,
            "min_cluster_size": 5,
            "min_samples": 5,
            # Not implemented by the engine; recorded as what really happens.
            "remove_small_clusters": False,
            "min_cluster_threshold": 5,
            "merge_into_misc": False,
            # Live: offer the Save cluster playlists preview after a
            # successful run (read by ClusterPlaylistFlow.offer_after_run).
            "create_playlists": False,
            "generate_report": True,
            "open_graph": False,
        }

        self._current_step = 0
        self._build_ui()
        self._show_step(0)

    def _build_ui(self) -> None:
        """Build the wizard UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        # Title and step indicator (real state: the step being shown).
        title_layout = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("Step 1: Feature Selection")
        self._title.setObjectName("cardTitle")
        apply_font(self._title, "title")
        title_layout.addWidget(self._title, 1)
        self._step_indicator = QtWidgets.QLabel("1 of 5")
        self._step_indicator.setObjectName("mutedLabel")
        apply_font(self._step_indicator, "caption")
        title_layout.addWidget(self._step_indicator)
        layout.addLayout(title_layout)

        self._steps = _StepPills(STEP_NAMES)
        layout.addWidget(self._steps)

        # Stack of pages
        self._stack = QtWidgets.QStackedWidget()
        self._pages = [
            self._create_page_features(),
            self._create_page_normalization(),
            self._create_page_algorithm(),
            self._create_page_postprocessing(),
            self._create_page_output(),
        ]

        for page in self._pages:
            self._stack.addWidget(page)

        layout.addWidget(self._stack, 1)

        # Buttons: Cancel on the left, Back / Next on the right.
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setFlat(True)
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)
        button_layout.addStretch()

        self._prev_btn = QtWidgets.QPushButton("Back")
        self._prev_btn.clicked.connect(self._on_prev)
        button_layout.addWidget(self._prev_btn)

        self._next_btn = QtWidgets.QPushButton("Next")
        self._next_btn.setObjectName("primaryBtn")
        self._next_btn.clicked.connect(self._on_next)
        button_layout.addWidget(self._next_btn)

        layout.addLayout(button_layout)

    @staticmethod
    def _hint(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("notesHint")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _intro(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("sectionSubtitle")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _caption(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("mutedLabel")
        apply_font(label, "caption", weight=600)
        return label

    def _page(self) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        """A page: a glass card holding the step's controls."""
        from gui.widgets.glass import GlassCard

        page = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        card = GlassCard()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        outer.addWidget(card)
        outer.addStretch(1)
        return page, layout

    def _create_page_features(self) -> QtWidgets.QWidget:
        """Create feature selection page."""
        page, layout = self._page()

        layout.addWidget(self._intro(
            "Choose what the analysis listens for. More features give finer groups "
            "but take longer to compute."
        ))

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self._feature_checkboxes = feature_toggles(grid)
        for feat, toggle in self._feature_checkboxes.items():
            toggle.setChecked(self.config["features"][feat])
        layout.addLayout(grid)

        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.setSpacing(6)
        preset_layout.addWidget(self._caption("Presets"))
        for preset_name, preset_config in FEATURE_PRESETS:
            btn = QtWidgets.QPushButton(preset_name)
            btn.setFlat(True)
            btn.setToolTip("Tick " + features_text(preset_config))
            btn.clicked.connect(lambda checked, cfg=preset_config: self._apply_feature_preset(cfg))
            preset_layout.addWidget(btn)

        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        layout.addStretch()
        return page

    def _create_page_normalization(self) -> QtWidgets.QWidget:
        """Create normalization page."""
        page, layout = self._page()

        layout.addWidget(self._intro(
            "Choose how the features are scaled, so each one counts equally."
        ))

        layout.addWidget(self._caption("Scaling"))
        self._norm_combo = QtWidgets.QComboBox()
        for display_text, value in [
            ("Standard (Z-score)", "standard"),
            ("MinMax (0-1)", "minmax"),
            ("Robust (outlier-resistant)", "robust"),
        ]:
            self._norm_combo.addItem(display_text, userData=value)
        self._norm_combo.setCurrentIndex(0)
        layout.addWidget(self._norm_combo)

        layout.addWidget(self._hint(
            "• Standard: mean 0, standard deviation 1 (the default, good for most libraries)\n"
            "• MinMax: values between 0 and 1 (good for bounded features)\n"
            "• Robust: less affected by outliers (good for skewed data)"
        ))

        # Dimensionality reduction before clustering is not implemented: the
        # engine always clusters on every selected feature. (The 2-D/3-D maps
        # use UMAP or t-SNE automatically, for display only.)
        layout.addSpacing(6)
        layout.addWidget(self._caption("Dimensionality reduction"))
        self._reduction_combo = QtWidgets.QComboBox()
        self._reduction_combo.addItems([
            "None (use all features)",
            "PCA (fast, linear)",
            "t-SNE (better visualization, slow)",
            "UMAP (balanced, if installed)",
        ])
        self._reduction_combo.setCurrentIndex(0)
        self._reduction_combo.setEnabled(False)
        self._reduction_combo.setToolTip(_NOT_AVAILABLE)
        layout.addWidget(self._reduction_combo)
        layout.addWidget(self._hint(
            "Clustering always uses all selected features. The map views reduce "
            "them to 2-D/3-D automatically (UMAP if installed, otherwise t-SNE), "
            "for display only."
        ))

        layout.addStretch()
        return page

    def _create_page_algorithm(self) -> QtWidgets.QWidget:
        """Create algorithm selection page."""
        page, layout = self._page()

        layout.addWidget(self._intro(
            "K-Means makes the number of groups you ask for. HDBSCAN finds the "
            "groups itself and leaves outliers unclustered (needs the optional "
            "'hdbscan' package)."
        ))

        layout.addWidget(self._caption("Algorithm"))
        self._algo_combo = QtWidgets.QComboBox()
        # userData is the engine's method name; the text is only for display
        # ("K-Means".lower() is "k-means", which the engine did not accept).
        self._algo_combo.addItem("K-Means", userData="kmeans")
        self._algo_combo.addItem("HDBSCAN", userData="hdbscan")
        self._algo_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self._algo_combo)

        # K-Means parameters
        self._kmeans_widget = QtWidgets.QWidget()
        kmeans_layout = QtWidgets.QFormLayout(self._kmeans_widget)
        kmeans_layout.setContentsMargins(0, 6, 0, 0)

        self._k_spinbox = QtWidgets.QSpinBox()
        self._k_spinbox.setRange(2, 100)
        self._k_spinbox.setValue(self.config["k"])
        self._k_spinbox.setToolTip("Number of clusters")
        kmeans_layout.addRow("Clusters (K):", self._k_spinbox)

        self._k_info = self._hint(
            "Start with 8–12 clusters and adjust from the results: more clusters "
            "make more specific, smaller groups."
        )
        kmeans_layout.addRow("", self._k_info)

        layout.addWidget(self._kmeans_widget)

        # HDBSCAN parameters
        self._hdbscan_widget = QtWidgets.QWidget()
        hdbscan_layout = QtWidgets.QFormLayout(self._hdbscan_widget)
        hdbscan_layout.setContentsMargins(0, 6, 0, 0)

        self._min_size_spinbox = QtWidgets.QSpinBox()
        self._min_size_spinbox.setRange(2, 100)
        self._min_size_spinbox.setValue(self.config["min_cluster_size"])
        self._min_size_spinbox.setToolTip("Minimum tracks per cluster")
        hdbscan_layout.addRow("Min cluster size:", self._min_size_spinbox)

        self._min_samples_spinbox = QtWidgets.QSpinBox()
        self._min_samples_spinbox.setRange(1, 50)
        self._min_samples_spinbox.setValue(self.config["min_samples"])
        self._min_samples_spinbox.setToolTip("Minimum samples in neighborhood")
        hdbscan_layout.addRow("Min samples:", self._min_samples_spinbox)

        self._hdbscan_info = self._hint(
            "Raise the minimum cluster size to leave fewer tracks unclustered; "
            "lower values make more, smaller clusters."
        )
        hdbscan_layout.addRow("", self._hdbscan_info)

        self._hdbscan_widget.setVisible(False)
        layout.addWidget(self._hdbscan_widget)

        layout.addStretch()
        return page

    def _create_page_postprocessing(self) -> QtWidgets.QWidget:
        """Create post-processing options page (not implemented by the engine)."""
        page, layout = self._page()

        layout.addWidget(self._intro(
            "Post-processing is not available yet: every cluster the algorithm finds "
            "is kept as it is. With HDBSCAN, tracks that fit no cluster are shown as "
            "“Unclustered”."
        ))

        self._remove_small_cb = QtWidgets.QCheckBox("Remove clusters smaller than threshold")
        self._remove_small_cb.setChecked(self.config["remove_small_clusters"])
        layout.addWidget(self._remove_small_cb)

        self._min_threshold_spin = QtWidgets.QSpinBox()
        self._min_threshold_spin.setRange(1, 100)
        self._min_threshold_spin.setValue(self.config["min_cluster_threshold"])
        layout.addWidget(self._caption("Minimum cluster size (tracks)"))
        layout.addWidget(self._min_threshold_spin)

        self._merge_cb = QtWidgets.QCheckBox("Merge small clusters into 'Miscellaneous'")
        self._merge_cb.setChecked(self.config["merge_into_misc"])
        layout.addWidget(self._merge_cb)

        for w in (self._remove_small_cb, self._min_threshold_spin, self._merge_cb):
            w.setEnabled(False)
            w.setToolTip(_NOT_AVAILABLE)

        layout.addStretch()
        return page

    def _create_page_output(self) -> QtWidgets.QWidget:
        """Create output page: what a run actually produces."""
        page, layout = self._page()

        layout.addWidget(self._intro(
            "A run saves the cluster data for the Map, the 3-D view and Save cluster "
            "playlists (in the library's Docs/ folder). It changes nothing in your "
            "library; playlists are only written when you click Save in the preview."
        ))

        self._create_playlists_cb = QtWidgets.QCheckBox(
            "Offer to save a playlist per cluster when the analysis finishes"
        )
        self._create_playlists_cb.setChecked(bool(self.config.get("create_playlists")))
        self._create_playlists_cb.setToolTip(
            "Opens the Save cluster playlists preview after a successful run. "
            "Nothing is written until you click Save."
        )
        self._generate_report_cb = QtWidgets.QCheckBox("Generate cluster quality report")
        self._generate_report_cb.setChecked(True)
        self._open_graph_cb = QtWidgets.QCheckBox("Open interactive graph after clustering")
        self._open_graph_cb.setChecked(False)
        layout.addWidget(self._create_playlists_cb)
        for cb in (self._generate_report_cb, self._open_graph_cb):
            cb.setEnabled(False)
            layout.addWidget(cb)
        self._generate_report_cb.setToolTip("Always offered when the scores can be computed.")
        self._open_graph_cb.setToolTip("Use “Open the map” in the results.")

        layout.addWidget(self._hint(
            "• Playlists: “Save cluster playlists…” on the Sound Map previews one "
            "playlist per cluster (or export a selection from the Map)\n"
            "• Report: “Quality report” on the Sound Map\n"
            "• Map: “Open the map” in the results"
        ))

        layout.addStretch()
        return page

    def _on_algorithm_changed(self, _index: int = 0) -> None:
        """Show the parameters of the selected algorithm."""
        is_kmeans = self._algo_combo.currentData() == "kmeans"
        self._kmeans_widget.setVisible(is_kmeans)
        self._hdbscan_widget.setVisible(not is_kmeans)

    def _apply_feature_preset(self, preset_config: dict) -> None:
        """Apply a feature preset."""
        for feat, cb in self._feature_checkboxes.items():
            cb.setChecked(preset_config.get(feat, False))

    def _show_step(self, step: int) -> None:
        """Show a specific step (callers save the page being left first)."""
        self._current_step = step
        self._stack.setCurrentIndex(step)
        self._steps.set_current(step)
        self._step_indicator.setText(f"{step + 1} of 5")

        titles = [
            "Step 1: Feature Selection",
            "Step 2: Normalization",
            "Step 3: Algorithm Selection",
            "Step 4: Post-Processing",
            "Step 5: Output",
        ]
        self._title.setText(titles[step])

        self._prev_btn.setEnabled(step > 0)
        self._next_btn.setText("Finish" if step == 4 else "Next")

    def _validate_step(self, step: int) -> bool:
        """Return False (after telling the user) if ``step`` cannot be left."""
        if step == 0 and not any(cb.isChecked() for cb in self._feature_checkboxes.values()):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Selection",
                "Select at least one audio feature to cluster on.",
            )
            return False
        return True

    def _save_step(self, step: int) -> None:
        """Copy page ``step``'s widgets into :attr:`config`."""
        if step == 0:  # Features
            for feat, cb in self._feature_checkboxes.items():
                self.config["features"][feat] = cb.isChecked()
        elif step == 1:  # Normalization
            norm_value = self._norm_combo.currentData()
            self.config["normalization"] = norm_value if norm_value else "standard"
            self.config["dimensionality_reduction"] = "none"
        elif step == 2:  # Algorithm
            self.config["algorithm"] = self._algo_combo.currentData() or "kmeans"
            self.config["k"] = self._k_spinbox.value()
            self.config["min_cluster_size"] = self._min_size_spinbox.value()
            self.config["min_samples"] = self._min_samples_spinbox.value()
        elif step == 3:  # Post-processing (disabled controls)
            self.config["remove_small_clusters"] = self._remove_small_cb.isChecked()
            self.config["min_cluster_threshold"] = self._min_threshold_spin.value()
            self.config["merge_into_misc"] = self._merge_cb.isChecked()
        elif step == 4:  # Output (only "offer to save playlists" is live)
            self.config["create_playlists"] = self._create_playlists_cb.isChecked()
            self.config["generate_report"] = self._generate_report_cb.isChecked()
            self.config["open_graph"] = self._open_graph_cb.isChecked()

    def _save_current_step(self) -> None:
        """Backwards-compatible alias: save the page currently shown."""
        self._save_step(self._current_step)

    def _save_all(self) -> None:
        for step in range(len(self._pages)):
            self._save_step(step)

    def set_config(self, config: dict) -> None:
        """Pre-fill every page from ``config`` (the keys a run config has;
        missing ones keep the wizard's defaults)."""
        features = config.get("features") or {}
        for feat, cb in self._feature_checkboxes.items():
            if feat in features:
                cb.setChecked(bool(features[feat]))
        index = self._norm_combo.findData(config.get("normalization", "standard"))
        if index >= 0:
            self._norm_combo.setCurrentIndex(index)
        index = self._algo_combo.findData(config.get("algorithm", "kmeans"))
        if index >= 0:
            self._algo_combo.setCurrentIndex(index)
        self._k_spinbox.setValue(int(config.get("k", self._k_spinbox.value())))
        self._min_size_spinbox.setValue(int(config.get("min_cluster_size", self._min_size_spinbox.value())))
        self._min_samples_spinbox.setValue(int(config.get("min_samples", self._min_samples_spinbox.value())))
        self._create_playlists_cb.setChecked(bool(config.get("create_playlists", False)))
        self._save_all()

    def _on_prev(self) -> None:
        """Go to previous step (keeping what was entered on this one)."""
        if self._current_step > 0:
            self._save_step(self._current_step)
            self._show_step(self._current_step - 1)

    def _on_next(self) -> None:
        """Go to next step or accept."""
        if not self._validate_step(self._current_step):
            return
        self._save_step(self._current_step)
        if self._current_step < 4:
            self._show_step(self._current_step + 1)
        else:
            if not self._validate_step(0):
                self._show_step(0)
                return
            self._save_all()
            self.accept()

    def get_config(self) -> Dict:
        """Get the final configuration (read from every page's widgets)."""
        self._save_all()
        return self.config
