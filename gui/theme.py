"""Centralised colour palette and QSS stylesheet for AlphaDEX Qt GUI."""
from __future__ import annotations

# ── Palette ───────────────────────────────────────────────────────────────────
SIDEBAR_BG      = "#0f172a"
SIDEBAR_HOVER   = "#1e293b"
SIDEBAR_ACTIVE  = "#1d4ed8"
SIDEBAR_TEXT    = "#cbd5e1"
SIDEBAR_HDR     = "#64748b"

CONTENT_BG      = "#f1f5f9"
CARD_BG         = "#ffffff"
CARD_BORDER     = "#e2e8f0"

TEXT_PRIMARY    = "#0f172a"
TEXT_SECONDARY  = "#64748b"
TEXT_MUTED      = "#94a3b8"

ACCENT          = "#3b82f6"
ACCENT_DARK     = "#1d4ed8"
SUCCESS         = "#22c55e"
WARNING         = "#f59e0b"
DANGER          = "#ef4444"

PROGRESS_BG     = "#e2e8f0"
PROGRESS_FG     = ACCENT

LOG_BG          = "#0f172a"
LOG_TEXT        = "#94a3b8"
LOG_TEXT_OK     = "#4ade80"
LOG_TEXT_WARN   = "#fbbf24"
LOG_TEXT_ERR    = "#f87171"


# ── Stylesheet ─────────────────────────────────────────────────────────────────
def build_stylesheet() -> str:
    return f"""
/* ── Global ─────────────────────────────────────────────────────────── */
QWidget {{
    font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: {TEXT_PRIMARY};
}}

QMainWindow {{
    background: {CONTENT_BG};
}}

/* ── Scroll bars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {CONTENT_BG};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {CARD_BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {CONTENT_BG};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {CARD_BORDER};
    border-radius: 4px;
    min-width: 20px;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
#sidebar {{
    background: {SIDEBAR_BG};
}}
#sidebarSectionLabel {{
    color: {SIDEBAR_HDR};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 8px 2px 8px;
}}
#navButton {{
    background: transparent;
    border: none;
    color: {SIDEBAR_TEXT};
    text-align: left;
    padding: 9px 12px 9px 16px;
    border-radius: 8px;
    font-size: 13px;
}}
#navButton:hover {{
    background: {SIDEBAR_HOVER};
    color: #f8fafc;
}}
#navButton[active="true"] {{
    background: {SIDEBAR_ACTIVE};
    color: #ffffff;
    font-weight: 600;
}}

/* ── Top Bar ─────────────────────────────────────────────────────────── */
#topBar {{
    background: {CARD_BG};
    border-bottom: 1px solid {CARD_BORDER};
}}
#appTitle {{
    font-size: 17px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.02em;
}}
#libraryPath {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
#libStats {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}

/* ── Cards / Workspace panels ────────────────────────────────────────── */
#workspaceCard {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
}}
#sectionTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
#sectionSubtitle {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    color: {TEXT_PRIMARY};
    padding: 7px 16px;
    border-radius: 8px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {CONTENT_BG};
    border-color: {TEXT_MUTED};
}}
QPushButton:pressed {{
    background: {CARD_BORDER};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    border-color: {PROGRESS_BG};
}}
QPushButton#primaryBtn {{
    background: {ACCENT};
    border: none;
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#primaryBtn:hover {{
    background: {ACCENT_DARK};
}}
QPushButton#primaryBtn:disabled {{
    background: {TEXT_MUTED};
}}
QPushButton#dangerBtn {{
    background: {DANGER};
    border: none;
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#successBtn {{
    background: {SUCCESS};
    border: none;
    color: #ffffff;
    font-weight: 600;
}}

/* ── Inputs ──────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QComboBox {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 6px;
    padding: 5px 8px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

/* ── Progress bar ────────────────────────────────────────────────────── */
QProgressBar {{
    background: {PROGRESS_BG};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    font-size: 11px;
    color: {TEXT_SECONDARY};
}}
QProgressBar::chunk {{
    background: {PROGRESS_FG};
    border-radius: 4px;
}}

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}

/* ── Tables / Treeviews ──────────────────────────────────────────────── */
QTreeWidget, QTableWidget, QListWidget {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    alternate-background-color: {CONTENT_BG};
    gridline-color: {CARD_BORDER};
}}
QHeaderView::section {{
    background: {CONTENT_BG};
    border: none;
    border-bottom: 1px solid {CARD_BORDER};
    padding: 5px 8px;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
QTreeWidget::item:selected, QTableWidget::item:selected,
QListWidget::item:selected {{
    background: {ACCENT};
    color: #ffffff;
}}

/* ── Tabs ────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {CARD_BORDER};
    border-radius: 0 8px 8px 8px;
    background: {CARD_BG};
}}
QTabBar::tab {{
    background: {CONTENT_BG};
    border: 1px solid {CARD_BORDER};
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 6px 14px;
    margin-right: 2px;
    color: {TEXT_SECONDARY};
}}
QTabBar::tab:selected {{
    background: {CARD_BG};
    color: {ACCENT_DARK};
    font-weight: 600;
}}
QTabBar::tab:hover {{
    background: {CARD_BG};
}}

/* ── Log drawer ──────────────────────────────────────────────────────── */
#logDrawer {{
    background: {LOG_BG};
    border-top: 1px solid #1e293b;
}}
#logText {{
    background: {LOG_BG};
    color: {LOG_TEXT};
    border: none;
    font-family: "Consolas", "JetBrains Mono", "Menlo", monospace;
    font-size: 12px;
}}

/* ── Status bar ──────────────────────────────────────────────────────── */
QStatusBar {{
    background: {CARD_BG};
    border-top: 1px solid {CARD_BORDER};
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}

/* ── Splitter ────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {CARD_BORDER};
    width: 1px;
}}

/* ── Checkboxes + Radio buttons ──────────────────────────────────────── */
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid {CARD_BORDER};
    border-radius: 4px;
    background: {CARD_BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QRadioButton::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ── Tooltips ────────────────────────────────────────────────────────── */
QToolTip {{
    background: {TEXT_PRIMARY};
    color: #f8fafc;
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
}}

/* ── Dialogs ─────────────────────────────────────────────────────────── */
QDialog {{
    background: {CONTENT_BG};
}}
"""
