"""The Qt app's information architecture: sidebar groups → pages → sections.

This module is the single source for what the sidebar shows, the order of
Ctrl+1..9, the status-bar labels, and how a navigation key is routed. It is
deliberately Qt-free so it can be read (and tested) without a display.

Vocabulary
----------
* **Page** — one sidebar entry (``home``, ``duplicates``, ``sound_map``, …).
* **Section** — a view inside a page, switched with the page's section tabs.
  A page with one section shows no tabs.
* **Panel** — the widget that provides a section: one of the workspace
  classes in ``gui.main_window._WORKSPACE_MAP`` (``similarity``, ``graph``,
  ``tag_fixer``, …). A panel can serve several sections of one page (the
  Genre panel serves "Fill genres" and "Normalize genres"); it is then told
  which view to show through ``set_section(view)``.

Navigation keys
---------------
``resolve(key)`` accepts, in this order:

1. ``"page/section"`` — an explicit page and section (``"settings/appearance"``);
2. a panel key — the old per-panel keys keep working (``"similarity"``,
   ``"graph"``, ``"clustered"``, ``"compression"``, ``"genres"``, …), so
   ``navigate_requested("graph")`` still lands on the map. A key that names
   both a page and a panel (``"duplicates"``) is the panel: it shows the
   Duplicate Finder, i.e. ``duplicates/find``;
3. a section key — section keys are unique across the app (``"compare"``);
4. a page key — that page, in whichever section it last showed;
5. a retired key from ``LEGACY_ALIASES`` (``"tools"``, the old Utilities page).

The sidebar and Ctrl+1..9 select *pages* (``AlphaDEXWindow.select_page``):
a page keeps showing the section it showed last.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    key: str                 # unique across the app, e.g. "compare"
    label: str               # tab text, e.g. "Compare two files"
    panel: str               # key in gui.main_window._WORKSPACE_MAP
    view: str | None = None  # passed to panel.set_section() when it serves several sections
    experimental: bool = False   # a prototype: shown with an "Experimental" pill


@dataclass(frozen=True)
class Page:
    key: str
    label: str
    icon: str
    group: str               # sidebar group title; "" = the untitled bottom group
    intent: str              # one line: what the page is for (sidebar tooltip)
    sections: tuple[Section, ...] = field(default_factory=tuple)

    def section(self, key: str) -> Section | None:
        return next((s for s in self.sections if s.key == key), None)

    @property
    def has_sections(self) -> bool:
        """True when the page shows section tabs (more than one section)."""
        return len(self.sections) > 1


def _single(key: str, label: str, icon: str, group: str, intent: str, panel: str = "") -> Page:
    return Page(key, label, icon, group, intent, (Section(key, label, panel or key),))


# Sidebar order. Ctrl+1..9 follow it; so does the snapshot script.
PAGES: tuple[Page, ...] = (
    _single("home", "Home", "⌂", "LIBRARY",
            "Your library at a glance: health, what needs attention, and the latest runs"),

    _single("indexer", "Indexer", "🗂", "ORGANIZE",
            "Normalize folder structure and file names, with a preview before anything moves"),
    _single("library_sync", "Library Sync", "🔄", "ORGANIZE",
            "Merge an incoming folder: add new tracks, skip duplicates, accept upgrades"),
    _single("file_cleanup", "File Cleanup", "🧹", "ORGANIZE",
            "Remove ' (1)' / ' - Copy' suffixes, previewed and undoable"),

    Page("duplicates", "Duplicates", "🔍", "CLEAN UP",
         "Find same-recording files, pick the keepers, quarantine the rest", (
             Section("find", "Find duplicates", "duplicates"),
             Section("compare", "Compare two files", "similarity"),
             Section("pair_review", "Pair review", "pair_review"),
             Section("fuzzy_finder", "Fuzzy finder", "fuzzy_finder", experimental=True),
             Section("scan_engine", "Scan engine", "scan_engine", experimental=True),
             Section("bucketing", "Bucketing", "bucketing", experimental=True),
         )),
    Page("metadata", "Metadata", "🏷", "CLEAN UP",
         "Fix wrong or missing tags and genres, reviewing every change before it is written", (
             Section("tag_fixer", "Tag Fixer", "tag_fixer"),
             Section("fill_genres", "Fill genres", "genres", view="fill"),
             Section("normalize_genres", "Normalize genres", "genres", view="normalize"),
             Section("undo_tags", "Undo changes", "tag_history"),
         )),

    _single("player", "Player", "▶", "LISTEN",
            "Browse the library, play tracks and build a queue"),
    Page("sound_map", "Sound Map", "🌐", "LISTEN",
         "Analyse how your tracks sound, explore them as a map, then play or save a selection", (
             Section("analyse", "Analyse", "clustered"),
             Section("map", "Map", "graph"),
         )),
    _single("playlists", "Playlists", "🎵", "LISTEN",
            "Rule-based playlists (folders, tempo/energy, Auto-DJ) and playlist repair"),

    Page("mirror", "Mirror", "📦", "TAKE AWAY",
         "A portable Opus copy of the library; the originals are never touched", (
             Section("build", "Build the mirror", "compression"),
             Section("verify", "Verify the mirror", "mirror_verify"),
         )),
    _single("exports", "Exports", "📤", "TAKE AWAY",
            "Artist/title and codec lists of the library, written to Docs/"),

    _single("activity", "Activity", "⏱", "",
            "Running jobs (with Cancel), the full log history and the latest problems"),
    Page("settings", "Settings", "⚙", "",
         "Metadata services, the default library, thresholds and appearance", (
             Section("services", "Metadata services", "settings", view="services"),
             Section("general", "General", "settings", view="general"),
             Section("appearance", "Appearance", "settings", view="appearance"),
         )),
    _single("help", "Help", "?", "",
            "Documentation, keyboard shortcuts and how to report a problem"),
)

PAGE_BY_KEY: dict[str, Page] = {p.key: p for p in PAGES}

# Retired keys that no single panel stands for any more → where they lead now.
# ("tools" was the Utilities grab-bag: its exports moved to Exports, File
# Cleanup got its own page, the Validator became Home's "Library health", the
# diagnostics went to Duplicates / Mirror, and the crash log to Activity.)
LEGACY_ALIASES: dict[str, str] = {
    "tools": "exports",
}


def nav_order() -> list[str]:
    """Page keys in sidebar order (Ctrl+1..9 use the first nine)."""
    return [p.key for p in PAGES]


def nav_groups() -> list[tuple[str, list[Page]]]:
    """``[(group title, [pages])]`` in sidebar order."""
    groups: list[tuple[str, list[Page]]] = []
    for page in PAGES:
        if groups and groups[-1][0] == page.group:
            groups[-1][1].append(page)
        else:
            groups.append((page.group, [page]))
    return groups


def panel_keys() -> list[str]:
    """Every panel key the pages use, in sidebar order, without repeats."""
    seen: list[str] = []
    for page in PAGES:
        for section in page.sections:
            if section.panel not in seen:
                seen.append(section.panel)
    return seen


def sections_of_panel(panel: str) -> list[tuple[Page, Section]]:
    return [(p, s) for p in PAGES for s in p.sections if s.panel == panel]


def label(page_key: str, section_key: str | None = None) -> str:
    """Status-bar text: "Page", or "Page · Section" on a page with tabs."""
    page = PAGE_BY_KEY.get(page_key)
    if page is None:
        return page_key.replace("_", " ").title()
    section = page.section(section_key) if section_key else None
    if section is not None and page.has_sections:
        return f"{page.label} · {section.label}"
    return page.label


def resolve(key: str, *, _depth: int = 0) -> tuple[str, str | None] | None:
    """``(page key, section key or None)`` for a navigation key, or None.

    ``None`` as the section means "the page, in whichever section it shows".
    For a panel that serves several sections the first one is returned; the
    main window refines that with the panel's current view.
    """
    if not key or _depth > 2:
        return None
    if "/" in key:
        page_key, _, section_key = key.partition("/")
        page = PAGE_BY_KEY.get(page_key)
        if page is None or page.section(section_key) is None:
            return None
        return page_key, section_key
    found = sections_of_panel(key)
    if found:
        page, section = found[0]
        return page.key, section.key
    for page in PAGES:
        if page.section(key) is not None:
            return page.key, key
    if key in PAGE_BY_KEY:
        return key, None
    if key in LEGACY_ALIASES:
        return resolve(LEGACY_ALIASES[key], _depth=_depth + 1)
    return None
