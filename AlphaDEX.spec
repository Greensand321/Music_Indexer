# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for AlphaDEX (the PySide6/Qt app — alpha_dex_gui.py).

Build (one-folder):
    pip install pyinstaller pyinstaller-hooks-contrib
    pyinstaller AlphaDEX.spec

Output lands in dist/AlphaDEX/ — AlphaDEX.exe plus its supporting files.
That folder is what an Inno Setup script packages into an installer.

Notes for whoever runs the first real build:

- This spec was authored and structurally validated (import-graph tracing,
  py_compile, syntax) in a sandbox that does not have PySide6/numpy/scipy/
  librosa/scikit-learn/hdbscan installed, so it has NOT been run through an
  actual `pyinstaller` invocation yet. Expect 1-2 rounds of adding missing
  hiddenimports/datas the first time this runs for real — that is normal for
  a scientific-Python stack this size, not a sign the spec is wrong.
- gui/compat.py unconditionally imports PySide6.QtWebEngineWidgets and
  PySide6.QtWebChannel as part of selecting the Qt binding (see _load()) —
  do NOT add those to excludes, even though nothing in the app currently
  *uses* a QtWebEngine view (the 3-D graph opens HTML in the system browser
  instead). Excluding them will crash the app at the first `gui.compat`
  import. pyinstaller-hooks-contrib ships the PySide6 hook that pulls in
  QtWebEngine's Chromium subprocess/locale files; this is the single
  heaviest part of the build (expect several hundred MB) and there is no
  way around it while compat.py imports it unconditionally.
"""

datas = [
    ("gui/fonts/*.ttf", "gui/fonts"),
    ("docs", "docs"),
]

# tag_fixer.py discovers metadata-service plugins by importing these modules
# by name at runtime (importlib.import_module(f'plugins.{name}')) rather than
# via plain `import` statements, so PyInstaller's static analysis can't be
# relied on to find them on its own.
hiddenimports = [
    "plugins.acoustid_plugin",
    "plugins.api_service",
    "plugins.lastfm",
    "plugins.test_plugin",
    # library_sync_indexer_engine/indexer_engine/fingerprint_generator.py
    # deliberately reaches across to this shared root-level module.
    "audio_norm",
]

# Confirmed unreachable from alpha_dex_gui.py by static import-graph tracing:
# the legacy Tkinter app and everything that hangs off it. These are excluded
# defensively — Analysis() already won't pull them in on its own — mainly to
# fail loudly if something starts importing them instead of silently bloating
# the build.
excludes = [
    "main_gui",
    "library_sync_review",
    "cluster_graph_panel",
    "mutagen_stub",
    "ttkbootstrap",
    "ttkthemes",
    "sv_ttk",
    "PyQt6",  # gui/compat.py only falls back to this if PySide6 is absent
]

a = Analysis(
    ["alpha_dex_gui.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AlphaDEX",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: point at an .ico once one exists for the app
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AlphaDEX",
)
