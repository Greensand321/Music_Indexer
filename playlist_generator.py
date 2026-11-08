"""Playlist generation utilities with edge-case handling.

DEFAULT_EXTS defines which audio file extensions are included when building
playlists. Existing playlists are overwritten by default; pass
``overwrite=False`` to skip files that already exist.

Preview-first API (used by the Qt Playlists workspace): the ``plan_*``
functions compute exactly what would be written (one :class:`PlannedPlaylist`
per file, with its final lines) without touching the disk, and
:func:`write_playlist_plan` writes exactly those lines, backing up any file it
overwrites first.
"""

import codecs, os, hashlib, difflib, re, shutil, time
from collections import defaultdict
from dataclasses import dataclass, field
from crash_watcher import record_event
from crash_logger import watcher
from utils.path_helpers import playlist_entry, safe_relpath

# Default extensions for playlist generation
DEFAULT_EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".opus"}
SKIP_SCAN_DIRS = {"trash", "docs", "not sorted", "playlists", "quarantine"}
#: Where :func:`update_playlists` keeps a copy of every playlist it rewrites
#: (under the library's ``Docs/``).
PLAYLIST_BACKUP_DIR = "playlist_backups"


_COPY_SUFFIX_RE = re.compile(r"(?:\s*-\s*copy|\s+copy)(?:\s*\d+)?$", re.IGNORECASE)
_NUMBER_SUFFIX_RE = re.compile(r"\s*[\(\[\{]\s*\d+\s*[\)\]\}]\s*$")


def _normalize_track_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0].lower()
    base = base.replace("_", " ").replace(".", " ")
    base = _COPY_SUFFIX_RE.sub("", base)
    base = _NUMBER_SUFFIX_RE.sub("", base)
    return re.sub(r"\s+", " ", base).strip()


def _tokenize_name(name: str) -> list[str]:
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", name) if tok]
    return [tok for tok in tokens if not tok.isdigit()]


def _build_audio_index(root: str, valid_exts: set[str]) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    name_index: dict[str, list[str]] = defaultdict(list)
    token_index: dict[str, set[str]] = defaultdict(set)
    for dirpath, dirnames, filenames in os.walk(root):
        rel_parts = set(os.path.relpath(dirpath, root).split(os.sep))
        if rel_parts & SKIP_SCAN_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_SCAN_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue
            full_path = os.path.join(dirpath, fname)
            norm_name = _normalize_track_name(full_path)
            if not norm_name:
                continue
            name_index[norm_name].append(full_path)
            for token in _tokenize_name(norm_name):
                token_index[token].add(full_path)
    return name_index, token_index


def _pick_best_match(
    missing_path: str,
    name_index: dict[str, list[str]],
    token_index: dict[str, set[str]],
) -> str | None:
    norm_missing = _normalize_track_name(missing_path)
    if not norm_missing:
        return None
    tokens = set(_tokenize_name(norm_missing))
    candidates = list(name_index.get(norm_missing, []))
    if not candidates and tokens:
        candidate_set: set[str] = set()
        for token in tokens:
            candidate_set.update(token_index.get(token, set()))
        candidates = list(candidate_set)
    if not candidates:
        return None

    orig_dir = os.path.dirname(missing_path)
    orig_ext = os.path.splitext(missing_path)[1].lower()

    def score(path: str) -> tuple[float, float]:
        cand_norm = _normalize_track_name(path)
        cand_tokens = set(_tokenize_name(cand_norm))
        overlap = len(tokens & cand_tokens) / len(tokens) if tokens else 0.0
        ratio = difflib.SequenceMatcher(None, norm_missing, cand_norm).ratio()
        bonus = 0.0
        if os.path.dirname(path) == orig_dir:
            bonus += 0.2
        if orig_ext and os.path.splitext(path)[1].lower() == orig_ext:
            bonus += 0.1
        return overlap + ratio + bonus, overlap

    best_path = None
    best_score = 0.0
    best_overlap = 0.0
    for path in candidates:
        total_score, overlap = score(path)
        if total_score > best_score:
            best_score = total_score
            best_overlap = overlap
            best_path = path

    if best_path is None:
        return None
    if best_overlap < 0.4 and best_score < 1.0:
        return None
    return best_path
DEFAULT_EXTS = {".mp3", ".flac", ".wav", ".aac", ".m4a", ".ogg", ".opus"}


def _sanitize_name(rel_path, existing):
    """Sanitize a relative path for playlist filename.

    Replaces path separators with underscores and appends a short hash if
    the sanitized name already exists in ``existing``.
    """
    # ``os.path.relpath(root, root)`` is ".", which used to give "..m3u" for
    # tracks sitting directly in the library folder.
    if rel_path in ("", os.curdir):
        rel_path = ""
    base = rel_path.replace(os.sep, "_") or "root"
    if base not in existing:
        return base
    h = hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:6]
    return f"{base}_{h}"


@watcher.traced
def generate_playlists(
    moves,
    root_path,
    output_dir=None,
    valid_exts=None,
    overwrite=True,
    log_callback=None,
    only_existing=False,
):
    """Generate M3U playlists based on move mapping.

    Parameters
    ----------
    moves : dict
        Mapping of original file paths to their new locations.
    root_path : str
        Root of the music library.
    output_dir : str or None
        Directory to write playlists into. Defaults to ``root_path/Playlists``.
    valid_exts : set or None
        Extensions to include. Defaults to ``DEFAULT_EXTS``.
    overwrite : bool
        Whether to overwrite existing playlists.
    log_callback : callable
        Function for logging messages.
    only_existing : bool
        Only touch folder playlists that already exist (never create new
        ones). Note that merging (``overwrite=False``) rewrites the playlist
        as a sorted set of lines, so it must never be used to patch a
        user's playlist: :func:`update_playlists` edits entries in place.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    valid_exts = {e.lower() for e in (valid_exts or DEFAULT_EXTS)}

    playlists_dir = output_dir or os.path.join(root_path, "Playlists")
    if only_existing and not os.path.isdir(playlists_dir):
        return
    os.makedirs(playlists_dir, exist_ok=True)
    record_event(f"playlist_generator: generating playlists in {playlists_dir}")

    dir_map = defaultdict(list)
    for old, new in moves.items():
        ext = os.path.splitext(new)[1].lower()
        if ext in valid_exts:
            dir_map[os.path.dirname(old)].append(new)

    used = set()
    for old_dir, files in dir_map.items():
        if not files:
            log_callback(f"\u26A0 Skipping empty folder: {old_dir}")
            continue

        rel = os.path.relpath(old_dir, root_path)
        name = _sanitize_name(rel, used)
        used.add(name)

        playlist_file = os.path.join(playlists_dir, f"{name}.m3u")
        if only_existing and not os.path.exists(playlist_file):
            continue

        rel_files = {os.path.relpath(p, playlists_dir) for p in files}

        if os.path.exists(playlist_file):
            if not overwrite:
                try:
                    text = _read_playlist_text(playlist_file).text
                except OSError as e:
                    # Merging into a playlist we could not read would replace
                    # it with only the new tracks: leave it alone instead.
                    log_callback(f"\u2717 Failed to read {playlist_file}; left unchanged: {e}")
                    continue
                existing = {line.strip() for line in text.splitlines() if line.strip()}
                rel_files.update(existing)
            log_callback(f"\u2192 Writing playlist: {playlist_file}")
        else:
            log_callback(f"\u2192 Writing playlist: {playlist_file}")

        try:
            with open(playlist_file, "w", encoding="utf-8") as f:
                for p in sorted(rel_files):
                    f.write(p + "\n")
        except Exception as e:
            log_callback(f"\u2717 Failed to write {playlist_file}: {e}")
    record_event("playlist_generator: playlist generation complete")


def write_playlist(tracks, outfile):
    """Write a simple M3U playlist from ``tracks`` to ``outfile``."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            for p in tracks:
                f.write(safe_relpath(p, os.path.dirname(outfile)) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to write playlist {outfile}: {e}")


def _find_library_root(paths) -> str | None:
    """Best guess at the library root for ``paths``: the nearest ancestor of
    their common path that has a ``Playlists`` folder.

    The common path alone is the album folder whenever every change is inside
    one album (the usual case for a rename), and ``<album>/Playlists`` never
    exists — which is why ``update_playlists`` used to do nothing there.
    """
    try:
        common = os.path.commonpath([os.path.abspath(p) for p in paths])
    except ValueError:  # different drives, or empty
        return None
    candidate = common if os.path.isdir(common) else os.path.dirname(common)
    while candidate:
        if os.path.isdir(os.path.join(candidate, "Playlists")):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return None
        candidate = parent
    return None


# ── Reading / writing existing playlists faithfully ──────────────────────

#: Encodings tried, in order, for a playlist that is not valid UTF-8. latin-1
#: decodes any byte string, so reading never fails on encoding alone; cp1252
#: comes first because that is what Windows tools write into ``.m3u`` files.
_FALLBACK_ENCODINGS = ("cp1252", "latin-1")


@dataclass
class PlaylistText:
    """A playlist file's text and the encoding it was read with.

    ``encoding`` is what :func:`_encode_playlist_text` writes back with, so an
    edit never silently changes a file's encoding (``utf-8-sig`` keeps a BOM).
    ``.m3u8`` files are UTF-8 by definition and are always written as UTF-8.
    """

    text: str
    encoding: str


def _read_playlist_text(path: str) -> PlaylistText:
    """Read ``path`` with a UTF-8 → cp1252 → latin-1 fallback.

    Raises ``OSError`` only when the file itself cannot be read.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data.startswith(codecs.BOM_UTF8):
        try:
            return PlaylistText(data[len(codecs.BOM_UTF8):].decode("utf-8"), "utf-8-sig")
        except UnicodeDecodeError:
            data = data[len(codecs.BOM_UTF8):]
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return PlaylistText(data.decode("utf-16"), "utf-16")
        except UnicodeDecodeError:
            pass
    try:
        return PlaylistText(data.decode("utf-8"), "utf-8")
    except UnicodeDecodeError:
        pass
    for enc in _FALLBACK_ENCODINGS:
        try:
            return PlaylistText(data.decode(enc), enc)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes every byte string")  # pragma: no cover


def _write_encoding(path: str, encoding: str) -> str:
    if path.lower().endswith(".m3u8") and encoding not in ("utf-8", "utf-8-sig"):
        return "utf-8"
    return encoding or "utf-8"


def _encode_playlist_text(path: str, text: str, encoding: str) -> bytes:
    """``text`` in the encoding ``path`` was read with.

    Raises ``UnicodeEncodeError`` when a new entry can't be represented in the
    file's own encoding (e.g. a Japanese file name in a cp1252 ``.m3u``): the
    caller must then leave the file alone rather than corrupt it.
    """
    return text.encode(_write_encoding(path, encoding))


def _append_write_log(log_dir, log_lines, log_callback) -> None:
    """Append ``log_lines`` to ``<log_dir>/playlist_writes.log``."""
    if not log_dir or not log_lines:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(log_dir, "playlist_writes.log"), "a", encoding="utf-8") as fh:
            for line in log_lines:
                fh.write(f"{stamp} {line}\n")
    except OSError as exc:
        log_callback(f"! Could not append to the playlist write log: {exc}")


def _split_line(raw: str) -> tuple:
    """``raw`` (one line, ending included) -> ``(content, line_ending)``."""
    content = raw.rstrip("\r\n")
    return content, raw[len(content):]


def _rewrite_playlist_lines(text: str, playlist_dir: str, normalized_map: dict) -> tuple:
    """Rewrite only the entries of ``text`` whose path is in ``normalized_map``.

    Everything else — order, comments, blank lines, line endings, ``#EXTINF``
    lines — is kept byte for byte. An entry mapped to ``None`` (deleted with
    no replacement) is removed together with the ``#EXTINF``/``#EXT…`` lines
    directly above it, so no orphaned ``#EXTINF`` is left pointing at the
    next track. Returns ``(new_text, n_changed_entries)``.
    """
    out: list = []
    # Index in ``out`` where the current entry's directive block starts
    # (its #EXTINF line), or None.
    block_start = None
    changed = 0
    for raw in text.splitlines(keepends=True):
        content, ending = _split_line(raw)
        entry = content.strip()
        if not entry:
            out.append(raw)
            block_start = None
            continue
        if entry.startswith("#"):
            if entry.upper().startswith("#EXTINF"):
                block_start = len(out)
            elif not entry.upper().startswith("#EXT") or entry.upper().startswith("#EXTM3U"):
                block_start = None  # a plain comment / the header is not part of the entry
            out.append(raw)
            continue
        is_abs = os.path.isabs(entry)
        abs_line = os.path.normcase(os.path.normpath(
            entry if is_abs else os.path.join(playlist_dir, entry)
        ))
        if abs_line not in normalized_map:
            out.append(raw)
            block_start = None
            continue
        new = normalized_map[abs_line]
        if new is None:
            if block_start is not None:
                del out[block_start:]
            changed += 1
            block_start = None
            continue
        # Absolute stays absolute, relative stays relative (absolute only on
        # another Windows drive), and the entry's own separator style is kept.
        new_entry = playlist_entry(new, playlist_dir, like=entry)
        if new_entry != entry:
            changed += 1
            lead = content[: len(content) - len(content.lstrip())]
            trail = content[len(content.rstrip()):]
            out.append(f"{lead}{new_entry}{trail}{ending}")
        else:
            out.append(raw)
        block_start = None
    return "".join(out), changed


@watcher.traced
def update_playlists(changes, library_root=None, log_callback=None):
    """Update the playlists in ``<library>/Playlists`` for moved or deleted tracks.

    ``changes`` is a mapping of ``old_path -> new_path``; a ``None`` value
    means the old path was removed without a replacement (its entry, and the
    ``#EXTINF`` line above it, are dropped). An iterable of paths is accepted
    for backward compatibility and treated as "nothing moved", so it changes
    nothing.

    Only entries whose path is a key of ``changes`` are rewritten. Order,
    comments, ``#EXTINF`` pairing, line endings and the file's encoding are
    kept. Missing entries that are *not* in ``changes`` are left alone — fuzzy
    repair belongs to the previewed Repair tab (:func:`plan_playlist_repair`),
    never to a side effect of a rename. No new playlist is ever created.

    Each playlist is copied to ``<library>/Docs/playlist_backups/`` before it
    is overwritten, and every backup and write is appended to
    ``<library>/Docs/playlist_writes.log``. A playlist that can't be read, or
    whose new entries can't be written in its own encoding, is left untouched
    (and reported through ``log_callback``).

    ``library_root`` is the library whose ``Playlists/`` folder is updated.
    Pass it whenever you know it. When omitted, the nearest ancestor of the
    changed paths that has a ``Playlists`` folder is used.

    Returns the number of playlist files rewritten (0 when nothing needed
    changing or no ``Playlists`` folder was found), so callers can report what
    really happened.
    """
    if log_callback is None:
        log_callback = lambda msg: None

    if not changes:
        return 0

    if isinstance(changes, dict):
        move_map = dict(changes)
    else:
        move_map = {p: p for p in changes}

    normalized_map = {
        os.path.normcase(os.path.normpath(os.path.abspath(old))): new
        for old, new in move_map.items()
        if old and new != old
    }

    all_paths = list(move_map.keys()) + [p for p in move_map.values() if p]
    root = library_root or (_find_library_root(all_paths) if all_paths else None)
    if not root:
        log_callback("• No Playlists folder found for the changed tracks; nothing to update")
        return 0

    playlists_dir = os.path.join(root, "Playlists")
    if not os.path.isdir(playlists_dir):
        log_callback(f"• {playlists_dir} does not exist; nothing to update")
        return 0
    if not normalized_map:
        return 0
    record_event("playlist_generator: updating existing playlists")

    docs_dir = os.path.join(root, "Docs")
    backup_dir = os.path.join(docs_dir, PLAYLIST_BACKUP_DIR)
    log_lines: list = []
    rewritten = 0

    for dirpath, dirnames, files in os.walk(playlists_dir):
        dirnames.sort()
        for fname in sorted(files):
            if not fname.lower().endswith((".m3u", ".m3u8")):
                continue
            pl_path = os.path.join(dirpath, fname)
            try:
                original = _read_playlist_text(pl_path)
            except OSError as exc:
                # Never write a playlist we could not read.
                log_callback(f"! Could not read {pl_path}; left unchanged: {exc}")
                log_lines.append(f"ERROR {pl_path}: unreadable, left unchanged: {exc}")
                continue

            new_text, n_changed = _rewrite_playlist_lines(
                original.text, dirpath, normalized_map
            )
            if not n_changed:
                continue
            try:
                data = _encode_playlist_text(pl_path, new_text, original.encoding)
            except UnicodeEncodeError as exc:
                log_callback(
                    f"! {pl_path} is {original.encoding}-encoded and can't hold the new "
                    f"file name(s); left unchanged: {exc}"
                )
                log_lines.append(f"ERROR {pl_path}: encoding {original.encoding}: {exc}")
                continue
            try:
                rel = os.path.relpath(pl_path, playlists_dir).replace(os.sep, "_")
                os.makedirs(backup_dir, exist_ok=True)
                bak = _backup_path(os.path.join(backup_dir, rel))
                shutil.copy2(pl_path, bak)
                log_lines.append(f"BACKUP {pl_path} -> {bak}")
            except OSError as exc:
                log_callback(f"! Could not back up {pl_path}; left unchanged: {exc}")
                log_lines.append(f"ERROR {pl_path}: backup failed, left unchanged: {exc}")
                continue
            try:
                with open(pl_path, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                log_callback(f"! Could not update {pl_path} (backup: {bak}): {exc}")
                log_lines.append(f"ERROR {pl_path}: {exc}")
                continue
            rewritten += 1
            log_lines.append(
                f"UPDATE {pl_path} ({n_changed} entr{'y' if n_changed == 1 else 'ies'})"
            )
            log_callback(f"✓ Updated playlist {pl_path}")

    _append_write_log(docs_dir, log_lines, log_callback)
    record_event("playlist_generator: playlist update complete")
    return rewritten


# ── Preview-first planning ────────────────────────────────────────────────


@dataclass
class PlannedPlaylist:
    """One playlist file that *would* be written, with its exact contents."""

    path: str
    lines: list = field(default_factory=list)
    exists: bool = False            # an existing file would be overwritten
    entries: int = 0                # track entries (non-comment lines)
    changes: list = field(default_factory=list)     # repair: (old_line, new_line)
    unresolved: list = field(default_factory=list)  # repair: entries still missing
    error: str = ""                 # repair: the playlist could not be read
    #: Encoding the file is written with. A repaired playlist keeps the
    #: encoding it was read with (``.m3u8`` is always UTF-8).
    encoding: str = "utf-8"
    #: ``(st_mtime_ns, st_size)`` of the file this item overwrites, as seen
    #: when the preview was made. Only the cluster playlist planner sets it
    #: (``plan_playlist`` does not); ``write_playlist_plan(verify=True)``
    #: skips an overwrite whose file no longer matches it.
    snapshot: tuple | None = None

    @property
    def action(self) -> str:
        return "overwrite" if self.exists else "new"


def relative_playlist_lines(tracks, outfile) -> list:
    """The entry lines a playlist at ``outfile`` gets for ``tracks``.

    Relative to the playlist's folder with native separators, and absolute
    only when no relative path exists (another drive on Windows): the rule
    every generated playlist follows. :func:`plan_playlist` and the cluster
    playlist export (``cluster_playlist_export``) both use it, so their
    entries can never differ.
    """
    base = os.path.dirname(os.path.abspath(outfile))
    return [safe_relpath(p, base) for p in tracks]


#: Backwards-compatible private name of :func:`relative_playlist_lines`.
_relative_lines = relative_playlist_lines


def plan_playlist(tracks, outfile) -> PlannedPlaylist:
    """Plan writing ``tracks`` to ``outfile`` (same format as :func:`write_playlist`)."""
    tracks = list(tracks)
    return PlannedPlaylist(
        path=os.path.abspath(outfile),
        lines=relative_playlist_lines(tracks, outfile),
        exists=os.path.exists(outfile),
        entries=len(tracks),
    )


def plan_folder_playlists(library_root, output_dir=None, valid_exts=None) -> list:
    """Plan one playlist per library folder that directly contains audio files.

    That is one playlist per *album folder* (``Artist/Album`` ->
    ``Artist_Album.m3u``), named exactly as :func:`generate_playlists` names
    them. The reserved top-level folders (Playlists, Docs, Trash, Quarantine,
    Not Sorted, Manual Review) are skipped.
    """
    from utils.reserved_folders import walk_library

    valid_exts = {e.lower() for e in (valid_exts or DEFAULT_EXTS)}
    playlists_dir = output_dir or os.path.join(library_root, "Playlists")
    by_dir = defaultdict(list)
    for dirpath, dirnames, filenames in walk_library(library_root):
        dirnames.sort()
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                by_dir[dirpath].append(os.path.join(dirpath, fname))

    used = set()
    plan = []
    for folder in sorted(by_dir):
        rel = os.path.relpath(folder, library_root)
        name = _sanitize_name(rel, used)
        used.add(name)
        outfile = os.path.join(playlists_dir, f"{name}.m3u")
        item = plan_playlist(by_dir[folder], outfile)
        item.lines = sorted(item.lines)  # generate_playlists writes them sorted
        plan.append(item)
    return plan


def _resolve_entry(entry, playlist_dir):
    is_abs = os.path.isabs(entry)
    path = entry if is_abs else os.path.join(playlist_dir, entry)
    return os.path.normpath(path), is_abs


def plan_playlist_repair(playlist_paths, library_root, prefer_opus=False, log_callback=None) -> list:
    """Plan repairs for ``playlist_paths`` (any folder) against ``library_root``.

    For every entry whose file no longer exists:

    1. a same-name sibling in the other codec is tried first (``.flac`` ->
       ``.opus`` when ``prefer_opus``, otherwise ``.opus`` -> ``.flac``);
    2. otherwise the library is searched by normalised file name
       (:func:`_pick_best_match`).

    Entries that cannot be resolved are kept unchanged and listed in
    ``unresolved``. Nothing is written; pass the items that have ``changes``
    to :func:`write_playlist_plan`.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    valid_exts = {e.lower() for e in DEFAULT_EXTS}
    name_index = token_index = None
    plan = []
    for pl_path in playlist_paths:
        pl_path = os.path.abspath(pl_path)
        pl_dir = os.path.dirname(pl_path)
        item = PlannedPlaylist(path=pl_path, exists=os.path.exists(pl_path))
        try:
            # A cp1252/latin-1 playlist (common for Windows-made .m3u files)
            # is read with a fallback instead of aborting the whole preview.
            original = _read_playlist_text(pl_path)
        except OSError as exc:
            item.error = str(exc)
            log_callback(f"! Could not read {pl_path}: {exc}")
            plan.append(item)
            continue
        item.encoding = _write_encoding(pl_path, original.encoding)
        raw_lines = [ln.rstrip("\r\n") for ln in original.text.splitlines()]

        for raw in raw_lines:
            entry = raw.strip()
            if not entry or entry.startswith("#"):
                item.lines.append(raw)
                continue
            item.entries += 1
            target, was_abs = _resolve_entry(entry, pl_dir)
            if os.path.exists(target):
                item.lines.append(raw)
                continue

            found = None
            stem, ext = os.path.splitext(target)
            ext = ext.lower()
            if ext == ".flac" and prefer_opus and os.path.exists(stem + ".opus"):
                found = stem + ".opus"
            elif ext == ".opus" and not prefer_opus and os.path.exists(stem + ".flac"):
                found = stem + ".flac"
            if found is None:
                if name_index is None:
                    name_index, token_index = _build_audio_index(library_root, valid_exts)
                found = _pick_best_match(target, name_index, token_index)

            if found:
                new_line = playlist_entry(found, pl_dir, like=entry)
                item.lines.append(new_line)
                item.changes.append((entry, new_line))
            else:
                item.lines.append(raw)
                item.unresolved.append(entry)
        plan.append(item)
    return plan


def _backup_path(path: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = f"{path}.{stamp}.bak"
    n = 2
    while os.path.exists(candidate):
        candidate = f"{path}.{stamp}-{n}.bak"
        n += 1
    return candidate


#: Reason ``write_playlist_plan(verify=True)`` reports for a skipped item.
CHANGED_SINCE_PREVIEW = "changed since the preview"


class _ChangedSincePreview(Exception):
    """Internal: the file an item targets is no longer what the preview saw."""


def _file_stamp(path: str):
    """``(st_mtime_ns, st_size)`` of ``path``, or None when it does not exist."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _verify_item(item) -> None:
    """Raise :class:`_ChangedSincePreview` when ``item``'s file changed since
    the preview: a "new" file that now exists, or an overwrite whose stamp no
    longer equals its ``snapshot``. A vanished overwrite is fine (written as
    new)."""
    now = _file_stamp(item.path)
    if now is None:
        return
    if not item.exists:
        raise _ChangedSincePreview()
    snapshot = getattr(item, "snapshot", None)
    if snapshot is not None and tuple(snapshot) != now:
        raise _ChangedSincePreview()


def write_playlist_plan(
    plan, backup=True, log_callback=None, log_dir=None, *, verify=False, cancel_event=None,
) -> dict:
    """Write each :class:`PlannedPlaylist` in ``plan`` exactly as planned.

    Any file that exists at write time is first copied to
    ``<name>.<timestamp>.bak`` beside it (when ``backup``). With ``log_dir``
    (normally ``<library>/Docs``) every write and backup is appended to
    ``playlist_writes.log`` there. Each file is written in its item's
    ``encoding``; an item whose lines can't be encoded that way is reported in
    ``errors`` and left untouched.

    ``verify=True`` re-checks each item against what the preview saw: an item
    planned as new whose file now exists, and an overwrite whose
    ``(st_mtime_ns, st_size)`` no longer equals its ``snapshot``, are skipped
    and reported as ``(path, "changed since the preview")``. An overwrite
    whose file has disappeared is written as new (nothing can be lost).

    ``cancel_event`` (a ``threading.Event``) stops writing *between* files:
    the files already written stay written, and the log lines gathered so far
    are still appended to ``playlist_writes.log``.

    Returns ``{"written": [...], "backups": [...], "errors": [(path, msg)],
    "cancelled": bool}``. With the defaults the behaviour is unchanged.
    """
    if log_callback is None:
        log_callback = lambda msg: None
    written, backups, errors, log_lines = [], [], [], []
    cancelled = False
    for item in plan:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            log_callback("! Cancelled: the remaining playlists were not written.")
            break
        path = item.path
        try:
            if verify:
                _verify_item(item)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            encoding =_write_encoding(path, getattr(item, "encoding", "") or "utf-8")
            try:
                data = "".join(line + "\n" for line in item.lines).encode(encoding)
            except UnicodeEncodeError as exc:
                # Never re-encode a playlist behind the user's back: report it.
                raise OSError(
                    f"the new entries can't be written in this playlist's encoding "
                    f"({encoding}): {exc}"
                ) from exc
            if backup and os.path.exists(path):
                bak = _backup_path(path)
                shutil.copy2(path, bak)
                backups.append(bak)
                log_lines.append(f"BACKUP {path} -> {bak}")
                log_callback(f"• Backed up {os.path.basename(path)} → {os.path.basename(bak)}")
            with open(path, "wb") as fh:
                fh.write(data)
            written.append(path)
            log_lines.append(f"WRITE {path} ({item.entries} entries)")
            log_callback(
                f"✓ Wrote {item.entries} entr{'y' if item.entries == 1 else 'ies'} → {path}"
            )
        except _ChangedSincePreview:
            errors.append((path, CHANGED_SINCE_PREVIEW))
            log_lines.append(f"ERROR {path}: {CHANGED_SINCE_PREVIEW}")
            log_callback(f"! Skipped {path}: {CHANGED_SINCE_PREVIEW}")
        except OSError as exc:
            errors.append((path, str(exc)))
            log_lines.append(f"ERROR {path}: {exc}")
            log_callback(f"! Could not write {path}: {exc}")
    _append_write_log(log_dir, log_lines, log_callback)
    return {"written": written, "backups": backups, "errors": errors, "cancelled": cancelled}
