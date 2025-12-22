"""Execute duplicate consolidation plans with safe ordering and reporting.

This module consumes the dry-run plan produced by ``duplicate_consolidation``
and performs the requested playlist rewrites, artwork transfers, metadata
normalization, and loser disposition handling (quarantine/delete). The
execution pipeline is intentionally conservative:

- Playlists are backed up before edits and validated after rewrites.
- A cancellation request stops processing after the current operation.
- Errors stop subsequent steps while preserving any backups already written.
- Every run produces machine-readable and human-readable reports.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from duplicate_consolidation import ArtworkDirective, ConsolidationPlan
from indexer_control import IndexCancelled, cancel_event as global_cancel_event
from utils.path_helpers import ensure_long_path

_MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
if _MUTAGEN_AVAILABLE:
    from mutagen import File as MutagenFile  # type: ignore
else:  # pragma: no cover - optional dependency
    MutagenFile = None  # type: ignore


def _default_log(msg: str) -> None:
    return None


@dataclass
class ExecutionConfig:
    """Configuration for consolidation execution."""

    library_root: str
    reports_dir: str
    playlists_dir: str | None = None
    quarantine_dir: str | None = None
    cancel_event: threading.Event | None = None
    log_callback: Callable[[str], None] | None = None
    apply_metadata: bool = True
    apply_artwork: bool = True


@dataclass
class ExecutionAction:
    """Individual action recorded in the audit log."""

    step: str
    target: str
    status: str
    detail: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "target": self.target,
            "status": self.status,
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }


@dataclass
class PlaylistRewriteResult:
    """Outcome of rewriting a single playlist."""

    playlist: str
    backup_path: str
    replaced_entries: int
    status: str
    detail: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "playlist": self.playlist,
            "backup_path": self.backup_path,
            "replaced_entries": self.replaced_entries,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class ExecutionResult:
    """Aggregate execution outcome and artifact locations."""

    success: bool
    actions: List[ExecutionAction]
    playlist_reports: List[PlaylistRewriteResult]
    quarantine_index: Dict[str, str]
    report_paths: Dict[str, str]


def _timestamped_dir(base: str) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"consolidation_run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=directory)
    tmp_path = tmp.name
    try:
        with open(tmp_path, "w", encoding=encoding) as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _atomic_write_json(path: str, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    _atomic_write_text(path, text)


def _iter_playlists(playlists_dir: str) -> Iterable[str]:
    if not os.path.isdir(playlists_dir):
        return []
    for dirpath, _dirs, files in os.walk(playlists_dir):
        for fname in files:
            if fname.lower().endswith(".m3u"):
                yield os.path.join(dirpath, fname)


def _normalize_playlist_entry(entry: str, playlist_dir: str) -> str:
    if not entry or entry.startswith("#"):
        return entry
    if os.path.isabs(entry):
        return os.path.normpath(entry)
    return os.path.normpath(os.path.join(playlist_dir, entry))


def _rewrite_playlist(path: str, mapping: Mapping[str, str]) -> tuple[int, List[str]]:
    playlist_dir = os.path.dirname(path)
    replaced = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [ln.rstrip("\n") for ln in handle]
    except FileNotFoundError:
        return 0, []

    new_lines: List[str] = []
    for ln in lines:
        normalized = _normalize_playlist_entry(ln, playlist_dir)
        if normalized in mapping:
            replaced += 1
            new_abs = mapping[normalized]
            rel = os.path.relpath(new_abs, playlist_dir)
            new_lines.append(rel)
        else:
            new_lines.append(ln)
    return replaced, new_lines


def _write_playlist_atomic(path: str, lines: Sequence[str]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=directory, encoding="utf-8")
    tmp_path = tmp.name
    try:
        with tmp:
            for ln in lines:
                tmp.write(ln + "\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _validate_playlist(path: str) -> List[str]:
    playlist_dir = os.path.dirname(path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [ln.rstrip("\n") for ln in handle]
    except FileNotFoundError:
        return ["Playlist missing after rewrite."]

    missing: List[str] = []
    for ln in lines:
        normalized = _normalize_playlist_entry(ln, playlist_dir)
        if not normalized or normalized.startswith("#"):
            continue
        if not os.path.exists(normalized):
            missing.append(normalized)
    return missing


def _backup_playlist(path: str, backup_root: str) -> str:
    rel = os.path.relpath(path, os.path.commonpath([path, backup_root]))
    dest = os.path.join(backup_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def _extract_artwork_bytes(path: str) -> bytes | None:
    sidecars = [f"{path}.artwork", f"{path}.cover", f"{path}.jpg"]
    for candidate in sidecars:
        if os.path.exists(candidate):
            with open(candidate, "rb") as handle:
                return handle.read()
    if MutagenFile is None:
        return None
    audio = MutagenFile(ensure_long_path(path))
    if audio is None:
        return None
    pictures = getattr(audio, "pictures", None)
    if pictures:
        pic = pictures[0]
        return bytes(pic.data) if hasattr(pic, "data") else bytes(pic)
    tags = getattr(audio, "tags", {}) or {}
    for key in list(tags.keys()):
        if key.startswith("APIC"):
            frame = tags[key]
            data = getattr(frame, "data", None)
            if data:
                return bytes(data)
    return None


def _apply_artwork(action: ArtworkDirective) -> bool:
    payload = _extract_artwork_bytes(action.source)
    if payload is None:
        return False
    sidecar = f"{action.target}.artwork"
    directory = os.path.dirname(sidecar)
    os.makedirs(directory, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=directory)
    tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_path, sidecar)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if MutagenFile is None:
        return True
    audio = MutagenFile(ensure_long_path(action.target))
    if audio is None:
        return True
    try:
        pics = getattr(audio, "pictures", None)
        if pics is not None:
            pics.clear()
            pics.append(payload)
        elif hasattr(audio, "tags"):
            audio.tags["APIC:Consolidation"] = payload
        audio.save()
    except Exception:
        return False
    return True


def _apply_metadata(target: str, planned_tags: Mapping[str, object]) -> bool:
    meta_path = f"{target}.metadata.json"
    try:
        _atomic_write_json(meta_path, dict(planned_tags))
    except Exception:
        return False
    if MutagenFile is None:
        return True
    audio = MutagenFile(ensure_long_path(target), easy=True)
    if audio is None:
        return True
    changed = False
    for key, value in planned_tags.items():
        if value is None:
            continue
        current = audio.tags.get(key) if audio.tags else None
        normalized = [str(value)] if not isinstance(value, list) else [str(v) for v in value]
        if current != normalized:
            audio[key] = normalized
            changed = True
    if changed:
        try:
            audio.save()
        except Exception:
            return False
    return True


def _quarantine_path(path: str, quarantine_root: str, library_root: str) -> str:
    try:
        rel = os.path.relpath(path, library_root)
        if rel.startswith(".."):
            rel = os.path.basename(path)
    except ValueError:
        rel = os.path.basename(path)
    dest = os.path.join(quarantine_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return dest


def execute_consolidation_plan(plan: ConsolidationPlan, config: ExecutionConfig) -> ExecutionResult:
    """Execute the provided consolidation plan with safety guarantees."""

    cancel = config.cancel_event or global_cancel_event
    log = config.log_callback or _default_log
    actions: List[ExecutionAction] = []
    playlist_results: List[PlaylistRewriteResult] = []
    quarantine_index: Dict[str, str] = {}

    run_root = _timestamped_dir(config.reports_dir)
    reports_dir = os.path.join(run_root, "reports")
    backups_dir = os.path.join(run_root, "playlist_backups")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)

    playlists_dir = config.playlists_dir or os.path.join(config.library_root, "Playlists")
    quarantine_dir = config.quarantine_dir or os.path.join(config.library_root, "Quarantine")

    loser_to_winner: Dict[str, str] = {}
    loser_disposition: Dict[str, str] = {}
    artwork_actions: List[ArtworkDirective] = []
    planned_tags: Dict[str, Dict[str, object]] = {}

    for group in plan.groups:
        planned_tags[group.winner_path] = dict(group.planned_winner_tags)
        for loser in group.losers:
            loser_to_winner[loser] = group.winner_path
            loser_disposition[loser] = group.loser_disposition.get(loser, "quarantine")
        artwork_actions.extend(group.artwork)

    def _record(step: str, target: str, status: str, detail: str, **metadata: object) -> None:
        actions.append(ExecutionAction(step=step, target=target, status=status, detail=detail, metadata=metadata))

    def _check_cancel(stage: str) -> None:
        if cancel.is_set():
            _record(stage, "execution", "cancelled", "Execution cancelled by user request.")
            raise IndexCancelled()

    success = True

    try:
        _check_cancel("start")

        # Step 1: backup playlists that will change
        impacted: List[str] = []
        for playlist in _iter_playlists(playlists_dir):
            try:
                with open(playlist, "r", encoding="utf-8") as handle:
                    lines = [ln.rstrip("\n") for ln in handle]
            except FileNotFoundError:
                continue
            playlist_dir = os.path.dirname(playlist)
            normalized = {_normalize_playlist_entry(ln, playlist_dir) for ln in lines if ln}
            if normalized & loser_to_winner.keys():
                impacted.append(playlist)

        for playlist in impacted:
            _check_cancel("backup_playlists")
            try:
                backup_path = _backup_playlist(playlist, backups_dir)
                _record("backup_playlists", playlist, "success", "Playlist backed up.", backup_path=backup_path)
            except Exception as exc:
                success = False
                _record("backup_playlists", playlist, "failed", f"Failed to backup playlist: {exc}")
                raise

        # Step 2: rewrite playlists
        for playlist in impacted:
            _check_cancel("rewrite_playlists")
            replaced, new_lines = _rewrite_playlist(playlist, loser_to_winner)
            if replaced == 0:
                playlist_results.append(
                    PlaylistRewriteResult(
                        playlist=playlist,
                        backup_path=_backup_playlist(playlist, backups_dir),
                        replaced_entries=0,
                        status="skipped",
                        detail="No matching loser entries found.",
                    )
                )
                _record("rewrite_playlists", playlist, "skipped", "No entries required updates.")
                continue
            try:
                _write_playlist_atomic(playlist, new_lines)
                playlist_results.append(
                    PlaylistRewriteResult(
                        playlist=playlist,
                        backup_path=os.path.join(backups_dir, os.path.relpath(playlist, playlists_dir)),
                        replaced_entries=replaced,
                        status="success",
                    )
                )
                _record("rewrite_playlists", playlist, "success", "Playlist rewritten.", replaced=replaced)
            except Exception as exc:
                success = False
                _record("rewrite_playlists", playlist, "failed", f"Failed to rewrite playlist: {exc}")
                raise

        # Step 3: validate rewritten playlists
        for result in playlist_results:
            _check_cancel("validate_playlists")
            missing = _validate_playlist(result.playlist)
            if missing:
                success = False
                try:
                    if os.path.exists(result.backup_path):
                        shutil.copy2(result.backup_path, result.playlist)
                finally:
                    _record(
                        "validate_playlists",
                        result.playlist,
                        "failed",
                        "Playlist validation failed; restored backup.",
                        missing=missing,
                    )
                raise RuntimeError(f"Playlist validation failed for {result.playlist}")
            _record("validate_playlists", result.playlist, "success", "Playlist validated.")

        # Step 4: apply artwork transfers
        if config.apply_artwork:
            for art in artwork_actions:
                _check_cancel("artwork")
                ok = _apply_artwork(art)
                status = "success" if ok else "failed"
                if not ok:
                    success = False
                _record("artwork", art.target, status, art.reason, source=art.source)
                if not ok:
                    raise RuntimeError(f"Artwork copy failed for {art.target}")

        # Step 5: apply metadata normalization
        if config.apply_metadata:
            for target, tags in planned_tags.items():
                _check_cancel("metadata")
                ok = _apply_metadata(target, tags)
                status = "success" if ok else "failed"
                if not ok:
                    success = False
                _record("metadata", target, status, "Metadata normalized.")
                if not ok:
                    raise RuntimeError(f"Metadata normalization failed for {target}")

        # Step 6: quarantine or delete losers
        for loser, disposition in loser_disposition.items():
            _check_cancel("loser_cleanup")
            if disposition == "delete":
                try:
                    if os.path.exists(loser):
                        os.remove(loser)
                    quarantine_index[loser] = "deleted"
                    _record("loser_cleanup", loser, "success", "Loser deleted.")
                except Exception as exc:
                    success = False
                    _record("loser_cleanup", loser, "failed", f"Failed to delete loser: {exc}")
                    raise
            else:
                dest = _quarantine_path(loser, quarantine_dir, config.library_root)
                try:
                    if os.path.exists(loser):
                        shutil.move(loser, dest)
                    quarantine_index[loser] = dest
                    _record("loser_cleanup", loser, "success", "Loser quarantined.", quarantine_path=dest)
                except Exception as exc:
                    success = False
                    _record("loser_cleanup", loser, "failed", f"Failed to quarantine loser: {exc}")
                    raise
    except IndexCancelled:
        success = False
    except Exception:
        success = False
    finally:
        audit_path = os.path.join(reports_dir, "audit.json")
        playlist_report_path = os.path.join(reports_dir, "playlist_report.json")
        quarantine_index_path = os.path.join(reports_dir, "quarantine_index.json")
        html_report_path = os.path.join(reports_dir, "execution_report.html")

        audit_payload = {
            "success": success,
            "actions": [a.to_dict() for a in actions],
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        playlist_payload = {
            "backups_dir": backups_dir,
            "playlists": [p.to_dict() for p in playlist_results],
        }
        quarantine_payload = dict(quarantine_index)

        try:
            _atomic_write_json(audit_path, audit_payload)
            _atomic_write_json(playlist_report_path, playlist_payload)
            _atomic_write_json(quarantine_index_path, quarantine_payload)
        except Exception:
            pass

        try:
            html_lines = [
                "<html><head><title>Duplicate Consolidation Execution</title></head><body>",
                "<h1>Execution Report</h1>",
                f"<p>Status: {'success' if success else 'failed'}</p>",
                "<ul>",
            ]
            for action in actions:
                html_lines.append(
                    f"<li><strong>{action.step}</strong> — {action.target}: {action.status} ({action.detail})"  # noqa: E501
                    "</li>"
                )
            html_lines.append("</ul>")
            if playlist_results:
                html_lines.append("<h2>Playlist Changes</h2><ul>")
                for res in playlist_results:
                    html_lines.append(
                        f"<li>{res.playlist} → {res.status} (replaced {res.replaced_entries}); "
                        f"backup: {res.backup_path}</li>"
                    )
                html_lines.append("</ul>")
            if quarantine_index:
                html_lines.append("<h2>Quarantined/Deleted</h2><ul>")
                for loser, dest in quarantine_index.items():
                    html_lines.append(f"<li>{loser} → {dest}</li>")
                html_lines.append("</ul>")
            html_lines.append("</body></html>")
            _atomic_write_text(html_report_path, "\n".join(html_lines))
        except Exception:
            pass

    report_paths = {
        "audit": audit_path,
        "playlist_report": playlist_report_path,
        "quarantine_index": quarantine_index_path,
        "html_report": html_report_path,
        "reports_root": reports_dir,
        "backups_root": backups_dir,
    }

    return ExecutionResult(
        success=success,
        actions=actions,
        playlist_reports=playlist_results,
        quarantine_index=quarantine_index,
        report_paths=report_paths,
    )

