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
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from duplicate_consolidation import (
    ArtworkDirective,
    ConsolidationPlan,
    _capture_library_state,
    consolidation_plan_from_dict,
)
from indexer_control import IndexCancelled, cancel_event as global_cancel_event
from utils.path_helpers import ensure_long_path

try:
    _MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
except ValueError:  # pragma: no cover - defensive: broken mutagen installs
    _MUTAGEN_AVAILABLE = False
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
    allow_review_required: bool = False
    operation_limit: int | None = 500
    confirm_operation_overage: bool = False
    allow_deletion: bool = False
    confirm_deletion: bool = False
    dry_run_execute: bool = False
    retain_losers: bool = False


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
    original_playlist: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "playlist": self.playlist,
            "backup_path": self.backup_path,
            "replaced_entries": self.replaced_entries,
            "status": self.status,
            "detail": self.detail,
            "original_playlist": self.original_playlist,
        }


@dataclass
class ExecutionResult:
    """Aggregate execution outcome and artifact locations."""

    success: bool
    actions: List[ExecutionAction]
    playlist_reports: List[PlaylistRewriteResult]
    quarantine_index: Dict[str, str]
    report_paths: Dict[str, str]


EXECUTION_REPORT_FILENAME = "execution_report.html"
INTERNAL_TAG_KEYS = {"album_type"}


def _timestamped_dir(base: str) -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"consolidation_run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def _atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> None:
    directory = os.path.dirname(path) or "."
    safe_directory = ensure_long_path(directory)
    safe_path = ensure_long_path(path)
    os.makedirs(safe_directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=safe_directory, encoding=encoding) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        os.replace(ensure_long_path(tmp_path), safe_path)
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


def _rewrite_playlist(path: str, mapping: Mapping[str, str], *, base_dir: str | None = None) -> tuple[int, List[str]]:
    playlist_dir = base_dir or os.path.dirname(path)
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


def _validate_playlist(path: str, *, base_dir: str | None = None) -> List[str]:
    playlist_dir = base_dir or os.path.dirname(path)
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
    try:
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
    finally:
        _close_mutagen_audio(audio)
    return None


def _apply_artwork(action: ArtworkDirective) -> tuple[bool, str]:
    payload = _extract_artwork_bytes(action.source)
    if payload is None:
        return False, "No artwork payload available to embed."

    sidecar = f"{action.target}.artwork"
    directory = os.path.dirname(sidecar)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=directory) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name
    try:
        os.replace(tmp_path, sidecar)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if MutagenFile is None:
        return True, "Mutagen unavailable; wrote artwork sidecar only."

    target_path = ensure_long_path(action.target)
    audio = MutagenFile(target_path)
    try:
        if audio is None:
            return False, "Unsupported audio format for embedding."

        ext = os.path.splitext(action.source)[1].lower()
        mime = "image/jpeg"
        if ext == ".png":
            mime = "image/png"

        try:
            if hasattr(audio, "pictures"):
                # FLAC/APE: replace front cover while preserving other pictures.
                from mutagen.flac import Picture  # type: ignore

                existing = [p for p in getattr(audio, "pictures") if getattr(p, "type", None) != 3]
                pic = Picture()
                pic.data = payload
                pic.type = 3
                pic.mime = mime
                pic.desc = "Front cover"
                audio.clear_pictures()
                for p in existing:
                    audio.add_picture(p)
                audio.add_picture(pic)
                audio.save()
                return True, "Embedded FLAC/APE picture."
            if audio.__class__.__module__.startswith("mutagen.mp4"):
                from mutagen.mp4 import MP4, MP4Cover  # type: ignore

                mp4 = audio if isinstance(audio, MP4) else MP4(target_path)
                cover_format = MP4Cover.FORMAT_PNG if mime == "image/png" else MP4Cover.FORMAT_JPEG
                mp4["covr"] = [MP4Cover(payload, imageformat=cover_format)]
                mp4.save()
                return True, "Embedded MP4 cover atom."
            # Default to ID3/APIC handling
            from mutagen.id3 import APIC, ID3, ID3NoHeaderError  # type: ignore

            try:
                tags = ID3(target_path)
            except ID3NoHeaderError:
                tags = ID3()
            preserved = [frame for frame in tags.getall("APIC") if getattr(frame, "type", None) != 3]
            tags.delall("APIC")
            for frame in preserved:
                tags.add(frame)
            tags.add(APIC(encoding=3, mime=mime, type=3, desc="Front cover", data=payload))
            tags.save(target_path)
            return True, "Embedded ID3 cover art."
        except Exception as exc:
            return False, f"Artwork embed failed: {exc}"
    finally:
        _close_mutagen_audio(audio)


def _apply_metadata(target: str, planned_tags: Mapping[str, object]) -> tuple[bool, str]:
    meta_path = f"{target}.metadata.json"
    try:
        _atomic_write_json(meta_path, dict(planned_tags))
    except Exception:
        return False, "Failed to write metadata sidecar."
    if MutagenFile is None:
        return True, "Mutagen unavailable; wrote metadata sidecar only."
    audio = MutagenFile(ensure_long_path(target), easy=True)
    try:
        if audio is None:
            return True, "Unsupported audio format; wrote metadata sidecar only."
        changed = False
        skipped: List[str] = []
        for key, value in planned_tags.items():
            if value is None:
                continue
            if key in INTERNAL_TAG_KEYS:
                skipped.append(key)
                continue
            normalized = [str(value)] if not isinstance(value, list) else [str(v) for v in value]
            try:
                current = audio.tags.get(key) if audio.tags else None
                if current != normalized:
                    audio[key] = normalized
                    changed = True
            except Exception:
                skipped.append(key)
        if changed:
            try:
                audio.save()
            except Exception:
                return False, "Failed to save metadata tags."
        if skipped:
            skipped_list = ", ".join(sorted(set(skipped)))
            return True, f"Metadata normalized; skipped unsupported keys: {skipped_list}."
        return True, "Metadata normalized."
    finally:
        _close_mutagen_audio(audio)


def _close_mutagen_audio(audio: object | None) -> None:
    if audio is None:
        return
    closer = getattr(audio, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass
    fileobj = getattr(audio, "fileobj", None)
    if fileobj is not None:
        file_close = getattr(fileobj, "close", None)
        if callable(file_close):
            try:
                file_close()
            except Exception:
                pass


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


def _compute_plan_signature(plan: ConsolidationPlan) -> str:
    canonical = {
        "generated_at": plan.generated_at.isoformat(),
        "groups": [g.to_dict() for g in plan.groups],
        "snapshot": {k: dict(v) for k, v in plan.source_snapshot.items()},
    }
    text = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _detect_state_drift(expected: Mapping[str, object], actual: Mapping[str, object]) -> str | None:
    if not expected:
        return "Missing expected snapshot."
    if not actual.get("exists"):
        if expected.get("exists"):
            return "File missing since preview."
        return None
    if expected.get("exists") is False and actual.get("exists"):
        return "File appeared after preview."
    for key in ("sha256", "size"):
        if key in expected and key in actual and expected[key] != actual[key]:
            return f"{key} mismatch (expected {expected.get(key)}, found {actual.get(key)})."
    return None


def _coerce_consolidation_plan(
    plan_input: ConsolidationPlan | Mapping[str, object] | str,
    log: Callable[[str], None],
) -> tuple[ConsolidationPlan, str]:
    if isinstance(plan_input, ConsolidationPlan):
        return plan_input, "in-memory plan"
    payload: object = plan_input
    source = "plan payload"
    if isinstance(plan_input, str):
        if os.path.exists(plan_input):
            if not os.path.isfile(plan_input):
                raise ValueError(f"Plan path is not a file: {plan_input}")
            try:
                with open(plan_input, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except OSError as exc:
                raise ValueError(f"Plan file could not be read: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f"Plan JSON could not be decoded: {exc}") from exc
            source = f"preview output {plan_input}"
        else:
            try:
                payload = json.loads(plan_input)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Plan JSON could not be decoded: {exc}") from exc
            source = "json payload"
    if isinstance(payload, Mapping):
        if "plan" in payload:
            inner = payload.get("plan")
            if not isinstance(inner, Mapping):
                raise ValueError("Plan payload has a non-mapping 'plan' field.")
            payload = inner
            source = f"{source} (wrapped)"
        return consolidation_plan_from_dict(payload), source
    raise ValueError("Plan input must be a ConsolidationPlan, mapping, or JSON string.")


def execute_consolidation_plan(
    plan_input: ConsolidationPlan | Mapping[str, object] | str,
    config: ExecutionConfig,
) -> ExecutionResult:
    """Execute the provided consolidation plan with safety guarantees."""

    cancel = config.cancel_event or global_cancel_event
    log = config.log_callback or _default_log
    actions: List[ExecutionAction] = []
    playlist_results: List[PlaylistRewriteResult] = []
    quarantine_index: Dict[str, str] = {}
    group_lookup: Dict[str, object] = {}
    plan: ConsolidationPlan | None = None
    computed_signature = ""

    run_root = _timestamped_dir(config.reports_dir)
    reports_dir = os.path.join(run_root, "reports")
    backups_dir = os.path.join(run_root, "playlist_backups")
    validation_playlists_dir = os.path.join(run_root, "playlist_validation")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(backups_dir, exist_ok=True)

    playlists_dir = config.playlists_dir or os.path.join(config.library_root, "Playlists")
    quarantine_dir = config.quarantine_dir or os.path.join(config.library_root, "Quarantine")

    def _record(step: str, target: str, status: str, detail: str, **metadata: object) -> None:
        actions.append(ExecutionAction(step=step, target=target, status=status, detail=detail, metadata=metadata))

    def _check_cancel(stage: str) -> None:
        if cancel.is_set():
            _record(stage, "execution", "cancelled", "Execution cancelled by user request.")
            raise IndexCancelled()

    loser_to_winner: Dict[str, str] = {}
    loser_disposition: Dict[str, str] = {}
    artwork_actions: List[ArtworkDirective] = []
    planned_tags: Dict[str, Dict[str, object]] = {}
    path_to_group: Dict[str, str] = {}
    playlist_targets: Dict[str, str] = {}
    playlist_base_dirs: Dict[str, str] = {}

    def _find_loser_references(
        working_playlists: Mapping[str, str],
        playlist_base_dirs: Mapping[str, str] | None,
    ) -> Dict[str, List[str]]:
        refs: Dict[str, List[str]] = {}
        for playlist, working_path in working_playlists.items():
            try:
                with open(working_path, "r", encoding="utf-8") as handle:
                    lines = [ln.rstrip("\n") for ln in handle]
            except FileNotFoundError:
                continue
            playlist_dir = (playlist_base_dirs or {}).get(playlist) or os.path.dirname(working_path)
            normalized = {_normalize_playlist_entry(ln, playlist_dir) for ln in lines if ln}
            hits = sorted([loser for loser in loser_to_winner.keys() if loser in normalized])
            if hits:
                refs[playlist] = hits
        return refs

    success = True

    try:
        try:
            plan, plan_source = _coerce_consolidation_plan(plan_input, log)
            log(f"Execute plan input resolved from {plan_source}.")
        except ValueError as exc:
            success = False
            _record("preflight", "plan", "blocked", f"Invalid consolidation plan: {exc}")
            raise RuntimeError(f"Execution blocked: invalid consolidation plan ({exc})") from exc

        group_lookup = {g.group_id: g for g in plan.groups}
        for group in plan.groups:
            planned_tags[group.winner_path] = dict(group.planned_winner_tags)
            for loser in group.losers:
                loser_to_winner[loser] = group.winner_path
                loser_disposition[loser] = group.loser_disposition.get(loser, "quarantine")
                path_to_group[loser] = group.group_id
            path_to_group[group.winner_path] = group.group_id
            artwork_actions.extend(group.artwork)

        computed_signature = _compute_plan_signature(plan)
        _check_cancel("start")

        if plan.plan_signature and plan.plan_signature != computed_signature:
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Plan signature mismatch; rerun preview to regenerate the plan.",
                expected_signature=plan.plan_signature,
                computed_signature=computed_signature,
            )
            raise RuntimeError("Execution blocked: plan signature mismatch.")

        if not plan.source_snapshot:
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Plan missing source snapshot; rerun preview to capture library state.",
            )
            raise RuntimeError("Execution blocked: missing source snapshot.")

        if plan.requires_review and not config.allow_review_required:
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Review-required groups present; enable override to continue.",
                review_required=plan.review_required_count,
            )
            raise RuntimeError("Execution blocked: review-required groups present.")

        planned_operations = len(loser_to_winner) + len(artwork_actions) + len(planned_tags)
        if config.operation_limit and planned_operations > config.operation_limit and not config.confirm_operation_overage:
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Planned operations exceed configured limit.",
                planned_operations=planned_operations,
                operation_limit=config.operation_limit,
            )
            raise RuntimeError("Execution blocked: operation limit exceeded.")

        drift: Dict[str, str] = {}
        for path, expected in plan.source_snapshot.items():
            actual = _capture_library_state(path)
            delta = _detect_state_drift(expected, actual)
            if delta:
                drift[path] = delta
        if drift:
            success = False
            for path, reason in drift.items():
                _record(
                    "preflight",
                    path,
                    "failed",
                    f"Library changed since preview: {reason}",
                    group_id=path_to_group.get(path),
                )
            raise RuntimeError("Execution blocked: library drift detected.")

        has_deletions = any(disposition == "delete" for disposition in loser_disposition.values())
        if has_deletions and not config.dry_run_execute:
            if not (config.allow_deletion and config.confirm_deletion):
                success = False
                _record(
                    "preflight",
                    "execution",
                    "blocked",
                    "Deletion requested but explicit confirmation was not provided.",
                    deletions_requested=sum(1 for d in loser_disposition.values() if d == "delete"),
                )
                raise RuntimeError("Execution blocked: deletions require confirmation.")

        # Step 1: backup playlists that will change
        impacted: List[str] = []
        playlist_groups: Dict[str, set[str]] = {}
        for playlist in _iter_playlists(playlists_dir):
            try:
                with open(playlist, "r", encoding="utf-8") as handle:
                    lines = [ln.rstrip("\n") for ln in handle]
            except FileNotFoundError:
                continue
            playlist_dir = os.path.dirname(playlist)
            normalized = {_normalize_playlist_entry(ln, playlist_dir) for ln in lines if ln}
            affected = normalized & loser_to_winner.keys()
            if affected:
                impacted.append(playlist)
                playlist_groups[playlist] = {path_to_group.get(path, "") for path in affected if path in path_to_group}

        playlist_backups: Dict[str, str] = {}
        for playlist in impacted:
            _check_cancel("backup_playlists")
            try:
                if config.dry_run_execute:
                    working_copy = os.path.join(validation_playlists_dir, os.path.relpath(playlist, playlists_dir))
                    os.makedirs(os.path.dirname(working_copy), exist_ok=True)
                    shutil.copy2(playlist, working_copy)
                    playlist_backups[playlist] = working_copy
                    playlist_targets[playlist] = working_copy
                    playlist_base_dirs[playlist] = os.path.dirname(playlist)
                    _record(
                        "backup_playlists",
                        playlist,
                        "success",
                        "Playlist copied for dry-run validation.",
                        backup_path=working_copy,
                        group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    )
                else:
                    backup_path = _backup_playlist(playlist, backups_dir)
                    playlist_backups[playlist] = backup_path
                    playlist_targets[playlist] = playlist
                    playlist_base_dirs[playlist] = os.path.dirname(playlist)
                    _record(
                        "backup_playlists",
                        playlist,
                        "success",
                        "Playlist backed up.",
                        backup_path=backup_path,
                        group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    )
            except Exception as exc:
                success = False
                _record("backup_playlists", playlist, "failed", f"Failed to backup playlist: {exc}")
                raise

        # Step 2: rewrite playlists
        for playlist in impacted:
            _check_cancel("rewrite_playlists")
            target_playlist = playlist_targets.get(playlist, playlist)
            base_dir = playlist_base_dirs.get(playlist, os.path.dirname(target_playlist))
            replaced, new_lines = _rewrite_playlist(target_playlist, loser_to_winner, base_dir=base_dir)
            if replaced == 0:
                playlist_results.append(
                    PlaylistRewriteResult(
                        playlist=target_playlist,
                        backup_path=playlist_backups.get(playlist, _backup_playlist(playlist, backups_dir)),
                        replaced_entries=0,
                        status="skipped",
                        detail="No matching loser entries found." + (" (dry-run copy)" if config.dry_run_execute else ""),
                        original_playlist=playlist if target_playlist != playlist else None,
                    )
                )
                _record(
                    "rewrite_playlists",
                    target_playlist,
                    "skipped",
                    "No entries required updates.",
                    group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    original_playlist=playlist if target_playlist != playlist else None,
                )
                continue
            try:
                _write_playlist_atomic(target_playlist, new_lines)
                playlist_results.append(
                    PlaylistRewriteResult(
                        playlist=target_playlist,
                        backup_path=playlist_backups.get(
                            playlist,
                            os.path.join(
                                backups_dir,
                                os.path.relpath(playlist, playlists_dir),
                            ),
                        ),
                        replaced_entries=replaced,
                        status="success",
                        original_playlist=playlist if target_playlist != playlist else None,
                    )
                )
                _record(
                    "rewrite_playlists",
                    target_playlist,
                    "success",
                    "Playlist rewritten.",
                    replaced=replaced,
                    group_ids=sorted(g for g in playlist_groups.get(playlist, []) if g),
                    original_playlist=playlist if target_playlist != playlist else None,
                )
            except Exception as exc:
                success = False
                _record("rewrite_playlists", target_playlist, "failed", f"Failed to rewrite playlist: {exc}")
                raise

        # Step 3: validate rewritten playlists
        for result in playlist_results:
            _check_cancel("validate_playlists")
            base_dir = playlist_base_dirs.get(result.original_playlist or result.playlist, os.path.dirname(result.playlist))
            missing = _validate_playlist(result.playlist, base_dir=base_dir)
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
                        group_ids=sorted(g for g in playlist_groups.get(result.original_playlist or result.playlist, []) if g),
                        original_playlist=result.original_playlist,
                    )
                raise RuntimeError(f"Playlist validation failed for {result.playlist}")
            _record(
                "validate_playlists",
                result.playlist,
                "success",
                "Playlist validated.",
                group_ids=sorted(g for g in playlist_groups.get(result.original_playlist or result.playlist, []) if g),
                original_playlist=result.original_playlist,
            )

        lingering_refs = _find_loser_references(playlist_targets or {}, playlist_base_dirs)
        if lingering_refs:
            success = False
            for playlist, losers in lingering_refs.items():
                _record(
                    "validate_playlists",
                    playlist,
                    "failed",
                    "Loser references remain after rewrite; aborting cleanup.",
                    losers=losers,
                    group_ids=sorted({path_to_group.get(l, "") for l in losers if l in path_to_group}),
                )
            raise RuntimeError("Playlist rewrites incomplete; loser references remain.")

        # Step 4: apply artwork transfers
        if config.dry_run_execute and config.apply_artwork:
            for art in artwork_actions:
                _record(
                    "artwork",
                    art.target,
                    "skipped",
                    "Dry-run execute enabled; artwork not applied.",
                    source=art.source,
                    group_id=path_to_group.get(art.target),
                )
        elif config.apply_artwork:
            for art in artwork_actions:
                _check_cancel("artwork")
                ok, detail = _apply_artwork(art)
                status = "success" if ok else "failed"
                if not ok:
                    success = False
                _record(
                    "artwork",
                    art.target,
                    status,
                    detail or art.reason,
                    source=art.source,
                    group_id=path_to_group.get(art.target),
                )
                if not ok:
                    raise RuntimeError(f"Artwork copy failed for {art.target}")

        # Step 5: apply metadata normalization
        if config.dry_run_execute and config.apply_metadata:
            for target, _tags in planned_tags.items():
                _record(
                    "metadata",
                    target,
                    "skipped",
                    "Dry-run execute enabled; metadata not written.",
                    group_id=path_to_group.get(target),
                )
        elif config.apply_metadata:
            for target, tags in planned_tags.items():
                _check_cancel("metadata")
                ok, detail = _apply_metadata(target, tags)
                status = "success" if ok else "failed"
                if not ok:
                    success = False
                _record("metadata", target, status, detail, group_id=path_to_group.get(target))
                if not ok:
                    raise RuntimeError(f"Metadata normalization failed for {target}")

        # Step 6: quarantine or delete losers
        for loser, disposition in loser_disposition.items():
            _check_cancel("loser_cleanup")
            effective = disposition if disposition in ("retain", "quarantine", "delete") else None
            if effective is None:
                effective = "retain" if config.retain_losers else "quarantine"
            if effective == "retain":
                quarantine_index[loser] = "retained"
                detail = "Loser retained by plan disposition."
                if config.dry_run_execute:
                    detail = "Dry-run execute enabled; loser retained in place."
                _record(
                    "loser_cleanup",
                    loser,
                    "skipped",
                    detail,
                    disposition=effective,
                    group_id=path_to_group.get(loser),
                )
                continue
            if config.dry_run_execute:
                quarantine_index[loser] = "retained"
                _record(
                    "loser_cleanup",
                    loser,
                    "skipped",
                    "Dry-run execute enabled; loser not moved or deleted.",
                    disposition=effective,
                    group_id=path_to_group.get(loser),
                )
                continue
            if effective == "delete":
                try:
                    if os.path.exists(loser):
                        os.remove(loser)
                    quarantine_index[loser] = "deleted"
                    _record(
                        "loser_cleanup",
                        loser,
                        "success",
                        "Loser deleted.",
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                    )
                except Exception as exc:
                    success = False
                    _record(
                        "loser_cleanup",
                        loser,
                        "failed",
                        f"Failed to delete loser: {exc}",
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                    )
                    raise
            else:
                dest = _quarantine_path(loser, quarantine_dir, config.library_root)
                try:
                    if os.path.exists(loser):
                        shutil.move(loser, dest)
                    quarantine_index[loser] = dest
                    _record(
                        "loser_cleanup",
                        loser,
                        "success",
                        "Loser quarantined.",
                        quarantine_path=dest,
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                    )
                except Exception as exc:
                    success = False
                    _record(
                        "loser_cleanup",
                        loser,
                        "failed",
                        f"Failed to quarantine loser: {exc}",
                        quarantine_path=dest,
                        disposition=effective,
                        group_id=path_to_group.get(loser),
                    )
                    raise
    except IndexCancelled:
        success = False
        _record("execution", "execution", "cancelled", "Execution cancelled by user request.")
    except Exception as exc:
        success = False
        _record("execution", "execution", "failed", f"Execution failed: {exc}")
    finally:
        audit_path = os.path.join(reports_dir, "audit.json")
        playlist_report_path = os.path.join(reports_dir, "playlist_report.json")
        quarantine_index_path = os.path.join(reports_dir, "quarantine_index.json")
        html_report_path = os.path.join(reports_dir, EXECUTION_REPORT_FILENAME)

        plan_signature = plan.plan_signature if plan else None
        source_snapshot = plan.source_snapshot if plan else {}
        audit_payload = {
            "success": success,
            "actions": [a.to_dict() for a in actions],
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "plan_signature": plan_signature or computed_signature,
            "computed_signature": computed_signature,
            "source_snapshot": {k: dict(v) for k, v in source_snapshot.items()},
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
            actions_by_group: Dict[str, List[ExecutionAction]] = {}
            for action in actions:
                meta = action.metadata or {}
                group_ids: set[str] = set()
                gid = meta.get("group_id")
                if isinstance(gid, str) and gid:
                    group_ids.add(gid)
                meta_groups = meta.get("group_ids")
                if isinstance(meta_groups, (list, tuple, set)):
                    group_ids.update([g for g in meta_groups if isinstance(g, str) and g])
                if not group_ids:
                    actions_by_group.setdefault("__general__", []).append(action)
                else:
                    for gid in group_ids:
                        actions_by_group.setdefault(gid, []).append(action)

            def _basename(path: str) -> str:
                return os.path.basename(path) or path

            def _group_state(group_actions: List[ExecutionAction]) -> str:
                statuses = {act.status for act in group_actions}
                for state in ("failed", "blocked", "cancelled"):
                    if state in statuses:
                        return state
                if statuses == {"skipped"}:
                    return "skipped"
                return "success" if "success" in statuses else "unknown"

            def _action_badges(group_actions: List[ExecutionAction]) -> List[str]:
                badges: List[str] = []
                meta_actions = [act for act in group_actions if act.step == "metadata"]
                meta_success = sum(1 for act in meta_actions if act.status == "success")
                meta_skipped = sum(1 for act in meta_actions if act.status == "skipped")
                meta_failed = sum(1 for act in meta_actions if act.status == "failed")
                if meta_success:
                    badges.append(f"metadata normalized ({meta_success})")
                if meta_skipped:
                    badges.append(f"metadata skipped ({meta_skipped})")
                if meta_failed:
                    badges.append(f"metadata failed ({meta_failed})")

                cleanup_actions = [act for act in group_actions if act.step == "loser_cleanup"]
                quarantined = sum(
                    1
                    for act in cleanup_actions
                    if act.metadata.get("disposition") == "quarantine"
                )
                deleted = sum(
                    1
                    for act in cleanup_actions
                    if act.metadata.get("disposition") == "delete"
                )
                retained = sum(
                    1
                    for act in cleanup_actions
                    if act.metadata.get("disposition") == "retain"
                )
                if quarantined:
                    badges.append(f"quarantined ({quarantined})")
                if deleted:
                    badges.append(f"deleted ({deleted})")
                if retained:
                    badges.append(f"retained ({retained})")

                art_actions = [act for act in group_actions if act.step == "artwork"]
                art_success = sum(1 for act in art_actions if act.status == "success")
                if art_success:
                    badges.append(f"artwork copied ({art_success})")
                return badges

            def _format_metadata_notes(meta: Dict[str, object]) -> str:
                notes = []
                for key, value in meta.items():
                    if key in {"group_id", "group_ids"}:
                        continue
                    if isinstance(value, (list, tuple, set)):
                        rendered = ", ".join(str(v) for v in value)
                    else:
                        rendered = str(value)
                    notes.append(f"{key}: {rendered}")
                return " | ".join(notes)

            metadata_actions = [act for act in actions if act.step == "metadata"]
            metadata_success = sum(1 for act in metadata_actions if act.status == "success")
            metadata_failed = sum(1 for act in metadata_actions if act.status == "failed")
            metadata_skipped = sum(1 for act in metadata_actions if act.status == "skipped")
            quarantined_count = sum(
                1 for dest in quarantine_index.values() if dest not in {"retained", "deleted"}
            )
            deleted_count = sum(1 for dest in quarantine_index.values() if dest == "deleted")
            groups_processed = len(group_lookup)
            winners_kept = groups_processed

            html_lines = [
                "<html><head><title>Duplicate Consolidation Execution</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:18px;color:#1b1b1b;}",
                "h1{margin-bottom:6px;}",
                ".summary-card{border:1px solid #dcdcdc;border-radius:10px;padding:14px;background:#f9fafb;}",
                ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;}",
                ".summary-item{background:white;border:1px solid #eee;border-radius:8px;padding:10px;}",
                ".summary-label{font-size:0.8em;color:#666;text-transform:uppercase;letter-spacing:.04em;}",
                ".summary-value{font-size:1.1em;font-weight:600;margin-top:4px;}",
                ".group{border:1px solid #ddd;padding:10px;margin-bottom:10px;border-radius:8px;}",
                ".group-summary{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}",
                ".group-title{font-weight:600;}",
                ".badge{background:#eef2ff;color:#1a237e;border-radius:999px;padding:2px 8px;font-size:0.8em;}",
                ".status-success{color:#1b7a1b;}",
                ".status-failed{color:#a30000;}",
                ".status-blocked{color:#a35b00;}",
                ".status-cancelled{color:#8a4b00;}",
                ".status-skipped{color:#555;}",
                ".muted{color:#666;font-size:0.9em;}",
                ".context{margin-top:10px;}",
                "details>summary{cursor:pointer;list-style:none;}",
                "details>summary::-webkit-details-marker{display:none;}",
                "table{width:100%;border-collapse:collapse;margin-top:8px;}",
                "th,td{border-bottom:1px solid #eee;padding:6px 4px;text-align:left;vertical-align:top;}",
                "th{font-size:0.85em;color:#555;}",
                "code{background:#f3f4f6;padding:2px 4px;border-radius:4px;font-size:0.85em;}",
                ".meta{color:#555;font-size:0.85em;margin-top:2px;}",
                "</style></head><body>",
                "<h1>Execution Report</h1>",
                "<div class='summary-card'>",
                "<div class='summary-grid'>",
                f"<div class='summary-item'><div class='summary-label'>Overall status</div>"
                f"<div class='summary-value'>{'success' if success else 'failed'}</div></div>",
                f"<div class='summary-item'><div class='summary-label'>Groups processed</div>"
                f"<div class='summary-value'>{groups_processed}</div></div>",
                f"<div class='summary-item'><div class='summary-label'>Winners kept</div>"
                f"<div class='summary-value'>{winners_kept}</div></div>",
                f"<div class='summary-item'><div class='summary-label'>Files quarantined</div>"
                f"<div class='summary-value'>{quarantined_count}"
                f"{' (deleted: ' + str(deleted_count) + ')' if deleted_count else ''}</div></div>",
                "<div class='summary-item'><div class='summary-label'>Metadata operations</div>"
                f"<div class='summary-value'>{metadata_success} ok / {metadata_failed} failed / {metadata_skipped} skipped</div></div>",
                "</div>",
                "<div class='muted' style='margin-top:10px;'>",
                f"Plan signature: <code>{plan_signature or computed_signature}</code><br/>",
                f"Reports directory: <code>{reports_dir}</code>",
                "</div>",
                "</div>",
                "<h2>Duplicate Groups</h2>",
            ]

            for gid in sorted([g for g in actions_by_group.keys() if g != "__general__"]):
                grp = group_lookup.get(gid)
                winner_path = getattr(grp, "winner_path", "Unknown winner")
                group_actions = actions_by_group.get(gid, [])
                state = _group_state(group_actions)
                badges = _action_badges(group_actions)
                badge_html = " ".join([f"<span class='badge'>{badge}</span>" for badge in badges])
                html_lines.append("<details class='group'>")
                html_lines.append(
                    "<summary>"
                    "<div class='group-summary'>"
                    f"<span class='group-title'>Group {gid}</span>"
                    f"<span class='muted'>• Winner: {_basename(winner_path)}</span>"
                    f"<span class='badge status-{state}'>"
                    f"{state}</span>"
                    f"{badge_html}"
                    "</div>"
                    "</summary>"
                )
                html_lines.append("<div class='context'>")
                html_lines.append(f"<div class='muted'>Winner path: <code>{winner_path}</code></div>")
                html_lines.append("<table>")
                html_lines.append(
                    "<tr><th>Operation</th><th>Target</th><th>Status</th><th>Notes</th></tr>"
                )
                for act in group_actions:
                    meta_notes = _format_metadata_notes(act.metadata)
                    notes = act.detail
                    if meta_notes:
                        notes = f"{notes}<div class='meta'>{meta_notes}</div>"
                    html_lines.append(
                        "<tr>"
                        f"<td><strong>{act.step}</strong></td>"
                        f"<td><span title='{act.target}'>{_basename(act.target)}</span></td>"
                        f"<td class='status-{act.status}'>{act.status}</td>"
                        f"<td>{notes}</td>"
                        "</tr>"
                    )
                html_lines.append("</table></div></details>")

            if actions_by_group.get("__general__"):
                html_lines.append("<details class='group'>")
                html_lines.append("<summary><div class='group-summary'>General Actions</div></summary>")
                html_lines.append("<div class='context'><table>")
                html_lines.append(
                    "<tr><th>Operation</th><th>Target</th><th>Status</th><th>Notes</th></tr>"
                )
                for act in actions_by_group["__general__"]:
                    meta_notes = _format_metadata_notes(act.metadata)
                    notes = act.detail
                    if meta_notes:
                        notes = f"{notes}<div class='meta'>{meta_notes}</div>"
                    html_lines.append(
                        "<tr>"
                        f"<td><strong>{act.step}</strong></td>"
                        f"<td><span title='{act.target}'>{_basename(act.target)}</span></td>"
                        f"<td class='status-{act.status}'>{act.status}</td>"
                        f"<td>{notes}</td>"
                        "</tr>"
                    )
                html_lines.append("</table></div></details>")

            if playlist_results:
                html_lines.append("<details class='group'>")
                html_lines.append("<summary><div class='group-summary'>Playlist Changes</div></summary>")
                html_lines.append("<div class='context'><table>")
                html_lines.append(
                    "<tr><th>Playlist</th><th>Status</th><th>Details</th></tr>"
                )
                for res in playlist_results:
                    label = res.playlist
                    if res.original_playlist:
                        label = f"{res.original_playlist} (validated copy: {res.playlist})"
                    html_lines.append(
                        "<tr>"
                        f"<td><code>{label}</code></td>"
                        f"<td class='status-{res.status}'>{res.status}</td>"
                        f"<td>replaced {res.replaced_entries}; backup: <code>{res.backup_path}</code></td>"
                        "</tr>"
                    )
                html_lines.append("</table></div></details>")

            if quarantine_index:
                html_lines.append("<details class='group'>")
                html_lines.append("<summary><div class='group-summary'>Quarantined Files</div></summary>")
                html_lines.append("<div class='context'><table>")
                html_lines.append("<tr><th>Original</th><th>Destination</th></tr>")
                for loser, dest in quarantine_index.items():
                    html_lines.append(
                        f"<tr><td><code>{loser}</code></td><td><code>{dest}</code></td></tr>"
                    )
                html_lines.append("</table></div></details>")
            html_lines.append("</body></html>")
            _atomic_write_text(html_report_path, "\n".join(html_lines))
            if not os.path.exists(ensure_long_path(html_report_path)):
                log(f"Report write failed: HTML report not found at {html_report_path}")
        except Exception as exc:
            log(f"Report write failed: unable to create HTML report at {html_report_path} ({exc})")

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
