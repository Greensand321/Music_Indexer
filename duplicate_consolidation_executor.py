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

import base64
import datetime
import hashlib
import html
import importlib.util
import json
import platform
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


@dataclass(frozen=True)
class PlanLoadResult:
    plan: ConsolidationPlan
    source: str
    path: str | None
    input_type: str


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
    quarantine_flatten: bool = False


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
    os.makedirs(base, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    for attempt in range(1000):
        suffix = f"_{attempt}" if attempt else ""
        path = os.path.join(base, f"consolidation_run_{ts}{suffix}")
        try:
            os.makedirs(path)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("Unable to create unique consolidation report directory.")


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
            try:
                with open(candidate, "rb") as handle:
                    return handle.read()
            except OSError:
                continue
    if MutagenFile is None:
        return None
    audio = None
    try:
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
    except Exception:
        return None
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
    if MutagenFile is None:
        return True, "Mutagen unavailable; metadata not written."
    audio = MutagenFile(ensure_long_path(target), easy=True)
    try:
        if audio is None:
            return True, "Unsupported audio format; metadata not written."
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


def _quarantine_path(path: str, quarantine_root: str, library_root: str, *, flatten: bool) -> str:
    if flatten:
        os.makedirs(quarantine_root, exist_ok=True)
        return os.path.join(quarantine_root, os.path.basename(path))
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
) -> PlanLoadResult:
    if isinstance(plan_input, ConsolidationPlan):
        return PlanLoadResult(plan=plan_input, source="in-memory plan", path=None, input_type="in-memory")
    payload: object = plan_input
    source = "plan payload"
    plan_path: str | None = None
    input_type = "mapping"
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
            plan_path = plan_input
            input_type = "file"
        else:
            try:
                payload = json.loads(plan_input)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Plan JSON could not be decoded: {exc}") from exc
            source = "json payload"
            input_type = "json string"
    if isinstance(payload, Mapping):
        if "plan" in payload:
            inner = payload.get("plan")
            if not isinstance(inner, Mapping):
                raise ValueError("Plan payload has a non-mapping 'plan' field.")
            payload = inner
            source = f"{source} (wrapped)"
        return PlanLoadResult(
            plan=consolidation_plan_from_dict(payload),
            source=source,
            path=plan_path,
            input_type=input_type,
        )
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
            plan_result = _coerce_consolidation_plan(plan_input, log)
            plan = plan_result.plan
            plan_source = plan_result.source
            plan_path = plan_result.path
            plan_input_type = plan_result.input_type
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
            generated_at_iso = plan.generated_at.isoformat()
            snapshot_count = len(plan.source_snapshot)
            canonical_snapshot = {k: dict(v) for k, v in plan.source_snapshot.items()}
            snapshot_id = hashlib.sha256(
                json.dumps(canonical_snapshot, sort_keys=True).encode("utf-8")
            ).hexdigest()
            drift_examples: List[str] = []
            for path, expected in plan.source_snapshot.items():
                if len(drift_examples) >= 5:
                    break
                actual = _capture_library_state(path)
                delta = _detect_state_drift(expected, actual)
                if delta:
                    drift_examples.append(f"{path}: {delta}")
            plan_path_mtime = None
            plan_path_size = None
            if plan_path:
                try:
                    plan_path_mtime = datetime.datetime.fromtimestamp(
                        os.path.getmtime(plan_path), datetime.timezone.utc
                    ).isoformat()
                    plan_path_size = os.path.getsize(plan_path)
                except OSError:
                    plan_path_mtime = None
                    plan_path_size = None
            log(
                "Plan signature mismatch summary: "
                + json.dumps(
                    {
                        "plan_path": plan_path,
                        "plan_source": plan_source,
                        "plan_input_type": plan_input_type,
                        "plan_path_mtime": plan_path_mtime,
                        "plan_path_size": plan_path_size,
                        "plan_signature": plan.plan_signature,
                        "computed_signature": computed_signature,
                        "generated_at": generated_at_iso,
                        "snapshot_id": snapshot_id,
                        "snapshot_count": snapshot_count,
                        "group_count": len(plan.groups),
                        "drift_examples": drift_examples,
                    },
                    sort_keys=True,
                )
            )
            success = False
            _record(
                "preflight",
                "execution",
                "blocked",
                "Plan signature mismatch — preview plan is stale or not the one being executed. Rerun Preview.",
                expected_signature=plan.plan_signature,
                computed_signature=computed_signature,
                plan_path=plan_path or "unknown",
                plan_source=plan_source,
                plan_input_type=plan_input_type,
                plan_path_mtime=plan_path_mtime,
                plan_path_size=plan_path_size,
                generated_at=generated_at_iso,
                snapshot_id=snapshot_id,
                snapshot_count=snapshot_count,
                group_count=len(plan.groups),
                drift_examples=drift_examples,
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
                    f"Dry-run execute enabled; artwork not applied (source: {art.source}).",
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
                    f"{detail or art.reason} (source: {art.source})",
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
                dest = _quarantine_path(
                    loser,
                    quarantine_dir,
                    config.library_root,
                    flatten=config.quarantine_flatten,
                )
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
        consolidated_report_path = os.path.join(reports_dir, "execution_report.json")
        html_report_path = os.path.join(reports_dir, EXECUTION_REPORT_FILENAME)

        plan_signature = plan.plan_signature if plan else None
        source_snapshot = plan.source_snapshot if plan else {}
        consolidated_payload = {
            "success": success,
            "actions": [a.to_dict() for a in actions],
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "plan_signature": plan_signature or computed_signature,
            "computed_signature": computed_signature,
            "source_snapshot": {k: dict(v) for k, v in source_snapshot.items()},
            "plan": plan.to_dict() if plan else None,
            "playlists": [p.to_dict() for p in playlist_results],
            "quarantine_index": dict(quarantine_index),
            "metadata_plans": {path: dict(tags) for path, tags in planned_tags.items()},
            "run_paths": {
                "reports_dir": reports_dir,
                "backups_dir": backups_dir,
            },
        }

        try:
            _atomic_write_json(consolidated_report_path, consolidated_payload)
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

            def _album_art_src(
                group: object,
                track_path: str,
                include_group_chosen: bool = True,
            ) -> str | None:
                candidates: List[str] = []
                if include_group_chosen and group is not None:
                    chosen = getattr(group, "chosen_artwork_source", None)
                    if isinstance(chosen, Mapping):
                        chosen_path = chosen.get("path")
                        if isinstance(chosen_path, str) and chosen_path:
                            candidates.append(chosen_path)
                if track_path:
                    candidates.append(track_path)
                for path in candidates:
                    payload = _extract_artwork_bytes(path)
                    if payload:
                        return _image_data_uri(payload)
                return None

            def _image_data_uri(payload: bytes) -> str:
                mime = _image_mime(payload)
                encoded = base64.b64encode(payload).decode("ascii")
                return f"data:{mime};base64,{encoded}"

            def _image_mime(payload: bytes) -> str:
                if payload.startswith(b"\x89PNG\r\n\x1a\n"):
                    return "image/png"
                if payload.startswith(b"\xff\xd8"):
                    return "image/jpeg"
                if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
                    return "image/webp"
                return "image/jpeg"

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
            generated_at_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            host_name = platform.node() or "unknown-host"

            def _status_badge_class(status: str) -> str:
                if status == "success":
                    return "ok"
                if status == "failed":
                    return "fail"
                return "warn"

            def _format_notes(detail: str, meta_notes: str) -> str:
                detail_html = html.escape(detail)
                if meta_notes:
                    meta_html = html.escape(meta_notes)
                    return f"{detail_html}<div class='tiny muted'>{meta_html}</div>"
                return detail_html

            def _group_type_hint(group_actions: List[ExecutionAction]) -> str:
                if not group_actions:
                    return "unknown"
                if all(act.step == "metadata" for act in group_actions):
                    return "metadata-only"
                return "mixed"

            def _group_quarantine_count(group_actions: List[ExecutionAction]) -> int:
                return sum(
                    1
                    for act in group_actions
                    if act.step == "loser_cleanup"
                    and act.metadata.get("disposition") == "quarantine"
                )

            def _group_has_quarantine(group_actions: List[ExecutionAction]) -> bool:
                return _group_quarantine_count(group_actions) > 0

            def _group_has_failure(group_actions: List[ExecutionAction]) -> bool:
                return any(
                    act.status in {"failed", "blocked", "cancelled"} for act in group_actions
                )

            def _run_status() -> tuple[str, str, str]:
                status = "success" if success else "failed"
                status_class = "ok" if success else "fail"
                emoji = "✅" if success else "❌"
                return status, status_class, f"{emoji} {status}"

            html_lines = [
                "<!doctype html>",
                "<html lang='en'>",
                "<head>",
                "<meta charset='utf-8' />",
                "<meta name='viewport' content='width=device-width,initial-scale=1' />",
                "<title>Execution Report</title>",
                "<style>",
                ":root{",
                "--bg:#ffffff;",
                "--text:#111;",
                "--muted:#666;",
                "--border:#e6e6e6;",
                "--card:#fafafa;",
                "--good:#1b7a1b;",
                "--bad:#a30000;",
                "--warn:#8a5b00;",
                "--chip:#f2f2f2;",
                "--mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;",
                "--sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, \"Apple Color Emoji\",\"Segoe UI Emoji\";",
                "}",
                "* { box-sizing: border-box; }",
                "body{",
                "margin: 18px;",
                "font-family: var(--sans);",
                "background: var(--bg);",
                "color: var(--text);",
                "line-height: 1.35;",
                "}",
                "header{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:12px 16px;",
                "align-items:baseline;",
                "justify-content:space-between;",
                "margin-bottom: 12px;",
                "}",
                "h1{",
                "font-size: 20px;",
                "margin: 0;",
                "letter-spacing: .2px;",
                "}",
                ".row{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:10px;",
                "align-items:center;",
                "}",
                ".card{",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "padding: 12px;",
                "background: var(--card);",
                "}",
                ".context-card{",
                "display:flex;",
                "gap:12px;",
                "align-items:flex-start;",
                "}",
                ".context-card-art{",
                "flex: 0 0 auto;",
                "}",
                ".context-card-details{",
                "flex: 1;",
                "}",
                ".card-stack{",
                "display:grid;",
                "gap:10px;",
                "}",
                ".summary{",
                "display:grid;",
                "grid-template-columns: 1fr;",
                "gap: 10px;",
                "margin-bottom: 14px;",
                "}",
                "@media (min-width: 900px){",
                ".summary{ grid-template-columns: 2fr 1fr; }",
                "}",
                ".kv{",
                "display:grid;",
                "grid-template-columns: 160px 1fr;",
                "gap:6px 10px;",
                "align-items:start;",
                "}",
                ".k{ color: var(--muted); font-size: 12px; }",
                ".v{ font-size: 13px; }",
                ".mono{ font-family: var(--mono); font-size: 12px; }",
                ".path{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "word-break: break-all;",
                "}",
                ".status{",
                "font-weight: 600;",
                "display:inline-flex;",
                "align-items:center;",
                "gap:6px;",
                "}",
                ".ok{ color: var(--good); }",
                ".fail{ color: var(--bad); }",
                ".warn{ color: var(--warn); }",
                ".pill{",
                "display:inline-flex;",
                "align-items:center;",
                "gap:8px;",
                "padding: 6px 10px;",
                "border-radius: 999px;",
                "background: var(--chip);",
                "border: 1px solid var(--border);",
                "font-size: 12px;",
                "white-space: nowrap;",
                "}",
                ".pill strong{ font-weight: 700; }",
                ".controls{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:10px;",
                "align-items:center;",
                "justify-content:space-between;",
                "margin: 14px 0 10px;",
                "}",
                ".controls .left, .controls .right{",
                "display:flex; flex-wrap:wrap; gap:10px; align-items:center;",
                "}",
                "input[type='search']{",
                "padding: 9px 10px;",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "width: min(520px, 92vw);",
                "font-size: 13px;",
                "background: #fff;",
                "}",
                "select{",
                "padding: 9px 10px;",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "font-size: 13px;",
                "background: #fff;",
                "}",
                "button.copy{",
                "border:1px solid var(--border);",
                "background:#fff;",
                "border-radius: 10px;",
                "padding: 8px 10px;",
                "font-size: 12px;",
                "cursor:pointer;",
                "}",
                "button.copy:active{ transform: translateY(1px); }",
                "details.group{",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "background: #fff;",
                "margin-bottom: 10px;",
                "overflow:hidden;",
                "}",
                "details.group[open]{ box-shadow: 0 1px 0 rgba(0,0,0,.03); }",
                "summary.group-summary{",
                "list-style:none;",
                "cursor:pointer;",
                "padding: 12px 12px;",
                "display:grid;",
                "grid-template-columns: auto 1fr;",
                "grid-template-rows: auto auto;",
                "column-gap: 12px;",
                "row-gap: 6px;",
                "background: #fff;",
                "}",
                "summary.group-summary::-webkit-details-marker{ display:none; }",
                ".album-art{",
                "width: 42px;",
                "height: 42px;",
                "display: inline-block;",
                "grid-column: 1;",
                "grid-row: 1 / span 2;",
                "align-self: center;",
                "border-radius: 8px;",
                "border: 1px solid var(--border);",
                "position: relative;",
                "overflow: hidden;",
                "background: #f7f7f7;",
                "}",
                ".album-art.album-art-thumb{",
                "width: 64px;",
                "height: 64px;",
                "border-radius: 6px;",
                "}",
                ".album-art::after{",
                "content: \"\";",
                "position: absolute;",
                "inset: 0;",
                "pointer-events: none;",
                "box-shadow: inset 0 0 0 1px rgba(0,0,0,.06), inset 0 0 10px rgba(0,0,0,.20);",
                "}",
                ".album-art img{",
                "width: 100%;",
                "height: 100%;",
                "object-fit: cover;",
                "display: block;",
                "}",
                ".group-top{",
                "grid-column: 2;",
                "grid-row: 1;",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:10px 12px;",
                "align-items:center;",
                "justify-content:space-between;",
                "}",
                "summary.group-summary .tiny{",
                "grid-column: 2;",
                "grid-row: 2;",
                "}",
                ".group-title{",
                "display:flex;",
                "align-items:baseline;",
                "gap:10px;",
                "min-width: 240px;",
                "}",
                ".gid{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "color: var(--muted);",
                "}",
                ".winner-short{",
                "font-weight: 650;",
                "font-size: 13px;",
                "max-width: 74ch;",
                "overflow:hidden;",
                "text-overflow: ellipsis;",
                "white-space: nowrap;",
                "}",
                ".chips{",
                "display:flex;",
                "flex-wrap:wrap;",
                "gap:8px;",
                "justify-content:flex-end;",
                "}",
                ".group-body{",
                "border-top: 1px solid var(--border);",
                "background: var(--card);",
                "padding: 12px;",
                "display:grid;",
                "grid-template-columns: 1fr;",
                "gap: 10px;",
                "}",
                "@media (min-width: 980px){",
                ".group-body{ grid-template-columns: 1fr 1.2fr; }",
                "}",
                ".section h3{",
                "font-size: 12px;",
                "margin: 0 0 8px;",
                "color: var(--muted);",
                "text-transform: uppercase;",
                "letter-spacing: .08em;",
                "}",
                "table.ops{",
                "width:100%;",
                "border-collapse: collapse;",
                "background:#fff;",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "overflow:hidden;",
                "}",
                "table.ops th, table.ops td{",
                "border-bottom:1px solid var(--border);",
                "padding: 9px 10px;",
                "font-size: 12px;",
                "vertical-align: top;",
                "}",
                "table.ops th{",
                "text-align:left;",
                "color: var(--muted);",
                "background: #fcfcfc;",
                "font-weight: 650;",
                "}",
                "table.ops tr:last-child td{ border-bottom: none; }",
                ".op-key{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "color:#222;",
                "white-space:nowrap;",
                "}",
                ".note{",
                "color: var(--muted);",
                "}",
                ".badge{",
                "display:inline-flex;",
                "align-items:center;",
                "padding: 3px 8px;",
                "border-radius: 999px;",
                "border:1px solid var(--border);",
                "font-size: 12px;",
                "font-weight: 650;",
                "background: #fff;",
                "white-space: nowrap;",
                "}",
                ".badge.ok{ border-color: rgba(27,122,27,.25); }",
                ".badge.fail{ border-color: rgba(163,0,0,.25); }",
                "details.quarantine{",
                "border:1px solid var(--border);",
                "border-radius: 10px;",
                "background:#fff;",
                "margin-top: 14px;",
                "overflow:hidden;",
                "}",
                "summary.quarantine-summary{",
                "cursor:pointer;",
                "padding: 12px;",
                "list-style:none;",
                "}",
                "summary.quarantine-summary::-webkit-details-marker{ display:none; }",
                "ul.q-list{",
                "margin:0;",
                "padding: 0 12px 12px 28px;",
                "color:#222;",
                "}",
                "ul.q-list li{",
                "font-family: var(--mono);",
                "font-size: 12px;",
                "word-break: break-all;",
                "margin: 6px 0;",
                "}",
                ".muted{ color: var(--muted); }",
                ".tiny{ font-size: 12px; }",
                ".hidden{ display:none !important; }",
                "</style>",
                "</head>",
                "<body>",
                "<header>",
                "<h1>Execution Report</h1>",
                f"<div class='status {_run_status()[1]}' id='runStatus' data-status='{_run_status()[0]}'>"
                f"{_run_status()[2]}</div>",
                "</header>",
                "<section class='summary'>",
                "<div class='card'>",
                "<div class='row' style='justify-content:space-between;'>",
                "<div class='row' style='gap:8px;'>",
                "<span class='pill'><strong>Groups</strong> <span id='statGroups'>0</span></span>",
                "<span class='pill'><strong>Winners</strong> <span id='statWinners'>0</span></span>",
                "<span class='pill'><strong>Quarantined</strong> <span id='statQuarantined'>0</span></span>",
                "<span class='pill'><strong>Ops</strong> <span id='statOpsOk'>0</span>/<span id='statOpsTotal'>0</span> ok</span>",
                "</div>",
                "<div class='row'>",
                "<button class='copy' data-copy='#planSignature'>Copy plan signature</button>",
                "<button class='copy' data-copy='#reportsDir'>Copy reports dir</button>",
                "</div>",
                "</div>",
                "<div style='height:10px'></div>",
                "<div class='kv'>",
                "<div class='k'>Plan signature</div>",
                f"<div class='v mono' id='planSignature'>{html.escape(plan_signature or computed_signature)}</div>",
                "<div class='k'>Reports directory</div>",
                f"<div class='v path' id='reportsDir'>{html.escape(reports_dir)}</div>",
                "</div>",
                "</div>",
                "<div class='card'>",
                "<div class='kv'>",
                "<div class='k'>Generated</div>",
                f"<div class='v tiny'>{html.escape(generated_at_iso)}</div>",
                "<div class='k'>Host</div>",
                f"<div class='v tiny'>{html.escape(host_name)}</div>",
                "<div class='k'>Notes</div>",
                "<div class='v tiny muted'>",
                "Groups are collapsed by default. Full paths and verbose messages are in “Context”.",
                "</div>",
                "</div>",
                "</div>",
                "</section>",
                "<section class='controls'>",
                "<div class='left'>",
                "<input id='search' type='search' placeholder='Search groups, winner filename, paths, notes…' />",
                "<select id='filter'>",
                "<option value='all'>All groups</option>",
                "<option value='has-quarantine'>Has quarantine</option>",
                "<option value='metadata-only'>Metadata only</option>",
                "<option value='failed'>Any failures</option>",
                "</select>",
                "</div>",
                "<div class='right'>",
                "<span class='tiny muted' id='visibleCount'>0 visible</span>",
                "<button class='copy' id='expandAll'>Expand all</button>",
                "<button class='copy' id='collapseAll'>Collapse all</button>",
                "</div>",
                "</section>",
                "<main id='groups'>",
            ]

            for gid in sorted([g for g in actions_by_group.keys() if g != "__general__"]):
                grp = group_lookup.get(gid)
                winner_path = getattr(grp, "winner_path", "Unknown winner")
                group_actions = actions_by_group.get(gid, [])
                state = _group_state(group_actions)
                badges = _action_badges(group_actions)
                group_quarantine_count = _group_quarantine_count(group_actions)
                group_type_hint = _group_type_hint(group_actions)
                group_has_quarantine = _group_has_quarantine(group_actions)
                group_has_failure = _group_has_failure(group_actions)
                search_tokens = [gid, winner_path, _basename(winner_path), state]
                search_tokens.extend(badges)
                for act in group_actions:
                    search_tokens.extend([act.step, act.target, act.status, act.detail])
                    meta_notes = _format_metadata_notes(act.metadata)
                    if meta_notes:
                        search_tokens.append(meta_notes)
                search_text = " ".join(str(token) for token in search_tokens if token)
                summary_line = (
                    f"Status: {state}. " + "; ".join(badges) if badges else f"Status: {state}."
                )
                html_lines.append(
                    "<details class='group' "
                    f"data-group-id='{html.escape(gid)}' "
                    f"data-has-quarantine='{str(group_has_quarantine).lower()}' "
                    f"data-has-failure='{str(group_has_failure).lower()}' "
                    f"data-type='{html.escape(group_type_hint)}' "
                    f"data-search='{html.escape(search_text.lower())}' "
                    f"data-quarantine-count='{group_quarantine_count}'>"
                )
                album_art_src = _album_art_src(grp, winner_path)
                html_lines.append("<summary class='group-summary'>")
                html_lines.append("<span class='album-art' title='Album Art'>")
                if album_art_src:
                    html_lines.append(f"<img src='{album_art_src}' alt='' />")
                html_lines.append("</span>")
                html_lines.append("<div class='group-top'>")
                html_lines.append("<div class='group-title'>")
                html_lines.append(f"<span class='gid'>Group {html.escape(gid)}</span>")
                html_lines.append(
                    "<span class='winner-short' "
                    f"title='{html.escape(winner_path)}'>{html.escape(_basename(winner_path))}</span>"
                )
                html_lines.append("</div>")
                html_lines.append("<div class='chips'>")
                html_lines.append(
                    f"<span class='badge {_status_badge_class(state)}' data-status='{html.escape(state)}'>"
                    f"{html.escape(state)}</span>"
                )
                for badge in badges:
                    html_lines.append(f"<span class='pill'>{html.escape(badge)}</span>")
                html_lines.append("</div>")
                html_lines.append("</div>")
                html_lines.append(f"<div class='tiny muted'>{html.escape(summary_line)}</div>")
                html_lines.append("</summary>")
                html_lines.append("<div class='group-body'>")
                html_lines.append("<div class='section'>")
                html_lines.append("<h3>Context</h3>")
                cleanup_actions = [act for act in group_actions if act.step == "loser_cleanup"]
                winner_art_src = _album_art_src(grp, winner_path)
                html_lines.append("<div class='card-stack'>")
                html_lines.append("<div class='card context-card' style='background:#fff;'>")
                html_lines.append("<div class='context-card-art'>")
                html_lines.append("<span class='album-art album-art-thumb' title='Album Art'>")
                if winner_art_src:
                    html_lines.append(f"<img src='{winner_art_src}' alt='' />")
                html_lines.append("</span>")
                html_lines.append("</div>")
                html_lines.append("<div class='kv context-card-details'>")
                html_lines.append("<div class='k'>Kept file (winner)</div>")
                html_lines.append(f"<div class='v path'>{html.escape(winner_path)}</div>")
                html_lines.append("<div class='k'>Group ID</div>")
                html_lines.append(f"<div class='v mono'>{html.escape(gid)}</div>")
                html_lines.append("<div class='k'>Debug</div>")
                html_lines.append(
                    "<div class='v tiny muted'>Shown for traceability; safe to ignore for normal review.</div>"
                )
                html_lines.append("</div>")
                html_lines.append("</div>")
                if cleanup_actions:
                    loser_entries = cleanup_actions
                else:
                    loser_entries = [None]
                for loser_action in loser_entries:
                    loser_path = loser_action.target if loser_action else ""
                    loser_disposition = ""
                    loser_reason = ""
                    if loser_action:
                        raw_disposition = loser_action.metadata.get("disposition")
                        loser_disposition = str(raw_disposition) if raw_disposition else ""
                        loser_reason = loser_action.detail or ""
                    loser_art_src = (
                        _album_art_src(grp, loser_path, include_group_chosen=False)
                        if loser_path
                        else None
                    )
                    html_lines.append("<div class='card context-card' style='background:#fff;'>")
                    html_lines.append("<div class='context-card-art'>")
                    html_lines.append("<span class='album-art album-art-thumb' title='Album Art'>")
                    if loser_art_src:
                        html_lines.append(f"<img src='{loser_art_src}' alt='' />")
                    html_lines.append("</span>")
                    html_lines.append("</div>")
                    html_lines.append("<div class='kv context-card-details'>")
                    html_lines.append("<div class='k'>Loser file</div>")
                    if loser_path:
                        html_lines.append(
                            f"<div class='v path'>{html.escape(loser_path)}</div>"
                        )
                    else:
                        html_lines.append("<div class='v path'>—</div>")
                    html_lines.append("<div class='k'>Disposition</div>")
                    if loser_disposition:
                        html_lines.append(
                            f"<div class='v'>{html.escape(loser_disposition)}</div>"
                        )
                    else:
                        html_lines.append("<div class='v'>—</div>")
                    if loser_reason:
                        html_lines.append("<div class='k'>Reason</div>")
                        html_lines.append(
                            f"<div class='v tiny muted'>{html.escape(loser_reason)}</div>"
                        )
                    html_lines.append("</div>")
                    html_lines.append("</div>")
                html_lines.append("</div>")
                html_lines.append("</div>")
                html_lines.append("<div class='section'>")
                html_lines.append("<h3>Operations</h3>")
                html_lines.append("<table class='ops'>")
                html_lines.append("<thead>")
                html_lines.append("<tr>")
                html_lines.append("<th style='width: 160px;'>Operation</th>")
                html_lines.append("<th>Target</th>")
                html_lines.append("<th style='width: 110px;'>Status</th>")
                html_lines.append("<th>Notes</th>")
                html_lines.append("</tr>")
                html_lines.append("</thead>")
                html_lines.append("<tbody>")
                for act in group_actions:
                    meta_notes = _format_metadata_notes(act.metadata)
                    notes = _format_notes(act.detail, meta_notes)
                    status_class = _status_badge_class(act.status)
                    html_lines.append(
                        "<tr>"
                        f"<td class='op-key'>{html.escape(act.step)}</td>"
                        f"<td class='path' title='{html.escape(act.target)}'>"
                        f"{html.escape(_basename(act.target))}</td>"
                        f"<td><span class='badge {status_class}'>{html.escape(act.status)}</span></td>"
                        f"<td class='note'>{notes}</td>"
                        "</tr>"
                    )
                html_lines.append("</tbody>")
                html_lines.append("</table>")
                html_lines.append("</div>")
                html_lines.append("</div>")
                html_lines.append("</details>")

            html_lines.append("</main>")
            html_lines.append("<details class='quarantine' id='quarantineIndex'>")
            html_lines.append("<summary class='quarantine-summary'>")
            html_lines.append("<div class='row' style='justify-content:space-between;'>")
            html_lines.append(
                "<div><strong>Quarantined Files</strong> <span class='muted'>(<span id='qCount'>0</span>)</span></div>"
            )
            html_lines.append("<div class='muted tiny'>Collapsed by default</div>")
            html_lines.append("</div>")
            html_lines.append("</summary>")
            html_lines.append("<ul class='q-list' id='qList'>")
            for loser, dest in quarantine_index.items():
                html_lines.append(
                    f"<li>{html.escape(loser)} → {html.escape(dest)}</li>"
                )
            html_lines.append("</ul>")
            html_lines.append("</details>")

            missing_sections: List[str] = []
            missing_sections.append(
                f"Metadata operations: {metadata_success} ok / {metadata_failed} failed / {metadata_skipped} skipped"
            )
            if deleted_count:
                missing_sections.append(
                    f"Files quarantined: {quarantined_count} (deleted: {deleted_count})"
                )
            else:
                missing_sections.append(f"Files quarantined: {quarantined_count}")

            general_actions = actions_by_group.get("__general__", [])
            playlist_results = playlist_results or []
            if general_actions or playlist_results:
                missing_sections.append("")
            if general_actions:
                missing_sections.append("General Actions")
                missing_sections.append("<table class='ops'>")
                missing_sections.append(
                    "<thead><tr><th>Operation</th><th>Target</th><th>Status</th><th>Notes</th></tr></thead>"
                )
                missing_sections.append("<tbody>")
                for act in general_actions:
                    meta_notes = _format_metadata_notes(act.metadata)
                    notes = _format_notes(act.detail, meta_notes)
                    status_class = _status_badge_class(act.status)
                    missing_sections.append(
                        "<tr>"
                        f"<td class='op-key'>{html.escape(act.step)}</td>"
                        f"<td class='path' title='{html.escape(act.target)}'>"
                        f"{html.escape(_basename(act.target))}</td>"
                        f"<td><span class='badge {status_class}'>{html.escape(act.status)}</span></td>"
                        f"<td class='note'>{notes}</td>"
                        "</tr>"
                    )
                missing_sections.append("</tbody></table>")
            if playlist_results:
                missing_sections.append("Playlist Changes")
                missing_sections.append("<table class='ops'>")
                missing_sections.append(
                    "<thead><tr><th>Playlist</th><th>Status</th><th>Details</th></tr></thead>"
                )
                missing_sections.append("<tbody>")
                for res in playlist_results:
                    label = res.playlist
                    if res.original_playlist:
                        label = f"{res.original_playlist} (validated copy: {res.playlist})"
                    status_class = _status_badge_class(res.status)
                    missing_sections.append(
                        "<tr>"
                        f"<td class='path'>{html.escape(label)}</td>"
                        f"<td><span class='badge {status_class}'>{html.escape(res.status)}</span></td>"
                        f"<td class='note'>replaced {res.replaced_entries}; backup: {html.escape(res.backup_path)}</td>"
                        "</tr>"
                    )
                missing_sections.append("</tbody></table>")

            if missing_sections:
                html_lines.append(
                    "<div style='margin-top:14px;'>"
                    "<strong>NEED TO INCORPORATE AT LATER DATE</strong>"
                    "</div>"
                )
                for section in missing_sections:
                    if not section:
                        html_lines.append("<div style='height:10px'></div>")
                        continue
                    html_lines.append(f"<div class='tiny'>{section}</div>")
            html_lines.append(
                "<script>"
                "(function(){"
                "const $ = (s, r=document) => r.querySelector(s);"
                "const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));"
                "const groupsEl = $('#groups');"
                "const groups = () => $$('#groups details.group');"
                "const searchEl = $('#search');"
                "const filterEl = $('#filter');"
                "const visibleCountEl = $('#visibleCount');"
                "const expandAllBtn = $('#expandAll');"
                "const collapseAllBtn = $('#collapseAll');"
                "function computeStats(){"
                "const g = groups();"
                "$('#statGroups').textContent = g.length.toString();"
                "$('#statWinners').textContent = g.length.toString();"
                "let qTotal = 0;"
                "for (const d of g){"
                "const chip = d.querySelector('.pill');"
                "const attr = d.getAttribute('data-quarantine-count');"
                "if (attr) qTotal += Number(attr) || 0;"
                "}"
                "$('#statQuarantined').textContent = qTotal.toString();"
                "let opsTotal = 0, opsOk = 0;"
                "for (const d of g){"
                "const rows = $$('table.ops tbody tr', d);"
                "opsTotal += rows.length;"
                "for (const r of rows){"
                "const badge = r.querySelector('.badge');"
                "if (badge && badge.textContent.trim().toLowerCase() === 'success') opsOk += 1;"
                "}"
                "}"
                "$('#statOpsTotal').textContent = opsTotal.toString();"
                "$('#statOpsOk').textContent = opsOk.toString();"
                "const qItems = $$('#qList li').filter(li => li.textContent.trim().length > 0);"
                "$('#qCount').textContent = qItems.length.toString();"
                "}"
                "function applyFilters(){"
                "const term = (searchEl.value || '').trim().toLowerCase();"
                "const mode = filterEl.value;"
                "let visible = 0;"
                "for (const d of groups()){"
                "const hay = (d.getAttribute('data-search') || '').toLowerCase();"
                "const matchesSearch = !term || hay.includes(term);"
                "const hasQuarantine = (d.getAttribute('data-has-quarantine') || 'false') === 'true';"
                "const hasFailure = (d.getAttribute('data-has-failure') || 'false') === 'true';"
                "const typeHint = (d.getAttribute('data-type') || '').toLowerCase();"
                "let matchesFilter = true;"
                "if (mode === 'has-quarantine') matchesFilter = hasQuarantine;"
                "if (mode === 'metadata-only') matchesFilter = typeHint === 'metadata-only';"
                "if (mode === 'failed') matchesFilter = hasFailure;"
                "const show = matchesSearch && matchesFilter;"
                "d.classList.toggle('hidden', !show);"
                "if (show) visible += 1;"
                "}"
                "visibleCountEl.textContent = `${visible} visible`;"
                "}"
                "for (const btn of $$('button.copy[data-copy]')){"
                "btn.addEventListener('click', async () => {"
                "const sel = btn.getAttribute('data-copy');"
                "const el = sel ? $(sel) : null;"
                "const text = el ? el.textContent.trim() : '';"
                "if (!text) return;"
                "try{"
                "await navigator.clipboard.writeText(text);"
                "const old = btn.textContent;"
                "btn.textContent = 'Copied';"
                "setTimeout(()=>btn.textContent=old, 900);"
                "}catch(e){"
                "const ta = document.createElement('textarea');"
                "ta.value = text;"
                "document.body.appendChild(ta);"
                "ta.select();"
                "document.execCommand('copy');"
                "ta.remove();"
                "}"
                "});"
                "}"
                "expandAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = true));"
                "collapseAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = false));"
                "searchEl?.addEventListener('input', applyFilters);"
                "filterEl?.addEventListener('change', applyFilters);"
                "computeStats();"
                "applyFilters();"
                "})();"
                "</script>"
            )
            html_lines.append("</body></html>")
            _atomic_write_text(html_report_path, "\n".join(html_lines))
            if not os.path.exists(ensure_long_path(html_report_path)):
                log(f"Report write failed: HTML report not found at {html_report_path}")
        except Exception as exc:
            log(f"Report write failed: unable to create HTML report at {html_report_path} ({exc})")

    report_paths = {
        "json_report": consolidated_report_path,
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
