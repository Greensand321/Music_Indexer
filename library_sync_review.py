"""Review-first Library Sync entry point and session models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import DEFAULT_FP_THRESHOLDS, load_config, save_config
from library_sync_review_report import DEFAULT_REPORT_VERSION

REVIEW_CONFIG_KEY = "library_sync_review"
REVIEW_FLAG_KEY = "use_library_sync_review"


class ScanState(str, Enum):
    """States tracked during a review scan lifecycle."""

    IDLE = "idle"
    READY = "ready"
    SCANNING_LIBRARY = "scanning_library"
    SCANNING_INCOMING = "scanning_incoming"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass
class ScanConfig:
    """Global and per-format fingerprint thresholds for a scan."""

    global_threshold: float = DEFAULT_FP_THRESHOLDS.get("default", 0.3)
    per_format_overrides: Dict[str, float] = field(default_factory=dict)
    preset_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_threshold": float(self.global_threshold),
            "per_format_overrides": dict(self.per_format_overrides),
            "preset_name": self.preset_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ScanConfig":
        data = data or {}
        overrides = data.get("per_format_overrides") or {}
        return cls(
            global_threshold=float(data.get("global_threshold", DEFAULT_FP_THRESHOLDS.get("default", 0.3))),
            per_format_overrides={str(k): float(v) for k, v in overrides.items()},
            preset_name=data.get("preset_name"),
        )


@dataclass
class ScanSession:
    """Persisted state for the review-first Library Sync experience."""

    library_root: str = ""
    incoming_root: str = ""
    scan_config: ScanConfig = field(default_factory=ScanConfig)
    scan_state: ScanState = ScanState.IDLE
    exported_report_version: int = DEFAULT_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "library_root": self.library_root,
            "incoming_root": self.incoming_root,
            "scan_config": self.scan_config.to_dict(),
            "scan_state": self.scan_state.value,
            "exported_report_version": int(self.exported_report_version),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ScanSession":
        data = data or {}
        state_value = data.get("scan_state") or ScanState.IDLE.value
        try:
            scan_state = ScanState(state_value)
        except ValueError:
            scan_state = ScanState.IDLE
        return cls(
            library_root=data.get("library_root", "") or "",
            incoming_root=data.get("incoming_root", "") or "",
            scan_config=ScanConfig.from_dict(data.get("scan_config")),
            scan_state=scan_state,
            exported_report_version=int(data.get("exported_report_version", DEFAULT_REPORT_VERSION)),
        )


def load_scan_session() -> ScanSession:
    """Load the persisted review scan session from config."""
    cfg = load_config()
    session = ScanSession.from_dict(cfg.get(REVIEW_CONFIG_KEY))
    cfg[REVIEW_CONFIG_KEY] = session.to_dict()
    save_config(cfg)
    return session


def save_scan_session(session: ScanSession) -> None:
    """Persist the review scan session to config."""
    cfg = load_config()
    cfg[REVIEW_CONFIG_KEY] = session.to_dict()
    save_config(cfg)


def is_review_enabled() -> bool:
    """Return whether the review-first entry point is enabled."""
    cfg = load_config()
    return bool(cfg.get(REVIEW_FLAG_KEY, False))


def set_review_enabled(enabled: bool) -> None:
    """Toggle the review-first entry point flag."""
    cfg = load_config()
    cfg[REVIEW_FLAG_KEY] = bool(enabled)
    save_config(cfg)


class ReportPreviewDialog(tk.Toplevel):
    """Simple dialog that summarizes report counts before export."""

    LABELS = {
        "new": "New",
        "collisions": "Collisions / Exact Matches",
        "low_confidence": "Low Confidence",
        "flagged_copy": "Flagged Copy",
        "flagged_replace": "Flagged Replace",
    }

    def __init__(self, master: tk.Misc | None, summary: dict[str, int]):
        super().__init__(master)
        self.title("Report Preview")
        self.resizable(False, False)
        self.result = False
        self.transient(master)
        self.grab_set()

        ttk.Label(self, text="Review counts before saving the export report:").pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        for key, label in self.LABELS.items():
            row = ttk.Frame(self)
            row.pack(fill="x", padx=10, pady=2)
            ttk.Label(row, text=f"{label}:").pack(side="left")
            ttk.Label(row, text=str(summary.get(key, 0)), font=("TkDefaultFont", 10, "bold")).pack(
                side="right"
            )

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=10, pady=(10, 10))
        ttk.Button(btn_row, text="Save Report", command=self._confirm).pack(side="left")
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._cancel)

    @classmethod
    def prompt(cls, master: tk.Misc | None, summary: dict[str, int]) -> bool:
        dialog = cls(master, summary)
        if master is not None:
            master.wait_window(dialog)
        else:
            dialog.wait_window()
        return bool(getattr(dialog, "result", False))

    def _confirm(self) -> None:
        self.result = True
        self.destroy()

    def _cancel(self) -> None:
        self.result = False
        self.destroy()


class LibrarySyncReviewWindow(tk.Toplevel):
    """Lightweight entry point for the redesign with persisted settings."""

    def __init__(self, master: tk.Widget | None = None):
        super().__init__(master)
        self.title("Library Sync (Review)")
        self.resizable(True, False)
        self.session = load_scan_session()

        self.library_var = tk.StringVar(value=self.session.library_root)
        self.incoming_var = tk.StringVar(value=self.session.incoming_root)
        self.global_threshold_var = tk.StringVar(value=str(self.session.scan_config.global_threshold))
        self.preset_var = tk.StringVar(value=self.session.scan_config.preset_name or "")
        self.report_version_var = tk.StringVar(value=str(self.session.exported_report_version))
        self.scan_state_var = tk.StringVar(value=self._describe_state(self.session.scan_state))

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 5}

        folder_frame = ttk.LabelFrame(self, text="Folders")
        folder_frame.pack(fill="x", **padding)

        self._add_browse_row(
            folder_frame, "Existing Library:", self.library_var, self._browse_library
        )
        self._add_browse_row(
            folder_frame, "Incoming Folder:", self.incoming_var, self._browse_incoming
        )

        scan_frame = ttk.LabelFrame(self, text="Scan Configuration")
        scan_frame.pack(fill="x", **padding)

        thr_row = ttk.Frame(scan_frame)
        thr_row.pack(fill="x", padx=5, pady=(5, 2))
        ttk.Label(thr_row, text="Global Threshold:").pack(side="left")
        ttk.Entry(thr_row, textvariable=self.global_threshold_var, width=10).pack(
            side="left", padx=(5, 0)
        )

        preset_row = ttk.Frame(scan_frame)
        preset_row.pack(fill="x", padx=5, pady=2)
        ttk.Label(preset_row, text="Preset Name:").pack(side="left")
        ttk.Entry(preset_row, textvariable=self.preset_var, width=20).pack(
            side="left", padx=(5, 0)
        )

        overrides_frame = ttk.Frame(scan_frame)
        overrides_frame.pack(fill="both", expand=True, padx=5, pady=(2, 5))
        ttk.Label(
            overrides_frame,
            text="Per-format overrides (ext=threshold per line):",
        ).pack(anchor="w")
        self.overrides_text = tk.Text(overrides_frame, height=4)
        self.overrides_text.pack(fill="x", expand=True, pady=(2, 0))
        self.overrides_text.insert("1.0", self._format_overrides())

        meta_frame = ttk.LabelFrame(self, text="Session")
        meta_frame.pack(fill="x", **padding)
        meta_row = ttk.Frame(meta_frame)
        meta_row.pack(fill="x", padx=5, pady=(5, 2))
        ttk.Label(meta_row, text="Scan State:").pack(side="left")
        ttk.Label(meta_row, textvariable=self.scan_state_var).pack(side="left", padx=(5, 0))

        version_row = ttk.Frame(meta_frame)
        version_row.pack(fill="x", padx=5, pady=(2, 5))
        ttk.Label(version_row, text="Exported Report Version:").pack(side="left")
        ttk.Entry(version_row, textvariable=self.report_version_var, width=6).pack(
            side="left", padx=(5, 0)
        )

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(0, 10))
        ttk.Button(btn_row, text="Save Session", command=self._persist_session).pack(
            side="left", padx=(10, 5)
        )
        ttk.Button(btn_row, text="Close", command=self._on_close).pack(side="right", padx=(5, 10))

    def _add_browse_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        command,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(5, 5))
        ttk.Button(row, text="Browse…", command=command).pack(side="left")

    def _browse_library(self) -> None:
        folder = filedialog.askdirectory(title="Select Library Root", initialdir=self.library_var.get() or None)
        if folder:
            self.library_var.set(folder)

    def _browse_incoming(self) -> None:
        folder = filedialog.askdirectory(title="Select Incoming Folder", initialdir=self.incoming_var.get() or None)
        if folder:
            self.incoming_var.set(folder)

    def _format_overrides(self) -> str:
        lines = []
        for ext, thr in self.session.scan_config.per_format_overrides.items():
            lines.append(f"{ext}={thr}")
        return "\n".join(lines)

    def _parse_overrides(self) -> Dict[str, float] | None:
        overrides: Dict[str, float] = {}
        raw = self.overrides_text.get("1.0", "end").strip()
        if not raw:
            return overrides
        for line in raw.splitlines():
            if "=" not in line:
                messagebox.showerror("Invalid Override", f"Expected 'ext=threshold' format, got: {line}")
                return None
            ext, value = line.split("=", 1)
            try:
                overrides[ext.strip()] = float(value.strip())
            except ValueError:
                messagebox.showerror("Invalid Threshold", f"Could not parse threshold for {ext.strip()}: {value}")
                return None
        return overrides

    def _persist_session(self) -> bool:
        overrides = self._parse_overrides()
        if overrides is None:
            return False
        try:
            global_thr = float(self.global_threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid Threshold", "Global threshold must be a number.")
            return False
        try:
            report_ver = int(self.report_version_var.get())
        except ValueError:
            messagebox.showerror("Invalid Report Version", "Exported report version must be an integer.")
            return False

        self.session = ScanSession(
            library_root=self.library_var.get().strip(),
            incoming_root=self.incoming_var.get().strip(),
            scan_config=ScanConfig(
                global_threshold=global_thr,
                per_format_overrides=overrides,
                preset_name=self.preset_var.get().strip() or None,
            ),
            scan_state=self.session.scan_state,
            exported_report_version=report_ver,
        )
        save_scan_session(self.session)
        self.scan_state_var.set(self._describe_state(self.session.scan_state))
        return True

    def _describe_state(self, state: ScanState) -> str:
        return state.value.replace("_", " ").title()

    def preview_report(self, summary: dict[str, int]) -> bool:
        """Open a modal preview dialog to confirm report export counts."""
        return ReportPreviewDialog.prompt(self, summary)

    def _on_close(self) -> None:
        if self._persist_session():
            self.destroy()
