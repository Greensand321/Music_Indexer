"""Legacy Tkinter configuration UI for metadata services.

This widget used to live in ``plugins/acoustid_plugin.py``, which forced a
module-level ``import tkinter`` into the plugin package. That violated the
project's "no Tkinter in backend modules" rule and made
``plugins.acoustid_plugin`` fail to import outright on any system without a
system Tkinter — which is what broke ``tests/test_musicbrainz_service.py`` in
those environments, since importing the module was enough to raise.

Only the UI moved; the ``AcoustIDService`` / ``MusicBrainzService`` lookup
classes stayed in the plugin module where they belong. This file sits at the
repository root alongside ``library_sync_review.py``, the project's other
legacy Tkinter panel.

Used only by the legacy Tkinter app (``main_gui.py``). The modern Qt app has
its own equivalent in ``gui/dialogs/settings_drawer.py``.
"""
import threading
import tkinter as tk
from tkinter import ttk

from config import load_config, save_config, SUPPORTED_SERVICES
from plugins.acoustid_plugin import AcoustIDService, MusicBrainzService
import tag_fixer


class MetadataServiceConfigFrame(tk.Frame):
    """Reusable configuration UI for metadata services."""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.cfg = load_config()
        self.service_var = tk.StringVar(value=self.cfg.get("metadata_service", "AcoustID"))
        self.api_var = tk.StringVar(value=self.cfg.get("metadata_api_key", tag_fixer.ACOUSTID_API_KEY))
        ua = self.cfg.get("musicbrainz_useragent", {})
        self.mb_app_var = tk.StringVar(value=ua.get("app", ""))
        self.mb_ver_var = tk.StringVar(value=ua.get("version", ""))
        self.mb_contact_var = tk.StringVar(value=ua.get("contact", ""))
        self.status_var = tk.StringVar()
        self.last_ok = False

        services = [s for s in SUPPORTED_SERVICES if s in ("AcoustID", "MusicBrainz")]
        tk.Label(self, text="Service:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.service_box = ttk.Combobox(self, textvariable=self.service_var, values=services, state="readonly")
        self.service_box.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.service_box.bind("<<ComboboxSelected>>", lambda _e: self._update_visible())

        self.api_lbl = tk.Label(self, text="API Key:")
        self.api_entry = ttk.Entry(self, textvariable=self.api_var, width=40)

        self.mb_frame = ttk.Frame(self)
        ttk.Label(self.mb_frame, text="App:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(self.mb_frame, textvariable=self.mb_app_var, width=30).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(self.mb_frame, text="Version:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(self.mb_frame, textvariable=self.mb_ver_var, width=20).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(self.mb_frame, text="Contact:").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(self.mb_frame, textvariable=self.mb_contact_var, width=40).grid(row=2, column=1, sticky="w", padx=5, pady=2)

        self.test_btn = ttk.Button(self, text="Test Connection", command=self._on_test)
        self.status_lbl = ttk.Label(self, textvariable=self.status_var)
        self.save_btn = ttk.Button(self, text="Save", command=self._on_save, state="disabled")

        self.inputs = [self.service_box, self.api_entry, self.mb_frame]
        self._update_visible()

    def _set_state(self, state: str) -> None:
        for w in self.inputs:
            w_state = getattr(w, "state", None)
            try:
                if isinstance(w, ttk.Frame):
                    for child in w.winfo_children():
                        child.configure(state=state)
                else:
                    w.configure(state=state)
            except Exception:
                pass

    def _save_values(self) -> None:
        cfg = load_config()
        cfg["metadata_service"] = self.service_var.get()
        cfg["metadata_api_key"] = self.api_var.get()
        cfg["musicbrainz_useragent"] = {
            "app": self.mb_app_var.get(),
            "version": self.mb_ver_var.get(),
            "contact": self.mb_contact_var.get(),
        }
        save_config(cfg)

    def _on_test(self) -> None:
        self._save_values()
        self._set_state("disabled")
        self.status_var.set("Testing…")

        if self.service_var.get() == "MusicBrainz":
            service = MusicBrainzService()
        else:
            service = AcoustIDService()

        def worker() -> None:
            ok, msg = service.test_connection()
            def done() -> None:
                self.last_ok = ok
                self.status_var.set(msg)
                self.status_lbl.configure(foreground="green" if ok else "red")
                self._set_state("normal")
                self.save_btn.configure(state="normal" if ok else "disabled")
            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_save(self) -> None:
        self._save_values()
        self.last_ok = False
        self.save_btn.configure(state="disabled")

    def _update_visible(self) -> None:
        svc = self.service_var.get()
        row = 1
        if svc == "AcoustID":
            self.mb_frame.grid_forget()
            self.api_lbl.grid(row=row, column=0, sticky="e", padx=5, pady=5)
            self.api_entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)
            row += 1
        else:
            self.api_lbl.grid_forget()
            self.api_entry.grid_forget()
            self.mb_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=5)
            row += 1
        self.test_btn.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.status_lbl.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        self.save_btn.grid(row=row + 1, column=1, sticky="e", padx=5, pady=5)
        self.save_btn.configure(state="normal" if self.last_ok else "disabled")
