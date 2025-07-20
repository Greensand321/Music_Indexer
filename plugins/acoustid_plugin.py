import threading
import tkinter as tk
from tkinter import ttk

import musicbrainzngs

from plugins.base import MetadataPlugin
from plugins.api_service import ApiService
from metadata_service import query_metadata
from config import load_config, save_config, CONFIG_PATH, SUPPORTED_SERVICES
import tag_fixer


class AcoustIDService(MetadataPlugin, ApiService):
    """Metadata lookup via the AcoustID web service."""

    def __init__(self):
        ApiService.__init__(self, CONFIG_PATH)

    def test_connection(self):
        import requests
        try:
            requests.get("https://api.acoustid.org/v2/", timeout=5)
        except Exception as e:
            return False, str(e)
        return True, "OK"

    def query(self, fingerprint: str):
        cfg = load_config()
        api_key = cfg.get("metadata_api_key", tag_fixer.ACOUSTID_API_KEY)
        return query_metadata("AcoustID", api_key, fingerprint)

    def identify(self, file_path: str) -> dict:  # for MetadataPlugin
        try:
            return self.query(file_path)
        except Exception:
            return {}

    @staticmethod
    def check_connection() -> bool:
        ok, _ = AcoustIDService().test_connection()
        return ok


class MusicBrainzService(ApiService):
    """MusicBrainz integration using ``musicbrainzngs``."""

    def __init__(self):
        ApiService.__init__(self, CONFIG_PATH)
        cfg = load_config()
        ua = cfg.get("musicbrainz_useragent", {})
        self.app = ua.get("app", "SoundVault")
        self.version = ua.get("version", "1.0")
        self.contact = ua.get("contact", "")

    def test_connection(self):
        from musicbrainzngs import MusicBrainzError

        try:
            musicbrainzngs.set_useragent(self.app, self.version, self.contact)
            res = musicbrainzngs.search_artists(query="Beatles", limit=1)
            n = len(res.get("artist-list", []))
            return True, f"OK – found {n} artist(s)"
        except MusicBrainzError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def query(self, fingerprint: str):
        musicbrainzngs.set_useragent(self.app, self.version, self.contact)
        return query_metadata("MusicBrainz", self.contact, fingerprint)


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
