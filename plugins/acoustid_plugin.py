import acoustid
import musicbrainzngs
from itertools import islice
import tkinter as tk
from tkinter import messagebox, ttk
import importlib

from plugins.base import MetadataPlugin
from utils.path_helpers import ensure_long_path
import tag_fixer
from tag_fixer import (
    ACOUSTID_APP_NAME,
    ACOUSTID_APP_VERSION,
)
from config import load_config, save_config, SUPPORTED_SERVICES
from metadata_service import query_metadata

musicbrainzngs.set_useragent(
    ACOUSTID_APP_NAME,
    ACOUSTID_APP_VERSION,
    "youremail@example.com",
)

class AcoustIDPlugin(MetadataPlugin):
    @staticmethod
    def _prompt_reconnect() -> bool:
        """Prompt for service selection and API key update."""
        cfg = load_config()
        root = tk.Tk()
        root.withdraw()
        top = tk.Toplevel(root)
        top.title("Metadata Connection Failed")

        services = []
        for svc in SUPPORTED_SERVICES:
            if svc == "Spotify" and importlib.util.find_spec("spotipy") is None:
                continue
            services.append(svc)

        tk.Label(top, text="Service:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        service_var = tk.StringVar(value=cfg.get("metadata_service", "AcoustID"))
        ttk.Combobox(top, textvariable=service_var, values=services, state="readonly").grid(row=0, column=1, padx=5, pady=5)

        tk.Label(top, text="API Key:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        api_var = tk.StringVar(value=cfg.get("metadata_api_key", tag_fixer.ACOUSTID_API_KEY))
        ttk.Entry(top, textvariable=api_var, width=40).grid(row=1, column=1, padx=5, pady=5)

        result = {"ok": False}

        def do_test() -> None:
            cfg = load_config()
            cfg["metadata_service"] = service_var.get()
            cfg["metadata_api_key"] = api_var.get()
            save_config(cfg)
            if service_var.get() == "AcoustID":
                tag_fixer.ACOUSTID_API_KEY = api_var.get()
            try:
                query_metadata(service_var.get(), api_var.get(), "")
            except Exception:
                messagebox.showerror("Connection", "Connection failed", parent=top)
            else:
                messagebox.showinfo("Connection", "Connection success", parent=top)

        def do_save() -> None:
            cfg = load_config()
            cfg["metadata_service"] = service_var.get()
            cfg["metadata_api_key"] = api_var.get()
            save_config(cfg)
            if service_var.get() == "AcoustID":
                tag_fixer.ACOUSTID_API_KEY = api_var.get()
            result["ok"] = True
            top.destroy()

        ttk.Button(top, text="Test Connection", command=do_test).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(top, text="Save", command=do_save).grid(row=2, column=1, padx=5, pady=5)

        top.protocol("WM_DELETE_WINDOW", top.destroy)
        top.grab_set()
        root.wait_window(top)
        root.destroy()
        return result["ok"]

    def identify(self, file_path: str) -> dict:
        while True:
            cfg = load_config()
            service = cfg.get("metadata_service", "AcoustID")
            api_key = cfg.get("metadata_api_key", tag_fixer.ACOUSTID_API_KEY)
            try:
                return query_metadata(service, api_key, file_path)
            except acoustid.NoBackendError:
                return {}
            except acoustid.FingerprintGenerationError:
                return {}
            except Exception:
                if not self._prompt_reconnect():
                    return {}
                continue

    @staticmethod
    def check_connection() -> bool:
        """Return True if the configured metadata service is reachable."""
        cfg = load_config()
        service = cfg.get("metadata_service", "AcoustID")
        api_key = cfg.get("metadata_api_key", tag_fixer.ACOUSTID_API_KEY)
        try:
            query_metadata(service, api_key, "")
        except Exception:
            return AcoustIDPlugin._prompt_reconnect()
        return True
