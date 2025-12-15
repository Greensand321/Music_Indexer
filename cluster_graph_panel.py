import os
import threading
from datetime import datetime

import numpy as np
import time
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from playlist_generator import write_playlist

from io import BytesIO
from PIL import Image, ImageTk
from mutagen import File as MutagenFile
from music_indexer_api import get_tags

logger = logging.getLogger(__name__)


class ClusterGraphPanel(ttk.Frame):
    """Interactive scatter plot with lasso selection for playlist creation."""

    def __init__(
        self,
        parent,
        tracks,
        X: np.ndarray,
        X2: np.ndarray,
        cluster_func,
        cluster_params,
        cluster_manager,
        algo_key: str,
        library_path,
        log_callback,
    ):
        super().__init__(parent)
        self.tracks = tracks
        self.X = X
        self.X2 = X2
        self.cluster_func = cluster_func
        self.cluster_params = cluster_params
        self.library_path = library_path
        self.log = log_callback
        self.cluster_manager = cluster_manager
        self.algo_key = algo_key

        fig = Figure(figsize=(5, 5))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.scatter = None
        self._loading_var = tk.StringVar(value="Running clustering …")
        self._loading_lbl = ttk.Label(self, textvariable=self._loading_var)
        self._loading_lbl.pack(pady=10)
        self._clusters_ready = False
        self._cluster_thread: threading.Thread | None = None
        self._cluster_start_ts: float | None = None
        self._start_clustering(cluster_params)

        self.lasso = None
        self.sel_scatter = None
        self.selected_indices: list[int] = []
        self.selected_tracks: list[str] = []

        # Hover support widgets will be set later via ``setup_hover``
        self.hover_panel = None
        self.hover_album_label = None
        self.hover_title_label = None
        self.hover_artist_label = None
        self._prev_hover_index: int | None = None

        # Preload album art thumbnails for snappy hover updates
        self.album_thumbnails = [self._load_thumbnail(p) for p in tracks]

    # UI widgets created externally will be assigned after instantiation
    lasso_btn: ttk.Widget
    ok_btn: ttk.Button
    gen_btn: ttk.Button

    def _start_clustering(self, params: dict):
        """Kick off clustering work in a background thread."""

        def _worker():
            try:
                cached = self.cluster_manager.get_cached_labels(self.algo_key, params)
                if cached is None:
                    labels = self.cluster_manager.compute_labels(
                        self.algo_key, params, self.cluster_func
                    )
                else:
                    labels = cached
            except Exception as exc:  # pragma: no cover - defensive path for GUI
                self.after(0, lambda: self.log(f"\u2717 Clustering failed: {exc}"))
                return
            self.after(0, lambda: self._on_clusters_ready(labels, params))

        if self._cluster_thread and self._cluster_thread.is_alive():
            return

        cached = self.cluster_manager.get_cached_labels(self.algo_key, params)
        if cached is not None:
            self._on_clusters_ready(cached, params)
            return

        self._clusters_ready = False
        self._sync_controls()
        self._loading_var.set("Running clustering …")
        self._loading_lbl.pack(pady=10)
        self._cluster_start_ts = time.perf_counter()
        logger.info("[perf] start clustering %s params=%s", self.algo_key, params)
        self._cluster_thread = threading.Thread(target=_worker, daemon=True)
        self._cluster_thread.start()

    # ─── Lasso Handling ─────────────────────────────────────────────────────
    def toggle_lasso(self):
        """Enter or exit lasso selection mode."""
        if self.lasso is None:
            self._clear_selection()
            self.lasso = LassoSelector(self.ax, onselect=self._on_lasso_select)
            self.log("Lasso mode enabled – draw a shape")
        else:
            self.lasso.disconnect_events()
            self.lasso = None
            self.log("Lasso mode disabled")
            self._clear_selection()

        self.ok_btn.configure(state="disabled")
        self.gen_btn.configure(state="disabled")
        self.canvas.draw_idle()

    # ─── Hover Metadata Panel ────────────────────────────────────────────────
    def setup_hover(self, panel, album_label, title_label, artist_label):
        """Register widgets used for hover display."""
        self.hover_panel = panel
        self.hover_album_label = album_label
        self.hover_title_label = title_label
        self.hover_artist_label = artist_label
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _load_thumbnail(self, path: str) -> ImageTk.PhotoImage:
        """Return a 64x64 thumbnail for ``path`` or a gray placeholder."""
        img = None
        try:
            audio = MutagenFile(path)
            img_data = None
            if hasattr(audio, "tags") and audio.tags is not None:
                for key in audio.tags.keys():
                    if key.startswith("APIC"):
                        img_data = audio.tags[key].data
                        break
            if img_data is None and getattr(audio, "pictures", None):
                pics = getattr(audio, "pictures", [])
                if pics:
                    img_data = pics[0].data
            if img_data:
                img = Image.open(BytesIO(img_data))
        except Exception:
            img = None

        if img is None:
            img = Image.new("RGB", (64, 64), "#777777")

        img.thumbnail((64, 64))
        return ImageTk.PhotoImage(img)

    def _on_motion(self, event):
        if not self.hover_panel:
            return
        if event.inaxes != self.ax:
            self._hide_hover()
            return

        cont, details = self.scatter.contains(event)
        if not cont or not details.get("ind"):
            self._hide_hover()
            return

        idx = details["ind"][0]
        if self._prev_hover_index == idx:
            return

        self._prev_hover_index = idx
        track = self.tracks[idx]
        tags = get_tags(track)
        title = tags.get("title") or os.path.basename(track)
        artist = tags.get("artist") or "Unknown"

        thumb = self.album_thumbnails[idx]
        self.hover_album_label.configure(image=thumb)
        self.hover_album_label.image = thumb
        self.hover_title_label.configure(text=title)
        self.hover_artist_label.configure(text=artist)
        self.hover_panel.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

    def _hide_hover(self):
        if self._prev_hover_index is not None:
            self._prev_hover_index = None
            if self.hover_panel:
                self.hover_panel.place_forget()

    def _on_lasso_select(self, verts):
        path = Path(verts)
        mask = path.contains_points(self.X2)
        self.selected_indices = [i for i, v in enumerate(mask) if v]
        self.log(f"\u2192 {len(self.selected_indices)} songs selected")
        self._update_highlight()
        if self.selected_indices:
            self.ok_btn.configure(state="normal")
        else:
            self.ok_btn.configure(state="disabled")

    def finalize_lasso(self):
        """Lock selection and enable playlist creation."""
        if not self.selected_indices:
            self.log("\u26a0 No points selected; please try again.")
            self.gen_btn.configure(state="disabled")
            return

        self.selected_tracks = [self.tracks[i] for i in self.selected_indices]
        self.gen_btn.configure(state="normal")

    # ─── Playlist Creation ──────────────────────────────────────────────────
    def create_playlist(self):
        if not self.selected_tracks:
            self.log("\u26a0 No songs selected")
            return

        playlists_dir = os.path.join(self.library_path, "Playlists")
        os.makedirs(playlists_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(playlists_dir, f"CustomLasso_{ts}.m3u")
        out = base
        idx = 1
        while os.path.exists(out):
            out = base.replace(".m3u", f"_{idx}.m3u")
            idx += 1

        try:
            write_playlist(self.selected_tracks, out)
        except Exception as e:  # pragma: no cover - simple GUI log
            self.log(f"\u2717 Failed to write playlist: {e}")
        else:
            self.log(f"\u2713 Playlist written: {out}")
            self.gen_btn.configure(state="disabled")

    # ─── Helpers ────────────────────────────────────────────────────────────
    def _clear_selection(self):
        self.selected_indices = []
        self.selected_tracks = []
        if self.sel_scatter is not None:
            self.sel_scatter.remove()
            self.sel_scatter = None

    def _update_highlight(self):
        if self.sel_scatter is not None:
            self.sel_scatter.remove()
        if not self.selected_indices:
            self.sel_scatter = None
        else:
            pts = self.X2[self.selected_indices]
            self.sel_scatter = self.ax.scatter(
                pts[:, 0],
                pts[:, 1],
                facecolors="none",
                edgecolors="k",
                s=60,
            )
        self.canvas.draw_idle()

    # ─── Re-clustering Support ─────────────────────────────────────────────
    def _compute_colors(self, labels):
        from matplotlib import cm

        uniq = sorted([l for l in set(labels) if l >= 0])
        base_cmap = cm.get_cmap("tab20")
        colors = base_cmap(np.linspace(0, 1, max(len(uniq), 1)))
        label_colors = {l: colors[i % len(colors)] for i, l in enumerate(uniq)}
        label_colors[-1] = (0.6, 0.6, 0.6, 0.5)
        point_colors = [label_colors.get(l, (0.6, 0.6, 0.6, 0.5)) for l in labels]
        return point_colors, label_colors

    def _draw_clusters(self, labels):
        point_colors, self.label_colors = self._compute_colors(labels)
        self.labels = labels
        if self.scatter is None:
            self.scatter = self.ax.scatter(
                self.X2[:, 0],
                self.X2[:, 1],
                c=point_colors,
                s=20,
            )
            self.ax.set_title("Lasso to select & generate playlist")
        else:
            self.scatter.set_offsets(self.X2)
            self.scatter.set_color(point_colors)
        self.canvas.draw_idle()

    def _on_clusters_ready(self, labels, params: dict):
        duration_ms = None
        if self._cluster_start_ts is not None:
            duration_ms = (time.perf_counter() - self._cluster_start_ts) * 1000
        self.cluster_params = params
        self._clusters_ready = True
        if self._loading_lbl.winfo_exists():
            self._loading_lbl.pack_forget()
        self._draw_clusters(labels)
        if duration_ms is not None:
            logger.info(
                "[perf] clusters ready (%s) in %.1f ms", self.algo_key, duration_ms
            )
        self._sync_controls()

    def recluster(self, params: dict):
        """Re-run clustering with new ``params`` and redraw the plot."""
        if self.lasso is not None:
            self.toggle_lasso()
        self._clear_selection()
        self._loading_var.set("Updating clusters …")
        self._loading_lbl.pack(pady=10)
        self._start_clustering(params)

    def _sync_controls(self):
        """Enable/disable controls based on clustering readiness."""
        state = "normal" if self._clusters_ready else "disabled"
        if hasattr(self, "lasso_btn"):
            self.lasso_btn.configure(state=state)
        if not self._clusters_ready:
            for attr in ("ok_btn", "gen_btn"):
                if hasattr(self, attr):
                    getattr(self, attr).configure(state="disabled")

    def refresh_control_states(self):
        """Public hook to sync controls after external widgets are attached."""
        self._sync_controls()

    def open_param_dialog(self):
        """Show dialog to edit HDBSCAN parameters and redraw clusters."""
        dlg = tk.Toplevel(self)
        dlg.title("HDBSCAN Parameters")
        dlg.grab_set()

        cs_var = tk.StringVar(value=str(self.cluster_params.get("min_cluster_size", 5)))
        ms_var = tk.StringVar(value=str(self.cluster_params.get("min_samples", "")))
        eps_var = tk.StringVar(value=str(self.cluster_params.get("cluster_selection_epsilon", "")))

        ttk.Label(dlg, text="Min cluster size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(dlg, textvariable=cs_var, width=10).grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Label(dlg, text="Min samples:").grid(row=1, column=0, sticky="w")
        ttk.Entry(dlg, textvariable=ms_var, width=10).grid(row=1, column=1, sticky="w", padx=(5, 0))
        ttk.Label(dlg, text="Epsilon:").grid(row=2, column=0, sticky="w")
        ttk.Entry(dlg, textvariable=eps_var, width=10).grid(row=2, column=1, sticky="w", padx=(5, 0))

        btns = ttk.Frame(dlg)
        btns.grid(row=3, column=0, columnspan=2, pady=(10, 5))

        def generate():
            try:
                params = {"min_cluster_size": int(cs_var.get())}
            except ValueError:
                messagebox.showerror("Invalid Value", f"{cs_var.get()} is not a valid number")
                return
            if ms_var.get().strip():
                try:
                    params["min_samples"] = int(ms_var.get())
                except ValueError:
                    messagebox.showerror("Invalid Value", f"{ms_var.get()} is not a valid number")
                    return
            if eps_var.get().strip():
                try:
                    params["cluster_selection_epsilon"] = float(eps_var.get())
                except ValueError:
                    messagebox.showerror("Invalid Value", f"{eps_var.get()} is not a valid number")
                    return
            self.recluster(params)
            dlg.destroy()

        ttk.Button(btns, text="Generate", command=generate).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=5)
