import os
from datetime import datetime

import numpy as np
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


class ClusterGraphPanel(ttk.Frame):
    """Interactive scatter plot with lasso selection for playlist creation."""

    def __init__(
        self,
        parent,
        tracks,
        features,
        cluster_func,
        cluster_params,
        library_path,
        log_callback,
        on_params_updated=None,
    ):
        super().__init__(parent)
        self.tracks = tracks
        self.features = features
        self.cluster_func = cluster_func
        self.cluster_params = cluster_params
        self.library_path = library_path
        self.log = log_callback
        self.on_params_updated = on_params_updated

        from sklearn.decomposition import PCA

        X = np.vstack(features)
        self.X2 = PCA(n_components=2).fit_transform(X)

        fig = Figure(figsize=(5, 5))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self.scatter = None
        self._draw_clusters(cluster_func(X, self._cluster_kwargs(cluster_params)))

        # Track resize events so the canvas redraws to the latest geometry
        self._pending_size: tuple[int, int] | None = None
        self._resize_after_id: str | None = None
        self.canvas_widget.bind("<Configure>", self._on_resize)
        # Force an initial layout pass once the widget is visible
        self.after_idle(self._apply_pending_resize)

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

    def _on_resize(self, _event):
        """Schedule a canvas redraw after geometry changes."""
        self._pending_size = (
            self.canvas_widget.winfo_width(),
            self.canvas_widget.winfo_height(),
        )
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(50, self._apply_pending_resize)

    def _apply_pending_resize(self):
        """Redraw the canvas using the latest widget size."""
        self._resize_after_id = None
        width, height = self._pending_size or (
            self.canvas_widget.winfo_width(),
            self.canvas_widget.winfo_height(),
        )
        if width <= 1 or height <= 1:
            return

        dpi = self.canvas.figure.get_dpi()
        self.canvas.figure.set_size_inches(width / dpi, height / dpi, forward=True)
        self.canvas.figure.tight_layout()
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
        self.ax.clear()
        self.scatter = self.ax.scatter(
            self.X2[:, 0],
            self.X2[:, 1],
            c=point_colors,
            s=20,
        )
        self.ax.set_title("Lasso to select & generate playlist")
        self.canvas.draw_idle()

    def _cluster_kwargs(self, params: dict | None = None) -> dict:
        """Return clustering-only parameters without metadata keys."""

        if params is None:
            params = self.cluster_params
        return {
            k: v
            for k, v in params.items()
            if k not in {"method", "engine"}
        }

    def recluster(self, params: dict):
        """Re-run clustering with new ``params`` and redraw the plot."""
        meta = {k: v for k, v in self.cluster_params.items() if k in {"method", "engine"}}
        self.cluster_params = {**meta, **params}
        if self.lasso is not None:
            self.toggle_lasso()
        self._clear_selection()
        X = np.vstack(self.features)
        labels = self.cluster_func(X, self._cluster_kwargs())
        self._draw_clusters(labels)
        if self.on_params_updated:
            self.on_params_updated(self.cluster_params)

    def open_param_dialog(self):
        """Show dialog to edit HDBSCAN parameters and redraw clusters."""
        dlg = tk.Toplevel(self)
        dlg.title("HDBSCAN Parameters")
        dlg.grab_set()

        cs_var = tk.StringVar(value=str(self.cluster_params.get("min_cluster_size", 25)))
        ms_var = tk.StringVar(value=str(self.cluster_params.get("min_samples", "")))
        eps_var = tk.StringVar(
            value=str(self.cluster_params.get("cluster_selection_epsilon", ""))
        )

        ttk.Label(dlg, text="Min cluster size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(dlg, textvariable=cs_var, width=10).grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Label(dlg, text="Min samples (optional):").grid(row=1, column=0, sticky="w")
        ttk.Entry(dlg, textvariable=ms_var, width=10).grid(row=1, column=1, sticky="w", padx=(5, 0))
        ttk.Label(dlg, text="Epsilon (advanced, optional):").grid(row=2, column=0, sticky="w")
        ttk.Entry(dlg, textvariable=eps_var, width=10).grid(row=2, column=1, sticky="w", padx=(5, 0))

        ttk.Label(
            dlg,
            text=(
                "Suggested min cluster size: 5–20 for <500 tracks, 10–50 for "
                "500–2k, 25–150 for 2k–10k, 50–500 for 10k+. Smaller values "
                "find niche moods; larger values find broad, stable clusters."
            ),
            wraplength=360,
            foreground="gray",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(
            dlg,
            text=(
                "Leave min samples empty to match min cluster size. Higher "
                "values require tighter similarity and increase noise."
            ),
            wraplength=360,
            foreground="gray",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(
            dlg,
            text=(
                "Epsilon is rarely needed. Leave blank unless you need to merge "
                "nearby clusters; keep it under 0.2 to avoid destroying results."
            ),
            wraplength=360,
            foreground="gray",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        btns = ttk.Frame(dlg)
        btns.grid(row=6, column=0, columnspan=2, pady=(10, 5))

        def _clamp_min_cluster_size(value: int) -> int:
            return max(5, min(value, 500))

        def _clamp_min_samples(value: int, min_cluster_size: int) -> int:
            clamped = max(1, min(value, 20))
            return min(clamped, min_cluster_size)

        def _clamp_epsilon(value: float) -> float:
            return max(0.0, min(value, 0.2))

        def generate():
            try:
                cs_val = int(cs_var.get())
            except ValueError:
                messagebox.showerror("Invalid Value", f"{cs_var.get()} is not a valid number")
                return
            cs_val = _clamp_min_cluster_size(cs_val)
            cs_var.set(str(cs_val))
            params = {"min_cluster_size": cs_val}

            if ms_var.get().strip():
                try:
                    ms_val = int(ms_var.get())
                except ValueError:
                    messagebox.showerror("Invalid Value", f"{ms_var.get()} is not a valid number")
                    return
                ms_val = _clamp_min_samples(ms_val, cs_val)
                ms_var.set(str(ms_val))
                params["min_samples"] = ms_val
            if eps_var.get().strip():
                try:
                    eps_val = float(eps_var.get())
                except ValueError:
                    messagebox.showerror("Invalid Value", f"{eps_var.get()} is not a valid number")
                    return
                eps_val = _clamp_epsilon(eps_val)
                eps_var.set(f"{eps_val}")
                params["cluster_selection_epsilon"] = eps_val
            self.recluster(params)
            dlg.destroy()

        ttk.Button(btns, text="Generate", command=generate).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=5)
