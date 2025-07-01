import os
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from playlist_generator import write_playlist


class ClusterGraphPanel(ttk.Frame):
    """Interactive scatter plot with lasso selection for playlist creation."""

    def __init__(self, parent, tracks, features, cluster_func, cluster_params, library_path, log_callback):
        super().__init__(parent)
        self.tracks = tracks
        self.features = features
        self.cluster_func = cluster_func
        self.cluster_params = cluster_params
        self.library_path = library_path
        self.log = log_callback

        from sklearn.decomposition import PCA

        X = np.vstack(features)
        self.X2 = PCA(n_components=2).fit_transform(X)

        labels = cluster_func(X, cluster_params)
        colors = [f"C{l}" for l in labels]

        fig = Figure(figsize=(5, 5))
        self.ax = fig.add_subplot(111)
        self.scatter = self.ax.scatter(self.X2[:, 0], self.X2[:, 1], c=colors, s=20)
        self.ax.set_title("Lasso to select & generate playlist")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.lasso = None
        self.sel_scatter = None
        self.selected_indices: list[int] = []
        self.selected_tracks: list[str] = []

    # UI widgets created externally will be assigned after instantiation
    lasso_btn: ttk.Widget
    ok_btn: ttk.Button
    gen_btn: ttk.Button
    auto_var: 'tk.BooleanVar'

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
        """Lock selection and optionally auto-create playlist."""
        if not self.selected_indices:
            self.log("\u26A0 No points selected; please try again.")
            self.gen_btn.configure(state="disabled")
            return

        self.selected_tracks = [self.tracks[i] for i in self.selected_indices]
        if self.auto_var.get():
            self.create_playlist()
        else:
            self.gen_btn.configure(state="normal")

    # ─── Playlist Creation ──────────────────────────────────────────────────
    def create_playlist(self):
        if not self.selected_tracks:
            self.log("\u26A0 No songs selected")
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

