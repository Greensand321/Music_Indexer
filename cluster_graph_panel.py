import os
import numpy as np
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from playlist_generator import write_playlist


class ClusterGraphPanel(ttk.Frame):
    """Interactive scatter plot for selecting clustered songs."""

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
        ax = fig.add_subplot(111)
        ax.scatter(self.X2[:, 0], self.X2[:, 1], c=colors, s=20)
        ax.set_title("Lasso to select & generate playlist")
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def onselect(verts):
            path = Path(verts)
            mask = path.contains_points(self.X2)
            self.selected = [tracks[i] for i, v in enumerate(mask) if v]
            self.log(f"\u2192 {len(self.selected)} songs selected")

        LassoSelector(ax, onselect)

        btn = ttk.Button(self, text="Generate Playlist", command=self._on_generate)
        btn.pack(pady=5)

    def _on_generate(self):
        if not getattr(self, "selected", []):
            self.log("\u26A0 No songs selected")
            return
        playlists_dir = os.path.join(self.library_path, "Playlists")
        os.makedirs(playlists_dir, exist_ok=True)
        out = os.path.join(
            playlists_dir,
            f"Interactive_{self.cluster_params['method']}.m3u",
        )
        write_playlist(self.selected, out)
        self.log(f"\u2713 Playlist written: {out}")
