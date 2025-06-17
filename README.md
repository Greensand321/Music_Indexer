# Music Indexer (aka ALPHADEX)

Scan, tag-fix, normalize, and generate playlists that preserve your original folder structures across duplicates.

## Prerequisites

- **Python 3.11+** (use [conda](https://docs.conda.io/en/latest/miniconda.html) or `venv`)
- **Git** command line
  ```bash
  git clone --recurse-submodules https://example.com/yourrepo.git
  ```
- **FFmpeg** installed and on your `PATH`
- **llama.exe** command line for LLM features

## Installation

Create and activate a virtual environment then install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # or "Scripts\activate" on Windows
pip install -r requirements.txt
```

## Quickstart

```bash
python main_gui.py
```

1. **Scan** your library
2. **Generate Playlist** (see the new *Playlists* tab)
3. Visit the **Help** tab for tips

## Roadmap (Upcoming)

- Playlist Engine
- Tempo/Energy Bucket Generation
- "More Like This" (Nearest-Neighbor)
- Clustered Playlists (K-Means, HDBSCAN)
- Deej-AI / Auto-DJ Integration
- Metadata Plugins (Discogs, Spotify)
- UI Polish & Theme Toggles

See [`docs/project_documentation.html`](docs/project_documentation.html) for technical details.
