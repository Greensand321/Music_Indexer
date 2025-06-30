# SoundVault Music Indexer

SoundVault organizes large music libraries. It deduplicates tracks, fixes tags via AcoustID, normalizes genres, and generates playlists while keeping your folder structure intact.

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

1. **Open** your library folder
2. Use the **Indexer** tab to dedupe and move files
3. **Fix Tags** via the AcoustID menu
4. **Generate Playlists** from your folder structure
5. Use the **Help** tab for inline assistance

## Roadmap (Upcoming Features)

These items are currently under development and not yet part of the stable release.

- Smarter Playlist Engine
- Tempo/Energy Buckets
- "More Like This" suggestions
- Clustered Playlists (K-Means, HDBSCAN)
- Deej-AI / Auto-DJ Integration
- Metadata Plugins (Discogs, Spotify)
- UI Polish & Theme Toggles (Completed)

See [`docs/project_documentation.html`](docs/project_documentation.html) for technical details.
