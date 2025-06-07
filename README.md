# Music Indexer / SoundVault

This project contains utilities and a GUI for organizing, tagging, and managing music files. Under the hood, it uses:

- [`mutagen`](https://mutagen.readthedocs.io/) to read/write audio metadata  
- [`musicbrainzngs`](https://musicbrainzngs.readthedocs.io/) to query MusicBrainz for tags  
- [`pydub`](https://github.com/jiaaro/pydub) + FFmpeg for audio previewing (“Sample Song Highlight”)  
- [`pyacoustid`](https://github.com/beetbox/pyacoustid) for AcoustID-based tag fixing  
- [`librosa`](https://librosa.org/) (optional) for BPM estimation  

## Prerequisites

1. **Python 3.10 – 3.12** (Python 3.13 lacks `audioop`, so “Sample Song Highlight” will be disabled)  
2. **FFmpeg** on your PATH (for `pydub` to work)
   - On Windows, you can install via:
     - **Winget**:
       ```powershell
       winget install "FFmpeg (Essentials Build)"
       ```
     - or download an Essentials build from https://www.gyan.dev/ffmpeg/builds/, unzip, and add its `bin\` folder to your system PATH.
   - Verify with:
     ```bash
     ffmpeg -version
     ```
3. **Chromaprint's `fpcalc`** on your PATH (for the tag fixer)
4. **Git** (to clone/pull the repository)
