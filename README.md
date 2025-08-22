# SW3 + Power Analyzer

Tkinter GUI to align and analyze **SW3/eegbin** optical measurements with **electrical power** logs (CSV).  
Includes grouping, many-to-many associations, global **trimming (HH:MM:SS)**, and plots: **Intensity vs Time**, **Optics vs Power**, **Decay Overlay**, and correlation/scatter.

![Quick Start Guide - How To](https://github.com/OSLUV/Sweep-Analyzer/blob/main/HowTo.png)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python gui.py
```

> Dependencies are pinned in `requirements.txt`. Update cautiously; SciPy/Matplotlib wheels must be available for your Python/OS/arch.  

## Build locally with PyInstaller

```bash
pip install pyinstaller==6.11.0
# macOS app bundle
pyinstaller --noconfirm --clean --windowed --name SW3PowerAnalyzer \
  --collect-data matplotlib --collect-data pandas \
  --collect-submodules matplotlib \
  --hidden-import matplotlib.backends.backend_tkagg \
  --hidden-import eegbin --hidden-import util \
  gui.py

# Windows/Linux single-file binaries
pyinstaller --noconfirm --clean --onefile --windowed --name SW3PowerAnalyzer \
  --collect-data matplotlib --collect-data pandas \
  --collect-submodules matplotlib \
  --hidden-import matplotlib.backends.backend_tkagg \
  --hidden-import eegbin --hidden-import util \
  gui.py
```

## Continuous Integration (GitHub Actions)

This repo includes a cross-platform workflow to build on **Ubuntu, Windows, and macOS (Intel & Apple Silicon)** and attach artifacts to releases when you push a tag starting with `v` (e.g. `v1.0.0`).  
See [`.github/workflows/build.yml`](.github/workflows/build.yml) for details.

### Usage
1. Commit and push to `main` to build and upload CI artifacts.
2. Create a version tag to publish a release:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

### macOS Gatekeeper

The macOS `.app` bundle is **unsigned**. Open it via right-click → Open (or code sign & notarize with your Apple Developer ID).

## Repo layout

```
.
├── gui.py
├── eegbin.py
├── util.py
├── requirements.txt
└── .github/
    └── workflows/
        └── build.yml
```

## Notes

- The GUI uses `TkAgg` and bundles Tcl/Tk via PyInstaller. On Linux CI we also install the `tk` package so import works at build-time.
- The app dynamically imports `eegbin`, so we include it as a PyInstaller **hidden import** (same for `util`).

