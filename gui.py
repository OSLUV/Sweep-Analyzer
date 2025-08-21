#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI for analyzing goniometer .sw3/eegbin files alongside electrical power logs.

Features
--------
• Load multiple SW3/eegbin files and multiple power CSV files
• Label/rename files; view quick metadata (first/last timestamp, sample count)
• Create multiple Groups; assign files to groups
• Associate SW3 file(s) with one or more power file(s) inside each group
• Align by absolute epoch timestamps (1 s resample for power); tolerance configurable
• Perform analyses equivalent to original scripts:
    - Intensity vs Time (Yaw=0°, Roll=0°, normalized to 1 m, EMA, trim start/end)
    - Optics vs Power (EMA, dual‑axis time series, Pearson r, cross‑correlation,
      scatter + regression, export aligned CSV)
    - Group Decay Overlay (EMA, normalized to % of peak; per‑group offset)

New in this build
-----------------
• **Reload Files**: reload selected/all files from disk.
• **Resizable panes**: left column split is resizable; tables have scrollbars.
• **Module Paths**: Settings → *Add Module Path…* so you can point the app at
  your `goniometer_software` repo (where `eegbin.py` and `util.py` live). If
  `eegbin`/`util` import fails, the app will prompt you to locate the folder and
  retry automatically. Paths are saved in the project JSON.

Run
---
    python gui.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set

# --- Third‑party ---
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# --- Tkinter UI ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# --- Optional SciPy for stats ---
try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

# Delay import of eegbin until needed (so the GUI can open without all deps installed)
eegbin = None

# Track extra module search paths (saved to project)
_EXTRA_MODULE_PATHS: List[str] = []

# -----------------------------
# Utility helpers
# -----------------------------

def human_datetime(epoch_s: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_s))
    except Exception:
        return "?"

def parse_hhmmss(s: str) -> int:
    """Return seconds from 'HH:MM:SS' (or 'MM:SS' or integer seconds)."""
    if s is None:
        return 0
    s = str(s).strip()
    if not s:
        return 0
    if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
        return int(s)
    parts = s.split(':')
    try:
        parts = [int(p) for p in parts]
    except Exception:
        raise ValueError(f"Invalid time format: {s}")
    if len(parts) == 2:
        mm, ss = parts
        return mm*60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh*3600 + mm*60 + ss
    raise ValueError(f"Invalid time format: {s}")

def fmt_hhmmss(seconds: int) -> str:
    neg = seconds < 0
    seconds = abs(int(seconds))
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"-{hh:02d}:{mm:02d}:{ss:02d}" if neg else f"{hh:02d}:{mm:02d}:{ss:02d}"

def _add_module_path(path: str):
    if path and path not in sys.path:
        sys.path.insert(0, path)
    if path and path not in _EXTRA_MODULE_PATHS:
        _EXTRA_MODULE_PATHS.append(path)

def ensure_eegbin_imported():
    """Try to import eegbin; if util or other local deps are missing,
    prompt the user to locate the repo folder and retry.
    """
    global eegbin
    if eegbin is not None:
        return
    import importlib
    # First, honor any saved extra paths
    for p in list(_EXTRA_MODULE_PATHS):
        _add_module_path(p)

    try:
        eegbin = importlib.import_module("eegbin")
        return
    except ModuleNotFoundError as e1:
        missing = getattr(e1, "name", "eegbin")
        # Ask the user to locate the folder containing eegbin.py and util.py
        if messagebox.askyesno(
            "Locate module",
            f"Could not import '{missing}'.\n\n"
            f"Select the folder that contains your goniometer code (e.g., where 'eegbin.py' and 'util.py' live)."
        ):
            folder = filedialog.askdirectory(title="Select folder containing eegbin.py and util.py")
            if folder:
                _add_module_path(folder)
                try:
                    eegbin = importlib.import_module("eegbin")
                    return
                except Exception as e2:
                    # As a last resort, try loading eegbin.py directly from the chosen folder
                    try:
                        import importlib.util, os
                        path = os.path.join(folder, "eegbin.py")
                        if os.path.exists(path):
                            spec = importlib.util.spec_from_file_location("eegbin", path)
                            mod = importlib.util.module_from_spec(spec)
                            sys.modules["eegbin"] = mod
                            spec.loader.exec_module(mod)  # type: ignore
                            eegbin = mod
                            return
                    except Exception:
                        pass
        # Failed
        messagebox.showerror("Import error",
                             f"Could not import 'eegbin'.\n"
                             f"Missing module: {missing}\n\n"
                             f"Tip: Put gui.py in the same folder as eegbin.py and util.py, or use Settings → Add Module Path…")
        raise

def ema_series(series: np.ndarray, span: int) -> np.ndarray:
    """Simple EMA using pandas for numerical stability; fall back if needed."""
    if span <= 1:
        return np.asarray(series, dtype=float)
    try:
        s = pd.Series(series, dtype='float64')
        return s.ewm(span=span, adjust=False).mean().to_numpy()
    except Exception:
        # Lightweight fallback
        alpha = 2 / (span + 1.0)
        y = np.empty_like(series, dtype=float)
        y[0] = series[0]
        for i in range(1, len(series)):
            y[i] = alpha*series[i] + (1-alpha)*y[i-1]
        return y

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, r (Pearson). Uses SciPy if available, else numpy."""
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    if scipy_stats is not None:
        r = scipy_stats.pearsonr(x, y)[0]
        slope, intercept, *_ = scipy_stats.linregress(x, y)
        return slope, intercept, r
    # Fallbacks
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), 0.0
    r = np.corrcoef(x, y)[0,1]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), float(r)

def merge_asof_seconds(df_left: pd.DataFrame, df_right: pd.DataFrame, tolerance_s: int) -> pd.DataFrame:
    """Nearest merge on seconds‑resolution timestamps with tolerance."""
    left = df_left.copy()
    right = df_right.copy()
    left['timestamp'] = pd.to_datetime(left['timestamp'], unit='s')
    right['timestamp'] = pd.to_datetime(right['timestamp'], unit='s')
    left = left.sort_values('timestamp')
    right = right.sort_values('timestamp')
    merged = pd.merge_asof(
        left, right,
        on='timestamp', direction='nearest', tolerance=pd.Timedelta(seconds=tolerance_s)
    )
    # Convert back to float epoch seconds for export convenience
    merged['timestamp_s'] = merged['timestamp'].astype('int64') // 10**9
    return merged

# -----------------------------
# Data models
# -----------------------------

@dataclass
class FileRecord:
    file_id: str                   # stable key
    kind: str                      # 'sw3' or 'power'
    path: str
    label: str
    meta: Dict[str, object] = field(default_factory=dict)

@dataclass
class GroupRecord:
    group_id: str
    name: str
    offset_s: int = 0
    file_ids: Set[str] = field(default_factory=set)  # sw3 and/or power
    # Associations: map sw3 file_id -> set of power file_ids
    associations: Dict[str, Set[str]] = field(default_factory=dict)

# -----------------------------
# Core app
# -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SW3 + Power Analyzer")
        self.geometry("1280x800")
        self.minsize(900, 600)

        # State
        self.files: Dict[str, FileRecord] = {}
        self.groups: Dict[str, GroupRecord] = {}

        self.power_column_map = {
            "timestamp": ["Timestamp", "timestamp", "epoch", "Epoch", "time", "Time"],
            "watts": ["W_Active", "watts", "Watts", "Power_W", "W"]
        }

        self._init_control_vars()
        self._build_ui()

        # Last generated figure (for quick save)
        self._last_fig: Optional[plt.Figure] = None

    # -------------------------
    # Tk variables
    # -------------------------
    def _init_control_vars(self):
        self.intensity_ema_span   = tk.IntVar(self, value=31)
        self.power_ema_span       = tk.IntVar(self, value=31)
        self.trim_start_s         = tk.IntVar(self, value=0)
        self.trim_end_s           = tk.IntVar(self, value=0)
        self.align_tolerance_s    = tk.IntVar(self, value=2)
        self.ccf_max_lag_s        = tk.IntVar(self, value=180)

        self.group_offset_hms     = tk.StringVar(self, value="00:00:00")
        self.overlay_ema_span     = tk.IntVar(self, value=31)

        self.normalize_to_1m      = tk.BooleanVar(self, value=True)
        self.only_yaw_roll_zero   = tk.BooleanVar(self, value=True)

        self.resample_seconds     = tk.IntVar(self, value=1)

        self.selected_group_id    = tk.StringVar(self, value="")
        self.selected_sw3_id      = tk.StringVar(self, value="")

    # -------------------------
    # UI layout
    # -------------------------
    def _build_ui(self):
        self._build_menu()

        root = self
        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: files + groups (now vertically resizable)
        left = ttk.Frame(main)
        main.add(left, weight=1)

        left_panes = ttk.Panedwindow(left, orient=tk.VERTICAL)
        left_panes.pack(fill=tk.BOTH, expand=True)

        files_container = ttk.Frame(left_panes)
        groups_container = ttk.Frame(left_panes)
        left_panes.add(files_container, weight=3)
        left_panes.add(groups_container, weight=2)

        # Right: analysis tabs
        right = ttk.Frame(main)
        main.add(right, weight=4)

        # Files frame
        self._build_files_frame(files_container)

        # Groups frame
        self._build_groups_frame(groups_container)

        # Tabs
        self._notebook = ttk.Notebook(right)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_tab_intensity_vs_time()
        self._build_tab_optics_vs_power()
        self._build_tab_group_decay()

        # Status bar
        self.status_var = tk.StringVar(self, value="Ready")
        status = ttk.Label(root, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Add SW3...", command=self.on_add_sw3_files)
        file_menu.add_command(label="Add Power CSV...", command=self.on_add_power_files)
        file_menu.add_separator()
        file_menu.add_command(label="Reload Files", command=self.on_reload_all_files)
        file_menu.add_separator()
        file_menu.add_command(label="Save Project...", command=self.on_save_project)
        file_menu.add_command(label="Load Project...", command=self.on_load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(label="Power Columns…", command=self.on_set_power_columns)
        settings_menu.add_separator()
        settings_menu.add_command(label="Add Module Path…", command=self.on_add_module_path)
        settings_menu.add_command(label="List Module Paths…", command=self.on_list_module_paths)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About",
            "SW3 + Power Analyzer\n\nAligns SW3/eegbin optical data with power logs by epoch time."
        ))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_files_frame(self, parent):
        frm = ttk.LabelFrame(parent, text="Files")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Container so we can grid tree + scrollbars
        container = ttk.Frame(frm)
        container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Treeview
        cols = ("label", "kind", "first", "last", "count", "path")
        self.tv_files = ttk.Treeview(container, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.tv_files.heading(c, text=c.capitalize())
            # Make columns stretchable
            w = 140 if c not in ("path", "label") else 220
            self.tv_files.column(c, width=w, anchor="w", stretch=True)

        # Scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tv_files.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tv_files.xview)
        self.tv_files.configure(yscroll=vsb.set, xscroll=hsb.set)

        # Grid layout
        self.tv_files.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, padx=3, pady=3)
        ttk.Button(btns, text="Add SW3…", command=self.on_add_sw3_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Add Power CSV…", command=self.on_add_power_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Rename…", command=self.on_rename_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Remove", command=self.on_remove_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Assign to Group…", command=self.on_assign_files_to_group).pack(side=tk.LEFT, padx=2)
        ttk.Separator(btns, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(btns, text="Reload Selected", command=lambda: self.on_reload_all_files(only_selected=True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Reload All", command=self.on_reload_all_files).pack(side=tk.LEFT, padx=2)

        self.tv_files.bind("<Double-1>", lambda e: self.on_rename_file())

    def _build_groups_frame(self, parent):
        frm = ttk.LabelFrame(parent, text="Groups & Associations")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        top = ttk.Frame(frm); top.pack(fill=tk.X, padx=3, pady=3)
        ttk.Button(top, text="New Group…", command=self.on_new_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Rename…", command=self.on_rename_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Delete", command=self.on_delete_group).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(top, text="Edit Associations…", command=self.on_edit_associations).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(top, text="Set Offset…", command=self.on_set_group_offset).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Clear Offset", command=self.on_clear_group_offset).pack(side=tk.LEFT, padx=2)

        # Container for tree + scrollbars
        container = ttk.Frame(frm)
        container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Groups list
        cols = ("name", "offset", "sw3_count", "power_count")
        self.tv_groups = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tv_groups.heading(c, text=c.capitalize())
            self.tv_groups.column(c, width=150, anchor="w", stretch=True)

        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tv_groups.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tv_groups.xview)
        self.tv_groups.configure(yscroll=vsb.set, xscroll=hsb.set)

        self.tv_groups.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.tv_groups.bind("<<TreeviewSelect>>", lambda e: self._on_group_selection_changed())

    def _build_tab_intensity_vs_time(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="Intensity vs Time")

        controls = ttk.Frame(tab)
        controls.pack(fill=tk.X, padx=6, pady=6)

        ttk.Label(controls, text="SW3:").pack(side=tk.LEFT, padx=3)
        self.cb_sw3 = ttk.Combobox(controls, state="readonly", width=40, textvariable=self.selected_sw3_id)
        self.cb_sw3.pack(side=tk.LEFT, padx=3)

        ttk.Label(controls, text="EMA span:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.intensity_ema_span, width=6).pack(side=tk.LEFT, padx=3)

        ttk.Label(controls, text="Trim start:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.trim_start_s, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Label(controls, text="s").pack(side=tk.LEFT)

        ttk.Label(controls, text="Trim end:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.trim_end_s, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Label(controls, text="s").pack(side=tk.LEFT)

        ttk.Checkbutton(controls, text="Normalize to 1 m", variable=self.normalize_to_1m).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(controls, text="Yaw=0 & Roll=0 only", variable=self.only_yaw_roll_zero).pack(side=tk.LEFT, padx=6)

        btns = ttk.Frame(tab); btns.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btns, text="Plot", command=self.on_plot_intensity_vs_time).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Save Figure…", command=self.on_save_last_figure).pack(side=tk.LEFT, padx=3)

        note = ttk.Label(tab, text="Trim accepts seconds (e.g. 90) or HH:MM:SS (e.g. 00:01:30).")
        note.pack(anchor="w", padx=6)

    def _build_tab_optics_vs_power(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="Optics vs Power")

        controls = ttk.Frame(tab)
        controls.pack(fill=tk.X, padx=6, pady=6)

        ttk.Label(controls, text="Group:").pack(side=tk.LEFT, padx=3)
        self.cb_group = ttk.Combobox(controls, state="readonly", width=40, textvariable=self.selected_group_id)
        self.cb_group.pack(side=tk.LEFT, padx=3)

        ttk.Label(controls, text="Intensity EMA:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.intensity_ema_span, width=6).pack(side=tk.LEFT, padx=3)

        ttk.Label(controls, text="Power EMA:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.power_ema_span, width=6).pack(side=tk.LEFT, padx=3)

        ttk.Label(controls, text="Align tol:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.align_tolerance_s, width=6).pack(side=tk.LEFT, padx=3)
        ttk.Label(controls, text="s").pack(side=tk.LEFT)

        ttk.Label(controls, text="Max lag:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.ccf_max_lag_s, width=6).pack(side=tk.LEFT, padx=3)
        ttk.Label(controls, text="s").pack(side=tk.LEFT)

        btns = ttk.Frame(tab); btns.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btns, text="Analyze Group", command=self.on_analyze_group).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Scatter + Fit", command=self.on_scatter_fit).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Export Aligned CSV…", command=self.on_export_aligned_csv).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Save Figure…", command=self.on_save_last_figure).pack(side=tk.LEFT, padx=3)

        note = ttk.Label(tab, text="Alignment uses absolute epoch timestamps with nearest merge; power is resampled to 1 s means.")
        note.pack(anchor="w", padx=6)

    def _build_tab_group_decay(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="Group Decay Overlay")

        controls = ttk.Frame(tab); controls.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(controls, text="EMA span:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(controls, textvariable=self.overlay_ema_span, width=6).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Plot All Groups", command=self.on_plot_group_decay).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Save Figure…", command=self.on_save_last_figure).pack(side=tk.LEFT, padx=3)

        note = ttk.Label(tab, text="Each group’s intensity is normalized to its own peak (100%). Per‑group offset is applied.")
        note.pack(anchor="w", padx=6)

    # -------------------------
    # Files logic
    # -------------------------
    def on_add_sw3_files(self):
        paths = filedialog.askopenfilenames(
            title="Add SW3/eegbin Files",
            filetypes=[("SW3/eegbin", "*.sw3 *.eegbin *.bin *.*")])
        if not paths:
            return
        ensure_eegbin_imported()
        added = 0
        for path in paths:
            try:
                with open(path, "rb") as fd:
                    buf = fd.read()
                # Try v3 then v2
                try:
                    sweep = eegbin.load_eegbin3(buf, from_path=path)
                except Exception:
                    sweep = eegbin.load_eegbin2(buf, from_path=path)
                meta = self._meta_from_sweep(sweep)
                file_id = self._next_file_id()
                rec = FileRecord(file_id=file_id, kind="sw3", path=path,
                                 label=os.path.basename(path), meta=meta)
                self.files[file_id] = rec
                added += 1
            except Exception as e:
                traceback.print_exc()
                messagebox.showwarning("Load error", f"Failed to load {path}\n{e}")
        if added:
            self._refresh_files_tv()
            self._refresh_sw3_combobox()
            self._set_status(f"Added {added} SW3 file(s).")

    def on_add_power_files(self):
        paths = filedialog.askopenfilenames(
            title="Add Power CSV Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not paths:
            return
        added = 0
        for path in paths:
            try:
                df, meta = self._load_power_csv_with_meta(path)
                # We only keep meta here; CSV is re‑loaded on demand (memory‑friendly).
                file_id = self._next_file_id()
                rec = FileRecord(file_id=file_id, kind="power", path=path,
                                 label=os.path.basename(path), meta=meta)
                self.files[file_id] = rec
                added += 1
            except Exception as e:
                traceback.print_exc()
                messagebox.showwarning("Load error", f"Failed to load {path}\n{e}")
        if added:
            self._refresh_files_tv()
            self._set_status(f"Added {added} power file(s).")

    def _meta_from_sweep(self, sweep) -> Dict[str, object]:
        rows = [r for r in sweep.rows if getattr(r, "timestamp", None) is not None]
        ts = [r.timestamp for r in rows]
        first = min(ts) if ts else None
        last = max(ts) if ts else None
        return {
            "first_ts": first,
            "last_ts": last,
            "count": len(rows),
            "lamp_name": getattr(sweep, "lamp_name", ""),
        }

    def _load_power_csv_with_meta(self, path: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
        df = pd.read_csv(path)
        ts_col = self._detect_col(df.columns, self.power_column_map["timestamp"])
        watt_col = self._detect_col(df.columns, self.power_column_map["watts"])
        if not ts_col or not watt_col:
            raise ValueError(
                f"CSV must have a timestamp and watts column. Tried {self.power_column_map}. "
                f"Columns found: {list(df.columns)}"
            )
        df = df[[ts_col, watt_col]].rename(columns={ts_col: "Timestamp", watt_col: "W_Active"})
        # ensure numeric
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
        df["W_Active"] = pd.to_numeric(df["W_Active"], errors="coerce")
        df = df.dropna(subset=["Timestamp", "W_Active"])
        meta = {
            "first_ts": float(df["Timestamp"].min()) if not df.empty else None,
            "last_ts": float(df["Timestamp"].max()) if not df.empty else None,
            "count": int(len(df)),
        }
        return df, meta

    def _detect_col(self, cols, candidates: List[str]) -> Optional[str]:
        lower = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    def _refresh_files_tv(self):
        tv = self.tv_files
        for item in tv.get_children():
            tv.delete(item)
        for file_id, rec in self.files.items():
            first = human_datetime(rec.meta.get("first_ts")) if rec.meta.get("first_ts") else ""
            last  = human_datetime(rec.meta.get("last_ts")) if rec.meta.get("last_ts") else ""
            count = rec.meta.get("count", "")
            tv.insert("", "end", iid=file_id, values=(rec.label, rec.kind, first, last, count, rec.path))

    def _refresh_sw3_combobox(self):
        sw3_items = [(fid, fr) for fid, fr in self.files.items() if fr.kind == "sw3"]
        self.cb_sw3["values"] = [fid for fid, _ in sw3_items]  # store id in values for simplicity
        # Keep current selection if still valid
        cur = self.selected_sw3_id.get()
        if cur and cur in self.files and self.files[cur].kind == "sw3":
            pass
        elif sw3_items:
            self.selected_sw3_id.set(sw3_items[0][0])
        else:
            self.selected_sw3_id.set("")

    def on_rename_file(self):
        sel = self.tv_files.selection()
        if not sel:
            return
        if len(sel) > 1:
            messagebox.showinfo("Rename", "Select a single file to rename.")
            return
        file_id = sel[0]
        rec = self.files[file_id]
        new = simpledialog.askstring("Rename", "New label:", initialvalue=rec.label, parent=self)
        if new:
            rec.label = new.strip()
            self._refresh_files_tv()

    def on_remove_files(self):
        sel = list(self.tv_files.selection())
        if not sel:
            return
        if not messagebox.askyesno("Remove", f"Remove {len(sel)} file(s) from the project?"):
            return
        for file_id in sel:
            # Remove from groups & associations
            for gid, g in self.groups.items():
                g.file_ids.discard(file_id)
                if file_id in g.associations:
                    del g.associations[file_id]
                for k in list(g.associations.keys()):
                    if file_id in g.associations[k]:
                        g.associations[k].discard(file_id)
            self.files.pop(file_id, None)
        self._refresh_files_tv()
        self._refresh_groups_tv()
        self._set_status("Removed selected files.")

    def on_assign_files_to_group(self):
        sel_files = list(self.tv_files.selection())
        if not sel_files:
            messagebox.showinfo("Assign", "Select files (SW3 and/or Power) to assign to a group.")
            return
        gid = self._ensure_group_selected_or_prompt()
        if not gid:
            return
        g = self.groups[gid]
        for fid in sel_files:
            g.file_ids.add(fid)
        self._refresh_groups_tv()
        self._set_status(f"Assigned {len(sel_files)} files to group '{g.name}'.")

    # -------------------------
    # Groups logic
    # -------------------------
    def on_new_group(self):
        name = simpledialog.askstring("New Group", "Group name:", parent=self)
        if not name:
            return
        gid = self._next_group_id()
        self.groups[gid] = GroupRecord(group_id=gid, name=name.strip())
        self._refresh_groups_tv()
        self._refresh_group_combobox()
        self._set_status(f"Created group '{name}'.")

    def on_rename_group(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid:
            return
        g = self.groups[gid]
        new = simpledialog.askstring("Rename Group", "New name:", initialvalue=g.name, parent=self)
        if new:
            g.name = new.strip()
            self._refresh_groups_tv()
            self._refresh_group_combobox()

    def on_delete_group(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid:
            return
        g = self.groups[gid]
        if not messagebox.askyesno("Delete Group", f"Delete group '{g.name}'? This does not delete files."):
            return
        del self.groups[gid]
        self._refresh_groups_tv()
        self._refresh_group_combobox()

    def on_edit_associations(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid:
            return
        g = self.groups[gid]

        # Prepare lists
        sw3s = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind == "sw3"]
        pows = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind == "power"]
        if not sw3s or not pows:
            messagebox.showinfo("Associations", "Assign at least one SW3 and one Power file to the group first.")
            return

        dlg = AssociationDialog(self, g, sw3s, pows)
        self.wait_window(dlg)
        if dlg.updated:
            self._set_status("Updated associations.")

    def on_set_group_offset(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid:
            return
        g = self.groups[gid]
        cur = fmt_hhmmss(g.offset_s)
        s = simpledialog.askstring("Set Offset", "Enter offset (HH:MM:SS, can be negative):", initialvalue=cur, parent=self)
        if s is None:
            return
        try:
            g.offset_s = parse_hhmmss(s)
        except Exception as e:
            messagebox.showwarning("Invalid", str(e))
            return
        self._refresh_groups_tv()

    def on_clear_group_offset(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid:
            return
        self.groups[gid].offset_s = 0
        self._refresh_groups_tv()

    def _ensure_group_selected_or_prompt(self) -> Optional[str]:
        sel = self.tv_groups.selection()
        if sel:
            return sel[0]
        # prompt user to choose
        if not self.groups:
            messagebox.showinfo("Groups", "Create a group first.")
            return None
        choices = list(self.groups.keys())
        labels = [self.groups[c].name for c in choices]
        idx = simpledialog.askinteger("Select Group", "Enter group number:\n" + "\n".join(f"{i+1}. {labels[i]}" for i in range(len(labels))),
                                      minvalue=1, maxvalue=len(labels), parent=self)
        if idx is None:
            return None
        return choices[idx-1]

    def _on_group_selection_changed(self):
        sel = self.tv_groups.selection()
        if sel:
            gid = sel[0]
            self.selected_group_id.set(gid)

    def _refresh_groups_tv(self):
        tv = self.tv_groups
        for item in tv.get_children():
            tv.delete(item)
        for gid, g in self.groups.items():
            sw3_count = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "sw3")
            pow_count = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "power")
            tv.insert("", "end", iid=gid, values=(g.name, fmt_hhmmss(g.offset_s), sw3_count, pow_count))

    def _refresh_group_combobox(self):
        items = [(gid, gr) for gid, gr in self.groups.items()]
        self.cb_group["values"] = [gid for gid, _ in items]
        if items and not self.selected_group_id.get():
            self.selected_group_id.set(items[0][0])

    # -------------------------
    # Project save/load
    # -------------------------
    def on_save_project(self):
        path = filedialog.asksaveasfilename(
            title="Save Project", defaultextension=".json",
            filetypes=[("Project JSON", "*.json")])
        if not path:
            return
        data = {
            "power_column_map": self.power_column_map,
            "module_paths": list(_EXTRA_MODULE_PATHS),
            "files": [asdict(fr) for fr in self.files.values()],
            "groups": [
                {
                    "group_id": g.group_id,
                    "name": g.name,
                    "offset_s": g.offset_s,
                    "file_ids": list(g.file_ids),
                    "associations": {k: list(v) for k, v in g.associations.items()},
                } for g in self.groups.values()
            ],
        }
        with open(path, "w") as fd:
            json.dump(data, fd, indent=2)
        self._set_status(f"Saved project to {path}")

    def on_load_project(self):
        path = filedialog.askopenfilename(
            title="Load Project", filetypes=[("Project JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "r") as fd:
            data = json.load(fd)
        self.power_column_map = data.get("power_column_map", self.power_column_map)
        # restore module paths
        _EXTRA_MODULE_PATHS.clear()
        for p in data.get("module_paths", []):
            _add_module_path(p)

        self.files = {fr["file_id"]: FileRecord(**fr) for fr in data.get("files", [])}
        self.groups = {}
        for g in data.get("groups", []):
            gr = GroupRecord(group_id=g["group_id"], name=g["name"], offset_s=g.get("offset_s", 0))
            gr.file_ids = set(g.get("file_ids", []))
            gr.associations = {k: set(v) for k, v in g.get("associations", {}).items()}
            self.groups[gr.group_id] = gr
        self._refresh_files_tv()
        self._refresh_groups_tv()
        self._refresh_group_combobox()
        self._refresh_sw3_combobox()
        self._set_status(f"Loaded project from {path}")

    def on_set_power_columns(self):
        top = tk.Toplevel(self); top.title("Power Columns Map")
        top.resizable(True, False)
        ttk.Label(top, text="Candidate names for timestamp column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        ts_entry = ttk.Entry(top, width=60)
        ts_entry.insert(0, ", ".join(self.power_column_map["timestamp"]))
        ts_entry.pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(top, text="Candidate names for watts column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        w_entry = ttk.Entry(top, width=60)
        w_entry.insert(0, ", ".join(self.power_column_map["watts"]))
        w_entry.pack(fill=tk.X, padx=6, pady=3)

        btns = ttk.Frame(top); btns.pack(fill=tk.X, padx=6, pady=6)
        def save():
            self.power_column_map["timestamp"] = [s.strip() for s in ts_entry.get().split(",") if s.strip()]
            self.power_column_map["watts"] = [s.strip() for s in w_entry.get().split(",") if s.strip()]
            top.destroy()
        ttk.Button(btns, text="OK", command=save).pack(side=tk.RIGHT, padx=3)
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=3)

    def on_add_module_path(self):
        folder = filedialog.askdirectory(title="Add folder to Python module search path (eegbin/util)")
        if not folder:
            return
        _add_module_path(folder)
        self._set_status(f"Added module path: {folder}")

    def on_list_module_paths(self):
        if not _EXTRA_MODULE_PATHS:
            messagebox.showinfo("Module Paths", "No extra module paths added.")
            return
        msg = "Extra module search paths (prepended to sys.path):\n\n"
        for p in _EXTRA_MODULE_PATHS:
            msg += f"• {p}\n"
        messagebox.showinfo("Module Paths", msg)

    # -------------------------
    # Reload logic
    # -------------------------
    def on_reload_all_files(self, only_selected: bool=False):
        """Reload file metadata (and schema inference) from disk.
        Useful when files are appended/updated during ongoing runs.
        """
        targets = []
        if only_selected:
            targets = list(self.tv_files.selection())
            if not targets:
                messagebox.showinfo("Reload", "No files selected.")
                return
        else:
            targets = list(self.files.keys())

        ensure_eegbin_imported()
        ok = 0
        err = 0
        for fid in targets:
            rec = self.files.get(fid)
            if not rec:
                continue
            try:
                if rec.kind == "sw3":
                    with open(rec.path, "rb") as fd:
                        buf = fd.read()
                    try:
                        sweep = eegbin.load_eegbin3(buf, from_path=rec.path)
                    except Exception:
                        sweep = eegbin.load_eegbin2(buf, from_path=rec.path)
                    rec.meta = self._meta_from_sweep(sweep)
                elif rec.kind == "power":
                    _, meta = self._load_power_csv_with_meta(rec.path)
                    rec.meta = meta
                ok += 1
            except Exception as e:
                err += 1
                traceback.print_exc()
        self._refresh_files_tv()
        self._refresh_sw3_combobox()
        self._set_status(f"Reloaded {ok} file(s){'; errors: '+str(err) if err else ''}.")

    # -------------------------
    # Analysis: Intensity vs Time
    # -------------------------
    def on_plot_intensity_vs_time(self):
        fid = self.selected_sw3_id.get() or (self.tv_files.selection()[0] if self.tv_files.selection() else None)
        if not fid or fid not in self.files or self.files[fid].kind != "sw3":
            messagebox.showinfo("Plot", "Choose an SW3 file (via the dropdown or select in the Files list).")
            return
        rec = self.files[fid]
        try:
            with open(rec.path, "rb") as fd:
                buf = fd.read()
            ensure_eegbin_imported()
            try:
                sweep = eegbin.load_eegbin3(buf, from_path=rec.path)
            except Exception:
                sweep = eegbin.load_eegbin2(buf, from_path=rec.path)
        except Exception as e:
            messagebox.showwarning("Load error", f"Failed to load {rec.path}\n{e}")
            return

        only_zero = self.only_yaw_roll_zero.get()
        normalize = self.normalize_to_1m.get()
        ema_span = int(self.intensity_ema_span.get())
        trim_start = int(self.trim_start_s.get())
        trim_end = int(self.trim_end_s.get())

        # Extract rows
        rows = []
        for r in sweep.rows:
            if hasattr(r, "valid") and not r.valid:
                continue
            if only_zero and not (getattr(r.coords, "yaw_deg", None) == 0 and getattr(r.coords, "roll_deg", None) == 0):
                continue
            if getattr(r, "timestamp", None) is None:
                continue
            # intensity (W/m^2) at measured distance -> normalize to 1 m if requested
            I_wm2 = float(getattr(r.capture, "integral_result", 0.0) or 0.0)
            if normalize:
                dist_m = float(getattr(r.coords, "lin_mm", 0.0) or 0.0) / 1000.0
                if dist_m > 0:
                    I_wm2 = I_wm2 * (dist_m**2)
            # Convert to µW/cm^2
            I_uW_cm2 = I_wm2 * 100.0
            rows.append( (float(r.timestamp), I_uW_cm2) )

        if not rows:
            messagebox.showinfo("Plot", "No rows available after filtering.")
            return

        rows.sort(key=lambda x: x[0])
        t = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)

        # Apply trim
        t0 = t[0]
        if trim_start > 0:
            mask = (t - t0) >= trim_start
            t, y = t[mask], y[mask]
        if trim_end > 0:
            t_end = t[-1]
            mask = (t_end - t) >= trim_end
            t, y = t[mask], y[mask]

        if len(y) == 0:
            messagebox.showinfo("Plot", "No data after trim.")
            return

        y_ema = ema_series(y, ema_span)

        # Build time axes
        t_hours = (t - t[0]) / 3600.0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(t_hours, y, s=8, alpha=0.15, label="Measured (Yaw=0, Roll=0)")
        ax.plot(t_hours, y_ema, linewidth=2, label=f"EMA (span={ema_span})")
        ax.set_title("Measured Light Intensity vs Time")
        ax.set_xlabel("Time Since Start (hours)")
        ax.set_ylabel("Normalized Intensity at 1 m (µW/cm²)")

        # Some padding
        if len(y) > 1:
            pad = (max(y)-min(y))*0.05
            ax.set_ylim(min(y)-pad, max(y)+pad)

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend()

        # Secondary x axis (absolute)
        try:
            import matplotlib.dates as mdates
            t0_num = mdates.epoch2num(float(t[0]))
            def hours_to_mpldate(x):
                arr = np.asarray(x, dtype=float)
                out = t0_num + arr / 24.0
                return out.item() if np.isscalar(x) else out
            secax = ax.secondary_xaxis("top", functions=(hours_to_mpldate, lambda x: (x - t0_num)*24.0))
            secax.set_xlabel("Absolute Date/Time")
            secax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
            fig.autofmt_xdate(rotation=45)
        except Exception:
            pass

        fig.tight_layout()
        self._last_fig = fig
        plt.show(block=False)
        self._set_status(f"Plotted intensity vs time for {rec.label}")

    # -------------------------
    # Analysis: Optics vs Power
    # -------------------------
    def on_analyze_group(self):
        gid = self.selected_group_id.get()
        if not gid or gid not in self.groups:
            messagebox.showinfo("Analyze", "Select a Group.")
            return
        g = self.groups[gid]
        # Collect associations
        if not g.associations:
            messagebox.showinfo("Analyze", "Set associations (SW3 → Power) for this group first.")
            return

        ema_int = int(self.intensity_ema_span.get())
        ema_pow = int(self.power_ema_span.get())
        tol_s = int(self.align_tolerance_s.get())
        max_lag = int(self.ccf_max_lag_s.get())
        normalize = self.normalize_to_1m.get()
        only_zero = self.only_yaw_roll_zero.get()

        # For each associated pair, produce aligned frame; then concatenate
        aligned_frames = []
        for sw3_id, pow_ids in g.associations.items():
            sw3_rec = self.files.get(sw3_id)
            if not sw3_rec or sw3_rec.kind != "sw3":
                continue
            # load SW3 rows
            try:
                with open(sw3_rec.path, "rb") as fd:
                    buf = fd.read()
                ensure_eegbin_imported()
                try:
                    sweep = eegbin.load_eegbin3(buf, from_path=sw3_rec.path)
                except Exception:
                    sweep = eegbin.load_eegbin2(buf, from_path=sw3_rec.path)
            except Exception as e:
                messagebox.showwarning("Load error", f"Failed to load {sw3_rec.path}\n{e}")
                continue

            sw_rows = []
            for r in sweep.rows:
                if hasattr(r, "valid") and not r.valid:
                    continue
                if only_zero and not (getattr(r.coords, "yaw_deg", None) == 0 and getattr(r.coords, "roll_deg", None) == 0):
                    continue
                if getattr(r, "timestamp", None) is None:
                    continue
                I_wm2 = float(getattr(r.capture, "integral_result", 0.0) or 0.0)
                if normalize:
                    dist_m = float(getattr(r.coords, "lin_mm", 0.0) or 0.0) / 1000.0
                    if dist_m > 0:
                        I_wm2 = I_wm2 * (dist_m**2)
                I_uW_cm2 = I_wm2 * 100.0
                sw_rows.append( (float(r.timestamp) + g.offset_s, I_uW_cm2) )  # apply group offset

            if not sw_rows:
                continue
            sw_rows.sort(key=lambda x: x[0])
            t_sw = np.array([r[0] for r in sw_rows], dtype=float)
            y_sw = np.array([r[1] for r in sw_rows], dtype=float)
            y_sw_ema = ema_series(y_sw, ema_int)
            df_sw = pd.DataFrame({"timestamp": t_sw, "intensity_ema": y_sw_ema})

            # For each power file associated
            for pid in pow_ids:
                pow_rec = self.files.get(pid)
                if not pow_rec or pow_rec.kind != "power":
                    continue
                try:
                    power_df, _ = self._load_power_csv_with_meta(pow_rec.path)
                except Exception as e:
                    messagebox.showwarning("Load error", f"Failed to load CSV {pow_rec.path}\n{e}")
                    continue
                # Resample to 1 s mean, compute EMA
                pdf = power_df.copy()
                pdf["timestamp"] = pd.to_datetime(pdf["Timestamp"], unit='s')
                pdf = pdf.set_index("timestamp").sort_index()
                pdf = pdf.resample(f"{self.resample_seconds.get()}S")["W_Active"].mean().to_frame("power")
                pdf["power_ema"] = pdf["power"].ewm(span=ema_pow, adjust=False).mean()
                pdf = pdf.dropna(subset=["power_ema"])
                pdf = pdf.reset_index()
                pdf["timestamp"] = pdf["timestamp"].astype('int64') // 10**9  # epoch seconds

                # Align
                df_join = merge_asof_seconds(df_sw, pdf[["timestamp", "power_ema"]].rename(columns={"power_ema": "power_ema"}), tol_s)
                df_join = df_join.dropna(subset=["power_ema"])
                df_join["group_id"] = gid
                df_join["sw3_id"] = sw3_id
                df_join["power_id"] = pid
                aligned_frames.append(df_join)

        if not aligned_frames:
            messagebox.showinfo("Analyze", "No aligned data could be formed. Check associations and timestamps.")
            return

        aligned = pd.concat(aligned_frames, ignore_index=True).sort_values("timestamp_s")
        if aligned.empty:
            messagebox.showinfo("Analyze", "Aligned data is empty.")
            return

        # Compute correlation & cross‑correlation on EMA series
        x = aligned["intensity_ema"].to_numpy(dtype=float)
        y = aligned["power_ema"].to_numpy(dtype=float)
        r = np.corrcoef(x, y)[0, 1] if np.std(x)>0 and np.std(y)>0 else float("nan")

        # Cross‑correlation scan (intensity relative to power): lag in seconds
        lags = np.arange(-int(self.ccf_max_lag_s.get()), int(self.ccf_max_lag_s.get())+1, 1, dtype=int)
        ccf_vals = []
        for lag in lags:
            if lag < 0:
                x_lag = x[-lag:]
                y_lag = y[:len(x_lag)]
            elif lag > 0:
                y_lag = y[lag:]
                x_lag = x[:len(y_lag)]
            else:
                x_lag = x
                y_lag = y
            if len(x_lag) == 0 or len(y_lag) == 0 or np.std(x_lag)==0 or np.std(y_lag)==0:
                ccf_vals.append(np.nan)
            else:
                ccf_vals.append(np.corrcoef(x_lag, y_lag)[0,1])
        ccf_vals = np.array(ccf_vals, dtype=float)
        best_idx = int(np.nanargmax(np.abs(ccf_vals))) if np.isfinite(ccf_vals).any() else None
        best_lag = lags[best_idx] if best_idx is not None else None
        best_ccf = ccf_vals[best_idx] if best_idx is not None else float("nan")

        # ---------------- Plots ----------------
        # 1) Time series (dual axis)
        fig_ts = plt.figure()
        ax1 = fig_ts.add_subplot(111)
        ax2 = ax1.twinx()
        t0 = aligned["timestamp_s"].min()
        t_hours = (aligned["timestamp_s"] - t0) / 3600.0
        ax1.plot(t_hours, aligned["intensity_ema"], label="Intensity (EMA)")
        ax2.plot(t_hours, aligned["power_ema"], label="Power (EMA)")
        ax1.set_xlabel("Time since start (hours)")
        ax1.set_ylabel("Intensity (µW/cm²)")
        ax2.set_ylabel("Power (W)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        # Build a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1+lines2, labels1+labels2, loc="best")
        fig_ts.tight_layout()
        self._last_fig = fig_ts
        plt.show(block=False)

        # 2) CCF plot
        fig_ccf = plt.figure()
        axc = fig_ccf.add_subplot(111)
        axc.plot(lags, ccf_vals)
        axc.axvline(0, linestyle="--", alpha=0.5)
        if best_idx is not None:
            axc.axvline(best_lag, linestyle=":")
            y_text = np.nanmax(ccf_vals) if np.isfinite(ccf_vals).any() else 0.0
            axc.text(best_lag, y_text, f"best lag={best_lag}s\nr={best_ccf:.3f}", ha="center", va="bottom")
        axc.set_title("Cross‑correlation (Intensity vs Power)")
        axc.set_xlabel("Lag (s) [positive = intensity lags power]")
        axc.set_ylabel("Correlation")
        axc.grid(True, linestyle="--", alpha=0.5)
        fig_ccf.tight_layout()
        self._last_fig = fig_ccf
        plt.show(block=False)

        # 3) Scatter + regression
        slope, intercept, r_lin = linear_regression(aligned["power_ema"].to_numpy(), aligned["intensity_ema"].to_numpy())
        fig_sc = plt.figure()
        axs = fig_sc.add_subplot(111)
        axs.scatter(aligned["power_ema"], aligned["intensity_ema"], s=10, alpha=0.3)
        if np.isfinite(slope) and np.isfinite(intercept):
            xs = np.linspace(aligned["power_ema"].min(), aligned["power_ema"].max(), 100)
            ys = slope*xs + intercept
            axs.plot(xs, ys, linewidth=2, label=f"Fit: y={slope:.3f}x+{intercept:.3f}  r={r_lin:.3f}")
        axs.set_xlabel("Power (W) [EMA]")
        axs.set_ylabel("Intensity (µW/cm²) [EMA]")
        axs.grid(True, linestyle="--", alpha=0.5)
        axs.legend()
        fig_sc.tight_layout()
        self._last_fig = fig_sc
        plt.show(block=False)

        msg = f"Pearson r (EMA series): {r:.3f}\nBest CCF lag: {best_lag}s (r={best_ccf:.3f})"
        self._set_status(msg)
        messagebox.showinfo("Analysis", msg)

        # Keep the aligned frame for export
        self._aligned_cache = aligned

    def on_scatter_fit(self):
        # Convenience: if user clicks it separately (already handled in on_analyze_group)
        if not hasattr(self, "_aligned_cache") or self._aligned_cache.empty:
            messagebox.showinfo("Scatter", "Run 'Analyze Group' first.")
            return
        aligned = self._aligned_cache
        slope, intercept, r_lin = linear_regression(aligned["power_ema"].to_numpy(), aligned["intensity_ema"].to_numpy())
        fig_sc = plt.figure()
        axs = fig_sc.add_subplot(111)
        axs.scatter(aligned["power_ema"], aligned["intensity_ema"], s=10, alpha=0.3)
        if np.isfinite(slope) and np.isfinite(intercept):
            xs = np.linspace(aligned["power_ema"].min(), aligned["power_ema"].max(), 100)
            ys = slope*xs + intercept
            axs.plot(xs, ys, linewidth=2, label=f"Fit: y={slope:.3f}x+{intercept:.3f}  r={r_lin:.3f}")
        axs.set_xlabel("Power (W) [EMA]")
        axs.set_ylabel("Intensity (µW/cm²) [EMA]")
        axs.grid(True, linestyle="--", alpha=0.5)
        axs.legend()
        fig_sc.tight_layout()
        self._last_fig = fig_sc
        plt.show(block=False)

    def on_export_aligned_csv(self):
        if not hasattr(self, "_aligned_cache") or self._aligned_cache.empty:
            messagebox.showinfo("Export", "No aligned data to export. Run 'Analyze Group' first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Aligned CSV", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        df = self._aligned_cache.copy()
        # Make columns user‑friendly
        df = df.rename(columns={"timestamp_s": "Timestamp", "intensity_ema": "Intensity_EMA_uW_cm2", "power_ema": "Power_EMA_W"})
        df[["Timestamp", "Intensity_EMA_uW_cm2", "Power_EMA_W"]].to_csv(path, index=False)
        self._set_status(f"Exported aligned CSV to {path}")

    # -------------------------
    # Analysis: Group Decay Overlay
    # -------------------------
    def on_plot_group_decay(self):
        if not self.groups:
            messagebox.showinfo("Plot", "No groups available.")
            return
        ema_span = int(self.overlay_ema_span.get())
        only_zero = self.only_yaw_roll_zero.get()
        normalize = self.normalize_to_1m.get()

        group_curves = []  # list of (group_name, time_hours, percent)
        for gid, g in self.groups.items():
            # Gather all SW3 in group
            sw3_ids = [fid for fid in g.file_ids if self.files.get(fid) and self.files[fid].kind == "sw3"]
            rows_all = []
            for sid in sw3_ids:
                rec = self.files[sid]
                try:
                    with open(rec.path, "rb") as fd:
                        buf = fd.read()
                    ensure_eegbin_imported()
                    try:
                        sweep = eegbin.load_eegbin3(buf, from_path=rec.path)
                    except Exception:
                        sweep = eegbin.load_eegbin2(buf, from_path=rec.path)
                except Exception as e:
                    messagebox.showwarning("Load error", f"Failed to load {rec.path}\n{e}")
                    continue
                for r in sweep.rows:
                    if hasattr(r, "valid") and not r.valid:
                        continue
                    if only_zero and not (getattr(r.coords, "yaw_deg", None) == 0 and getattr(r.coords, "roll_deg", None) == 0):
                        continue
                    if getattr(r, "timestamp", None) is None:
                        continue
                    I_wm2 = float(getattr(r.capture, "integral_result", 0.0) or 0.0)
                    if normalize:
                        dist_m = float(getattr(r.coords, "lin_mm", 0.0) or 0.0) / 1000.0
                        if dist_m > 0:
                            I_wm2 = I_wm2 * (dist_m**2)
                    I_uW_cm2 = I_wm2 * 100.0
                    rows_all.append( (float(r.timestamp) + g.offset_s, I_uW_cm2) )

            if not rows_all:
                continue
            rows_all.sort(key=lambda x: x[0])
            t = np.array([r[0] for r in rows_all], dtype=float)
            y = np.array([r[1] for r in rows_all], dtype=float)
            y_ema = ema_series(y, ema_span)
            # Normalize to peak = 100%
            if y_ema.max() > 0:
                y_pct = (y_ema / y_ema.max()) * 100.0
            else:
                y_pct = np.zeros_like(y_ema)
            t_hours = (t - t[0]) / 3600.0
            group_curves.append( (self.groups[gid].name, t_hours, y_pct) )

        if not group_curves:
            messagebox.showinfo("Plot", "No data to plot.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for name, th, yp in group_curves:
            ax.plot(th, yp, label=name, linewidth=2)
        ax.set_title("Decay Overlay by Group (EMA, normalized to each group’s peak)")
        ax.set_xlabel("Hours since group start")
        ax.set_ylabel("Intensity (% of peak)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        self._last_fig = fig
        plt.show(block=False)
        self._set_status("Plotted group decay overlay.")

    # -------------------------
    # Misc
    # -------------------------
    def on_save_last_figure(self):
        if self._last_fig is None:
            messagebox.showinfo("Save Figure", "No figure to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Figure", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        self._last_fig.savefig(path, dpi=150, bbox_inches="tight")
        self._set_status(f"Saved figure to {path}")

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _next_file_id(self) -> str:
        base = "F"
        i = 1
        while f"{base}{i:03d}" in self.files:
            i += 1
        return f"{base}{i:03d}"

    def _next_group_id(self) -> str:
        base = "G"
        i = 1
        while f"{base}{i:03d}" in self.groups:
            i += 1
        return f"{base}{i:03d}"


# -----------------------------
# Association editor dialog
# -----------------------------

class AssociationDialog(tk.Toplevel):
    def __init__(self, master: App, group: GroupRecord, sw3_list, power_list):
        super().__init__(master)
        self.title(f"Associations for '{group.name}'")
        self.resizable(True, True)
        self.group = group
        self.updated = False

        # Layout: SW3 list on left, Power list on right, with add/remove buttons
        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(pane); pane.add(left, weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=1)

        ttk.Label(left, text="SW3 files").pack(anchor="w")
        self.lb_sw3 = tk.Listbox(left, selectmode=tk.SINGLE, exportselection=False)
        for fid, fr in sw3_list:
            self.lb_sw3.insert(tk.END, f"{fr.label} ({fid})")
        self.sw3_ids = [fid for fid, _ in sw3_list]
        self.lb_sw3.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        ttk.Label(right, text="Power files").pack(anchor="w")
        self.lb_pow = tk.Listbox(right, selectmode=tk.MULTIPLE, exportselection=False)
        for fid, fr in power_list:
            self.lb_pow.insert(tk.END, f"{fr.label} ({fid})")
        self.pow_ids = [fid for fid, _ in power_list]
        self.lb_pow.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Buttons
        btns = ttk.Frame(self); btns.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btns, text="Map selected SW3 → selected Power", command=self._map_selected).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Remove mapping", command=self._remove_mapping).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=3)

        # Mapping preview
        self.tv_map = ttk.Treeview(self, columns=("sw3", "powers"), show="headings", height=6)
        self.tv_map.heading("sw3", text="SW3")
        self.tv_map.heading("powers", text="Power")
        self.tv_map.column("sw3", stretch=True, width=200)
        self.tv_map.column("powers", stretch=True, width=400)
        self.tv_map.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._refresh_mapping()

        self.lb_sw3.bind("<<ListboxSelect>>", lambda e: self._refresh_mapping())

    def _map_selected(self):
        idx_sw = self.lb_sw3.curselection()
        idx_pw = self.lb_pow.curselection()
        if not idx_sw or not idx_pw:
            return
        sw_id = self.sw3_ids[idx_sw[0]]
        pow_ids = [self.pow_ids[i] for i in idx_pw]
        s = self.group.associations.get(sw_id, set())
        s |= set(pow_ids)
        self.group.associations[sw_id] = s
        self.updated = True
        self._refresh_mapping()

    def _remove_mapping(self):
        idx_sw = self.lb_sw3.curselection()
        if not idx_sw:
            return
        sw_id = self.sw3_ids[idx_sw[0]]
        # remove any selected power from mapping
        idx_pw = self.lb_pow.curselection()
        if not idx_pw:
            # remove all
            if sw_id in self.group.associations:
                del self.group.associations[sw_id]
        else:
            s = self.group.associations.get(sw_id, set())
            for i in idx_pw:
                s.discard(self.pow_ids[i])
            if s:
                self.group.associations[sw_id] = s
            else:
                self.group.associations.pop(sw_id, None)
        self.updated = True
        self._refresh_mapping()

    def _refresh_mapping(self):
        for item in self.tv_map.get_children():
            self.tv_map.delete(item)
        # Show only the currently selected SW3 for convenience
        idx_sw = self.lb_sw3.curselection()
        if idx_sw:
            sw_id = self.sw3_ids[idx_sw[0]]
            powers = sorted(list(self.group.associations.get(sw_id, set())))
            self.tv_map.insert("", "end", values=(sw_id, ", ".join(powers)))
        else:
            # show all
            for sw_id, pset in self.group.associations.items():
                self.tv_map.insert("", "end", values=(sw_id, ", ".join(sorted(list(pset)))))


# -----------------------------
# Entrypoint
# -----------------------------

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
