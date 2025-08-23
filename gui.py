#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SW3 + Power Analyzer GUI

Layout
------
• Single vertical stack: Files (top) → Groups (middle) → Controls (bottom)

Capabilities
------------
• Load SW3/eegbin optical files and power CSV logs
• Label files; create groups; associate SW3↔Power (many-to-many)
• **Group-level trimming** (HH:MM:SS) at start and/or end; applied globally:
  - Intensity vs Time (file & group modes)
  - Optics vs Power (time series + correlation/scatter/export)
  - Group Decay Overlay
• Align streams by epoch timestamps (power resampled to N seconds; nearest-merge with tolerance)
• Per-analysis controls: show/hide points & EMA lines; independent alphas
• Friendly names in legends (no F###/G###), paired colors for OVP
• Sessions save/load everything (files, groups, associations, trims, settings)

Notes
-----
• No dependency on imgui; a tiny util shim is injected if util.py is missing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import colormaps as mpl_colormaps

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

# ------------------------------------------------------------------
# Optional util shim (so importing eegbin doesn't require imgui)
# ------------------------------------------------------------------

def _ensure_util_shim():
    import types
    if 'util' in sys.modules:
        return
    util = types.ModuleType('util')
    def inclusive_range(start, stop, step):
        if step == 0: return []
        out = []
        if step > 0:
            while start <= stop:
                out.append(start); start += step
        else:
            while start >= stop:
                out.append(start); start += step
        return out
    def do_editable_raw(preamble, value, units="", width=100):
        return (False, value)
    def do_editable(preamble, value, units="", width=100, enable=True):
        return value
    util.inclusive_range = inclusive_range
    util.do_editable_raw = do_editable_raw
    util.do_editable = do_editable
    sys.modules['util'] = util

_ensure_util_shim()

# lazily imported module
eegbin = None
_EXTRA_MODULE_PATHS: List[str] = []

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _get_cmap(name: str):
    try:
        return mpl_colormaps.get_cmap(name)
    except Exception:
        return plt.get_cmap(name)

def human_datetime(epoch_s: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_s))
    except Exception:
        return ""

def parse_hhmmss(s: str) -> int:
    """Return seconds (can be negative if string has a leading -)."""
    if s is None:
        return 0
    s = str(s).strip()
    if not s:
        return 0
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    parts = s.split(":")
    parts = [p for p in parts if p != ""]
    try:
        parts = [int(p) for p in parts]
    except Exception as e:
        raise ValueError(f"Invalid time format: {s}") from e
    if len(parts) == 2:
        return parts[0]*60 + parts[1]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    raise ValueError(f"Invalid time format: {s}")

def fmt_hhmmss(seconds: int) -> str:
    neg = seconds < 0
    s = abs(int(seconds))
    hh, rem = divmod(s, 3600)
    mm, ss = divmod(rem, 60)
    out = f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"-{out}" if neg else out

def _add_module_path(path: str):
    if path and path not in sys.path:
        sys.path.insert(0, path)
    if path and path not in _EXTRA_MODULE_PATHS:
        _EXTRA_MODULE_PATHS.append(path)

def ensure_eegbin_imported():
    """Add saved module paths; import eegbin (prompts for folder if needed)."""
    global eegbin
    if eegbin is not None:
        return
    import importlib
    for p in list(_EXTRA_MODULE_PATHS):
        _add_module_path(p)
    try:
        eegbin = importlib.import_module("eegbin")
        return
    except ModuleNotFoundError as e1:
        missing = getattr(e1, "name", "eegbin")
        if messagebox.askyesno("Locate module",
                               "Could not import '{}'.\nSelect the folder that contains 'eegbin.py' and 'util.py'."
                               .format(missing)):
            folder = filedialog.askdirectory(title="Select folder containing eegbin.py and util.py")
            if folder:
                _add_module_path(folder)
                try:
                    eegbin = importlib.import_module("eegbin")
                    return
                except Exception:
                    pass
        messagebox.showerror("Import error",
                             "Could not import 'eegbin'. Missing: {}\n\n"
                             "Tip: Put gui.py next to eegbin.py/util.py, or use Settings → Add Module Path…".format(missing))
        raise

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return np.asarray(arr, dtype=float)
    s = pd.Series(arr, dtype="float64")
    return s.ewm(span=int(span), adjust=False).mean().to_numpy()

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), float("nan")
    if scipy_stats is not None:
        r = scipy_stats.pearsonr(x, y)[0]
        slope, intercept, *_ = scipy_stats.linregress(x, y)
        return float(slope), float(intercept), float(r)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), 0.0
    r = float(np.corrcoef(x, y)[0,1])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), r

def merge_asof_seconds(df_left: pd.DataFrame, df_right: pd.DataFrame, tolerance_s: int) -> pd.DataFrame:
    left = df_left.copy()
    right = df_right.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], unit="s")
    right["timestamp"] = pd.to_datetime(right["timestamp"], unit="s")
    left = left.sort_values("timestamp")
    right = right.sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="nearest",
                           tolerance=pd.Timedelta(seconds=int(tolerance_s)))
    merged["timestamp_s"] = merged["timestamp"].astype("int64") // 10**9
    return merged

def _prepare_power_series(power_df: pd.DataFrame, tmin: float, tmax: float,
                          ema_span: int, resample_seconds: int) -> Optional[pd.DataFrame]:
    """Select window, resample, compute EMA; return columns ['timestamp','power_ema'] or None if empty."""
    mask = (power_df["Timestamp"] >= tmin) & (power_df["Timestamp"] <= tmax)
    cols = ["Timestamp", "W_Active"]
    pdf = power_df.loc[mask, cols].copy()
    if pdf.empty:
        return None
    # Build datetime index and resample
    pdf.loc[:, "timestamp"] = pd.to_datetime(pdf["Timestamp"], unit="s")
    pdf = pdf.set_index("timestamp").sort_index()
    pdf = pdf.resample(f"{int(resample_seconds)}s")["W_Active"].mean().to_frame("power")
    pdf.loc[:, "power_ema"] = pdf["power"].ewm(span=int(ema_span), adjust=False).mean()
    # Reset and create integer seconds column without overwriting dtype in place
    pdf = pdf.dropna(subset=["power_ema"]).reset_index()
    pdf = pdf.rename(columns={"timestamp": "timestamp_dt"})
    pdf["timestamp"] = (pdf["timestamp_dt"].astype("int64") // 10**9).astype("int64")
    pdf = pdf.drop(columns=["timestamp_dt"])
    return pdf[["timestamp", "power_ema"]]

def get_cmap_colors(n: int, cmap_name: str) -> List[Tuple[float, float, float, float]]:
    cmap = _get_cmap(cmap_name)
    xs = np.linspace(0.1, 0.9, max(1, n))
    return [cmap(x) for x in xs]

def _to_rgb(c):
    return np.array(mpl_colors.to_rgb(c), dtype=float)

def lighten(color, amount=0.6):
    """Lighten color by mixing with white; amount in [0..1]."""
    c = _to_rgb(color)
    return tuple(np.clip(1 - (1 - c) * (1 - amount), 0, 1))

def darken(color, amount=0.35):
    """Darken color by scaling toward black; amount in [0..1]."""
    c = _to_rgb(color)
    return tuple(np.clip(c * (1 - (1 - amount)), 0, 1))

# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclass
class FileRecord:
    file_id: str
    kind: str   # 'sw3' or 'power'
    path: str
    label: str
    meta: Dict[str, object] = field(default_factory=dict)

@dataclass
class GroupRecord:
    group_id: str
    name: str
    trim_start_s: int = 0
    trim_end_s: int = 0
    file_ids: Set[str] = field(default_factory=set)
    associations: Dict[str, Set[str]] = field(default_factory=dict)  # sw3_id -> set(power_id)

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SW3 + Power Analyzer")
        # Slightly taller default to avoid hiding top bars on small screens
        self.geometry("1320x900")
        self.minsize(960, 640)

        self.files: Dict[str, FileRecord] = {}
        self.groups: Dict[str, GroupRecord] = {}

        # display maps for friendly names
        self._sw3_display_to_id: Dict[str, str] = {}
        self._group_display_to_id: Dict[str, str] = {}

        self.power_column_map = {
            "timestamp": ["Timestamp", "timestamp", "epoch", "Epoch", "time", "Time"],
            "watts": ["W_Active", "watts", "Watts", "Power_W", "W"]
        }

        self._init_vars()
        self._build_ui()

        # figure registry and aligned cache
        self._figs: Dict[str, List[plt.Figure]] = {"ivt": [], "ovp": [], "gdo": [], "corr": []}
        self._aligned_cache: Optional[pd.DataFrame] = None

    # ---- control vars ----
    def _init_vars(self):
        # analysis params
        self.intensity_ema_span = tk.IntVar(self, value=31)
        self.power_ema_span     = tk.IntVar(self, value=31)
        self.overlay_ema_span   = tk.IntVar(self, value=31)
        self.trim_start_s       = tk.IntVar(self, value=0)   # per-IVT extra trim
        self.trim_end_s         = tk.IntVar(self, value=0)
        self.align_tolerance_s  = tk.IntVar(self, value=2)
        self.ccf_max_lag_s      = tk.IntVar(self, value=180)
        self.resample_seconds   = tk.IntVar(self, value=1)

        # filters
        self.normalize_to_1m    = tk.BooleanVar(self, value=True)
        self.only_yaw_roll_zero = tk.BooleanVar(self, value=True)

        # per-analysis style
        self.ivt_show_points    = tk.BooleanVar(self, value=True)
        self.ivt_show_ema       = tk.BooleanVar(self, value=True)
        self.ivt_point_alpha    = tk.DoubleVar(self, value=0.25)
        self.ivt_line_alpha     = tk.DoubleVar(self, value=1.0)

        self.ovp_show_points    = tk.BooleanVar(self, value=False)
        self.ovp_show_int_ema   = tk.BooleanVar(self, value=True)
        self.ovp_show_pow_ema   = tk.BooleanVar(self, value=True)
        self.ovp_point_alpha    = tk.DoubleVar(self, value=0.15)
        self.ovp_line_alpha     = tk.DoubleVar(self, value=1.0)

        self.gdo_show_points    = tk.BooleanVar(self, value=False)
        self.gdo_show_ema       = tk.BooleanVar(self, value=True)
        self.gdo_point_alpha    = tk.DoubleVar(self, value=0.15)
        self.gdo_line_alpha     = tk.DoubleVar(self, value=1.0)

        # combos (friendly)
        self.ivt_sw3_display    = tk.StringVar(self, value="")
        self.ivt_group_display  = tk.StringVar(self, value="")
        self.ovp_group_display  = tk.StringVar(self, value="")

        # source mode
        self.source_mode        = tk.StringVar(self, value="file")  # 'file' or 'group'
        self.combine_group_sw3  = tk.BooleanVar(self, value=True)

        # UI
        self.controls_visible   = tk.BooleanVar(self, value=True)

    # ---- UI ----
    def _build_ui(self):
        self._build_menu()

        # Single vertical panedwindow with three panes
        self._paned = ttk.Panedwindow(self, orient=tk.VERTICAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        files_container = ttk.Frame(self._paned)
        groups_container = ttk.Frame(self._paned)
        self._controls_frame = ttk.Frame(self._paned)

        self._paned.add(files_container, weight=3)
        self._paned.add(groups_container, weight=2)
        self._paned.add(self._controls_frame, weight=1)

        self._build_files_frame(files_container)
        self._build_groups_frame(groups_container)
        self._build_controls(self._controls_frame)

        # status
        self.status_var = tk.StringVar(self, value="Ready")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, side=tk.BOTTOM)

        # initial vertical sash positions (favor controls) with minimum Files height
        self.after(120, self._set_initial_sashes)

    def _build_menu(self):
        menubar = tk.Menu(self)

        # File
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Add SW3…", command=self.on_add_sw3_files)
        file_menu.add_command(label="Add Power CSV…", command=self.on_add_power_files)
        file_menu.add_separator()
        file_menu.add_command(label="Reload Files", command=self.on_reload_all_files)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session…", command=self.on_save_session)
        file_menu.add_command(label="Load Session…", command=self.on_load_session)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        # Settings
        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(label="Power Columns…", command=self.on_set_power_columns)
        settings_menu.add_separator()
        settings_menu.add_command(label="Add Module Path…", command=self.on_add_module_path)
        settings_menu.add_command(label="List Module Paths…", command=self.on_list_module_paths)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        # View
        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_checkbutton(label="Show Controls Panel", variable=self.controls_visible, command=self.on_toggle_controls)
        view_menu.add_command(label="Compact Controls Height", command=self.on_compact_controls_height)
        view_menu.add_command(label="Favor Controls Section", command=self.on_favor_controls)
        view_menu.add_separator()
        view_menu.add_command(label="Maximize Files Section", command=self.on_maximize_files)
        view_menu.add_command(label="Maximize Groups Section", command=self.on_maximize_groups)
        view_menu.add_command(label="Maximize Controls Section", command=self.on_maximize_controls)
        menubar.add_cascade(label="View", menu=view_menu)

        # Help
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "SW3 + Power Analyzer\nAligns SW3/eegbin optical data with power logs by epoch time."
        ))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_files_frame(self, parent):
        frm = ttk.LabelFrame(parent, text="Files")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # --- TOP button bar (so it's always visible even if the pane is short) ---
        btns = ttk.Frame(frm); btns.pack(fill=tk.X, padx=3, pady=(6,3))
        ttk.Button(btns, text="Add SW3…", command=self.on_add_sw3_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Add Power CSV…", command=self.on_add_power_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Rename…", command=self.on_rename_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Remove", command=self.on_remove_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Assign to Group…", command=self.on_assign_files_to_group).pack(side=tk.LEFT, padx=2)
        ttk.Separator(btns, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(btns, text="Reload Selected", command=lambda: self.on_reload_all_files(only_selected=True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Reload All", command=self.on_reload_all_files).pack(side=tk.LEFT, padx=2)

        # --- Treeview container BELOW the button bar ---
        container = ttk.Frame(frm); container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(3,6))

        cols = ("label", "kind", "first", "last", "count", "path")
        self.tv_files = ttk.Treeview(container, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.tv_files.heading(c, text=c.capitalize())
            w = 140 if c not in ("path", "label") else 220
            self.tv_files.column(c, width=w, anchor="w", stretch=True)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tv_files.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tv_files.xview)
        self.tv_files.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tv_files.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1); container.columnconfigure(0, weight=1)

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
        ttk.Button(top, text="Set Trim…", command=self.on_set_group_trim).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Clear Trim", command=self.on_clear_group_trim).pack(side=tk.LEFT, padx=2)

        container = ttk.Frame(frm); container.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        cols = ("name", "trim", "sw3_count", "power_count")
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
        container.rowconfigure(0, weight=1); container.columnconfigure(0, weight=1)
        self.tv_groups.bind("<<TreeviewSelect>>", lambda e: self._on_group_selection_changed())

    def _build_controls(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.X, expand=False, padx=6, pady=6)

        # ---- Intensity vs Time ----
        tab_ivt = ttk.Frame(nb); nb.add(tab_ivt, text="Intensity vs Time")
        r0 = ttk.Frame(tab_ivt); r0.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(r0, text="Source:").pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(r0, text="File", value="file", variable=self.source_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(r0, text="Group", value="group", variable=self.source_mode).pack(side=tk.LEFT)
        ttk.Label(r0, text="SW3 File:").pack(side=tk.LEFT, padx=(12,3))
        self.cb_ivt_sw3 = ttk.Combobox(r0, state="readonly", width=34, textvariable=self.ivt_sw3_display)
        self.cb_ivt_sw3.pack(side=tk.LEFT, padx=3)
        ttk.Label(r0, text="Group:").pack(side=tk.LEFT, padx=(12,3))
        self.cb_ivt_group = ttk.Combobox(r0, state="readonly", width=28, textvariable=self.ivt_group_display)
        self.cb_ivt_group.pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(tab_ivt, text="Combine group SW3s into one series", variable=self.combine_group_sw3).pack(anchor="w", padx=10)

        r1 = ttk.Frame(tab_ivt); r1.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(r1, text="EMA span:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(r1, textvariable=self.intensity_ema_span, width=6).pack(side=tk.LEFT, padx=3)
        ttk.Label(r1, text="Trim start:").pack(side=tk.LEFT, padx=6)
        ttk.Entry(r1, textvariable=self.trim_start_s, width=8).pack(side=tk.LEFT, padx=3); ttk.Label(r1, text="s").pack(side=tk.LEFT)
        ttk.Label(r1, text="Trim end:").pack(side=tk.LEFT, padx=6)
        ttk.Entry(r1, textvariable=self.trim_end_s, width=8).pack(side=tk.LEFT, padx=3); ttk.Label(r1, text="s").pack(side=tk.LEFT)
        ttk.Checkbutton(r1, text="Normalize to 1m", variable=self.normalize_to_1m).pack(side=tk.LEFT, padx=12)
        ttk.Checkbutton(r1, text="Yaw=0 & Roll=0 only", variable=self.only_yaw_roll_zero).pack(side=tk.LEFT, padx=6)

        r2 = ttk.Frame(tab_ivt); r2.pack(fill=tk.X, padx=6, pady=6)
        ttk.Checkbutton(r2, text="Show points", variable=self.ivt_show_points).pack(side=tk.LEFT, padx=(0,8))
        ttk.Checkbutton(r2, text="Show EMA", variable=self.ivt_show_ema).pack(side=tk.LEFT, padx=(0,8))
        ttk.Label(r2, text="Points α:").pack(side=tk.LEFT); tk.Scale(r2, variable=self.ivt_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=6)
        ttk.Label(r2, text="EMA α:").pack(side=tk.LEFT); tk.Scale(r2, variable=self.ivt_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=6)

        r3 = ttk.Frame(tab_ivt); r3.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(r3, text="Plot", command=self.on_plot_intensity_vs_time).pack(side=tk.LEFT, padx=3)
        ttk.Button(r3, text="Save Figure…", command=self.on_save_last_figure).pack(side=tk.LEFT, padx=3)
        ttk.Button(r3, text="Close IVT Plots", command=lambda: self._close_plots('ivt')).pack(side=tk.LEFT, padx=10)
        ttk.Label(tab_ivt, text="Trim accepts seconds (e.g., 90) or HH:MM:SS (e.g., 00:01:30).").pack(anchor="w", padx=10)

        # ---- Optics vs Power ----
        tab_ovp = ttk.Frame(nb); nb.add(tab_ovp, text="Optics vs Power")
        o1 = ttk.Frame(tab_ovp); o1.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(o1, text="Group:").pack(side=tk.LEFT, padx=3)
        self.cb_group = ttk.Combobox(o1, state="readonly", width=38, textvariable=self.ovp_group_display)
        self.cb_group.pack(side=tk.LEFT, padx=3)
        ttk.Label(o1, text="Intensity EMA:").pack(side=tk.LEFT, padx=6)
        ttk.Entry(o1, textvariable=self.intensity_ema_span, width=6).pack(side=tk.LEFT)
        ttk.Label(o1, text="Power EMA:").pack(side=tk.LEFT, padx=6)
        ttk.Entry(o1, textvariable=self.power_ema_span, width=6).pack(side=tk.LEFT)
        ttk.Label(o1, text="Align tol:").pack(side=tk.LEFT, padx=6)
        ttk.Entry(o1, textvariable=self.align_tolerance_s, width=6).pack(side=tk.LEFT); ttk.Label(o1, text="s").pack(side=tk.LEFT)

        o2 = ttk.Frame(tab_ovp); o2.pack(fill=tk.X, padx=6, pady=6)
        ttk.Checkbutton(o2, text="Show points", variable=self.ovp_show_points).pack(side=tk.LEFT, padx=(0,8))
        ttk.Checkbutton(o2, text="Show Intensity EMA", variable=self.ovp_show_int_ema).pack(side=tk.LEFT, padx=(0,8))
        ttk.Checkbutton(o2, text="Show Power EMA", variable=self.ovp_show_pow_ema).pack(side=tk.LEFT, padx=(0,8))
        ttk.Label(o2, text="Points α:").pack(side=tk.LEFT); tk.Scale(o2, variable=self.ovp_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=6)
        ttk.Label(o2, text="Lines α:").pack(side=tk.LEFT); tk.Scale(o2, variable=self.ovp_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=6)

        o3 = ttk.Frame(tab_ovp); o3.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(o3, text="Analyze Group (Time Series)", command=self.on_analyze_group).pack(side=tk.LEFT, padx=3)
        ttk.Button(o3, text="Correlation & Scatter", command=self.on_corr_and_scatter).pack(side=tk.LEFT, padx=3)
        ttk.Button(o3, text="Export Aligned CSV…", command=self.on_export_aligned_csv).pack(side=tk.LEFT, padx=3)
        ttk.Button(o3, text="Save Figure…", command=self.on_save_last_figure).pack(side=tk.LEFT, padx=3)
        ttk.Button(o3, text="Close OVP Plots", command=lambda: self._close_plots('ovp', also=('corr',))).pack(side=tk.LEFT, padx=10)

        # ---- Group Decay Overlay ----
        tab_gdo = ttk.Frame(nb); nb.add(tab_gdo, text="Group Decay Overlay")
        g1 = ttk.Frame(tab_gdo); g1.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(g1, text="EMA span:").pack(side=tk.LEFT, padx=3)
        ttk.Entry(g1, textvariable=self.overlay_ema_span, width=6).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(g1, text="Show points", variable=self.gdo_show_points).pack(side=tk.LEFT, padx=(12,8))
        ttk.Checkbutton(g1, text="Show EMA", variable=self.gdo_show_ema).pack(side=tk.LEFT, padx=(0,8))
        ttk.Label(g1, text="Points α:").pack(side=tk.LEFT); tk.Scale(g1, variable=self.gdo_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=6)
        ttk.Label(g1, text="Lines α:").pack(side=tk.LEFT); tk.Scale(g1, variable=self.gdo_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT, padx=6)
        ttk.Button(g1, text="Close GDO Plots", command=lambda: self._close_plots('gdo')).pack(side=tk.RIGHT, padx=6)

        g2 = ttk.Frame(tab_gdo); g2.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(g2, text="Plot Selected Groups", command=lambda: self.on_plot_group_decay(selected_only=True)).pack(side=tk.LEFT, padx=6)
        ttk.Button(g2, text="Plot All Groups", command=lambda: self.on_plot_group_decay(selected_only=False)).pack(side=tk.LEFT, padx=6)
        ttk.Label(tab_gdo, text="Pick groups below (default: all).").pack(anchor="w", padx=10)

        frame_list = ttk.Frame(tab_gdo); frame_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.lb_groups_select = tk.Listbox(frame_list, selectmode=tk.EXTENDED, exportselection=False)
        vs = ttk.Scrollbar(frame_list, orient="vertical", command=self.lb_groups_select.yview)
        self.lb_groups_select.configure(yscroll=vs.set)
        self.lb_groups_select.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        frame_list.rowconfigure(0, weight=1); frame_list.columnconfigure(0, weight=1)
        bsel = ttk.Frame(tab_gdo); bsel.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(bsel, text="Select All", command=self._select_all_groups).pack(side=tk.LEFT, padx=3)
        ttk.Button(bsel, text="Select None", command=lambda: self.lb_groups_select.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=3)

    # ---- View helpers ----
    def _set_initial_sashes(self):
        """Favor Controls, but guarantee Files has enough height to show its button bar."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height() or self.winfo_height() or 800
            MIN_FILES = 220   # ensure top button row stays visible
            MIN_GROUPS = 180
            # initial proportional split
            pos0 = int(total_h * 0.25)
            pos1 = int(total_h * 0.55)
            # enforce minimum heights
            pos0 = max(MIN_FILES, pos0)
            pos1 = max(pos0 + MIN_GROUPS, pos1)
            self._paned.sashpos(0, pos0)
            self._paned.sashpos(1, pos1)
        except Exception:
            pass

    def on_toggle_controls(self):
        vis = self.controls_visible.get()
        try:
            panes = self._paned.panes()
            if vis and str(self._controls_frame) not in panes:
                self._paned.add(self._controls_frame, weight=1)
            elif not vis and str(self._controls_frame) in panes:
                self._paned.forget(self._controls_frame)
        except Exception:
            pass

    def on_compact_controls_height(self):
        """Shrink the controls section to a compact height (~220 px)."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            # Ensure groups keeps at least ~200 px and controls ~220 px
            self._paned.sashpos(0, 160)
            self._paned.sashpos(1, max(360, total_h - 220))
        except Exception:
            pass

    def on_favor_controls(self):
        """Give more height to the Controls section (Files ~25%, Groups ~30%, Controls ~45%)."""
        try:
            self._set_initial_sashes()
        except Exception:
            pass

    def on_maximize_files(self):
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, total_h - 40)
            self._paned.sashpos(1, total_h - 20)
        except Exception:
            pass

    def on_maximize_groups(self):
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, 120)
            self._paned.sashpos(1, total_h - 40)
        except Exception:
            pass

    def on_maximize_controls(self):
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, 120)
            self._paned.sashpos(1, total_h - 10)
        except Exception:
            pass

    # ---- File ops ----
    def on_add_sw3_files(self):
        paths = filedialog.askopenfilenames(title="Add SW3/eegbin Files",
                                            filetypes=[("SW3/eegbin", "*.sw3 *.eegbin *.bin *.*")])
        if not paths: return
        ensure_eegbin_imported()
        n=0
        for path in paths:
            try:
                with open(path, "rb") as fd: buf = fd.read()
                try: sweep = eegbin.load_eegbin3(buf, from_path=path)
                except Exception: sweep = eegbin.load_eegbin2(buf, from_path=path)
                meta = self._meta_from_sweep(sweep)
                fid = self._next_file_id()
                self.files[fid] = FileRecord(fid, "sw3", path, os.path.basename(path), meta)
                n += 1
            except Exception as e:
                traceback.print_exc(); messagebox.showwarning("Load error", f"Failed to load {path}\n{e}")
        if n:
            self._refresh_files_tv()
            self._refresh_display_mappings()
            self._set_status(f"Added {n} SW3 file(s).")

    def on_add_power_files(self):
        paths = filedialog.askopenfilenames(title="Add Power CSV Files",
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not paths: return
        n=0
        for path in paths:
            try:
                _, meta = self._load_power_csv_with_meta(path)
                fid = self._next_file_id()
                self.files[fid] = FileRecord(fid, "power", path, os.path.basename(path), meta)
                n+=1
            except Exception as e:
                traceback.print_exc(); messagebox.showwarning("Load error", f"Failed to load {path}\n{e}")
        if n:
            self._refresh_files_tv()
            self._refresh_display_mappings()
            self._set_status(f"Added {n} power file(s).")

    def _meta_from_sweep(self, sweep) -> Dict[str, object]:
        rows = [r for r in sweep.rows if getattr(r, "timestamp", None) is not None]
        ts = [r.timestamp for r in rows]
        first = float(min(ts)) if ts else None
        last = float(max(ts)) if ts else None
        return {"first_ts": first, "last_ts": last, "count": len(rows), "lamp_name": getattr(sweep, "lamp_name", "")}

    def _load_power_csv_with_meta(self, path: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
        df = pd.read_csv(path)
        ts_col = self._detect_col(df.columns, self.power_column_map["timestamp"])
        w_col  = self._detect_col(df.columns, self.power_column_map["watts"])
        if not ts_col or not w_col:
            raise ValueError(f"CSV must include timestamp + watts. Candidates tried: {self.power_column_map}. Found: {list(df.columns)}")
        df = df[[ts_col, w_col]].rename(columns={ts_col:"Timestamp", w_col:"W_Active"})
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
        df["W_Active"]  = pd.to_numeric(df["W_Active"], errors="coerce")
        df = df.dropna(subset=["Timestamp","W_Active"])
        meta = {"first_ts": float(df["Timestamp"].min()) if not df.empty else None,
                "last_ts":  float(df["Timestamp"].max()) if not df.empty else None,
                "count": int(len(df))}
        return df, meta

    def _detect_col(self, cols, candidates: List[str]) -> Optional[str]:
        m = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in m:
                return m[cand.lower()]
        return None

    def _refresh_files_tv(self):
        self.tv_files.delete(*self.tv_files.get_children())
        for fid, rec in self.files.items():
            first = human_datetime(rec.meta.get("first_ts")) if rec.meta.get("first_ts") else ""
            last  = human_datetime(rec.meta.get("last_ts")) if rec.meta.get("last_ts") else ""
            count = rec.meta.get("count","")
            self.tv_files.insert("", "end", iid=fid, values=(rec.label, rec.kind, first, last, count, rec.path))

    def _refresh_groups_tv(self):
        self.tv_groups.delete(*self.tv_groups.get_children())
        for gid, g in self.groups.items():
            sw3c = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "sw3")
            powc = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "power")
            self.tv_groups.insert("", "end", iid=gid, values=(g.name, self._format_group_trim(g), sw3c, powc))
        self._refresh_group_select_list()
        self._refresh_display_mappings()

    def _refresh_group_select_list(self):
        self.lb_groups_select.delete(0, tk.END)
        for gid, g in self.groups.items():
            self.lb_groups_select.insert(tk.END, f"{gid}  {g.name}")

    def _refresh_display_mappings(self):
        # files (sw3)
        sw3_ids = [fid for fid, fr in self.files.items() if fr.kind=="sw3"]
        labels = [self.files[fid].label for fid in sw3_ids]
        counts = {}
        for lab in labels: counts[lab] = counts.get(lab,0)+1
        self._sw3_display_to_id.clear()
        sw3_display = []
        for fid in sw3_ids:
            lab = self.files[fid].label
            disp = lab if counts[lab]==1 else f"{lab} [{fid}]"
            self._sw3_display_to_id[disp] = fid
            sw3_display.append(disp)
        self.cb_ivt_sw3["values"] = sw3_display
        # groups
        self._group_display_to_id.clear()
        group_display = []
        name_counts = {}
        for g in self.groups.values():
            name_counts[g.name] = name_counts.get(g.name,0)+1
        for gid, g in self.groups.items():
            disp = g.name if name_counts[g.name]==1 else f"{g.name} [{gid}]"
            self._group_display_to_id[disp] = gid
            group_display.append(disp)
        self.cb_ivt_group["values"] = group_display
        self.cb_group["values"] = group_display

    def _display_to_sw3_id(self, disp: str) -> Optional[str]:
        return self._sw3_display_to_id.get(disp)

    def _display_to_group_id(self, disp: str) -> Optional[str]:
        return self._group_display_to_id.get(disp)

    # ---- File actions ----
    def on_rename_file(self):
        sel = self.tv_files.selection()
        if not sel: return
        if len(sel) > 1:
            messagebox.showinfo("Rename", "Select a single file to rename."); return
        fid = sel[0]
        rec = self.files[fid]
        new = simpledialog.askstring("Rename", "New label:", initialvalue=rec.label, parent=self)
        if new:
            rec.label = new.strip()
            self._refresh_files_tv()
            self._refresh_display_mappings()

    def on_remove_files(self):
        sel = list(self.tv_files.selection())
        if not sel: return
        if not messagebox.askyesno("Remove", f"Remove {len(sel)} file(s) from the project?"):
            return
        for fid in sel:
            for g in self.groups.values():
                g.file_ids.discard(fid)
                g.associations.pop(fid, None)
                for k in list(g.associations.keys()):
                    g.associations[k].discard(fid)
                    if not g.associations[k]:
                        g.associations.pop(k, None)
            self.files.pop(fid, None)
        self._refresh_files_tv()
        self._refresh_groups_tv()

    def on_assign_files_to_group(self):
        sel = list(self.tv_files.selection())
        if not sel:
            messagebox.showinfo("Assign", "Select files (SW3 and/or Power) to assign to a group."); return
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        for fid in sel: g.file_ids.add(fid)
        self._refresh_groups_tv()
        self._set_status(f"Assigned {len(sel)} file(s) to group '{g.name}'.")

    # ---- Groups ----
    def on_new_group(self):
        name = simpledialog.askstring("New Group", "Group name:", parent=self)
        if not name: return
        gid = self._next_group_id()
        self.groups[gid] = GroupRecord(gid, name.strip())
        self._refresh_groups_tv()
        self._set_status(f"Created group '{name}'.")

    def on_rename_group(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        new = simpledialog.askstring("Rename Group", "New name:", initialvalue=g.name, parent=self)
        if new:
            g.name = new.strip()
            self._refresh_groups_tv()

    def on_delete_group(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        if not messagebox.askyesno("Delete Group", "Delete this group (files remain in project)?"):
            return
        del self.groups[gid]
        self._refresh_groups_tv()

    def on_edit_associations(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        sw3s = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind=="sw3"]
        pows = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind=="power"]
        if not sw3s or not pows:
            messagebox.showinfo("Associations", "Assign at least one SW3 and one Power file to the group first."); return
        dlg = AssociationDialog(self, g, sw3s, pows); self.wait_window(dlg)
        if dlg.updated: self._set_status("Updated associations.")

    def _format_group_trim(self, g) -> str:
        s = fmt_hhmmss(max(0, int(getattr(g, "trim_start_s", 0))))
        e = fmt_hhmmss(max(0, int(getattr(g, "trim_end_s", 0))))
        return "—" if (s=="00:00:00" and e=="00:00:00") else f"{s} | {e}"

    def _group_time_window(self, g: GroupRecord) -> Optional[Tuple[float, float]]:
        """Compute trimmed time window [tmin,tmax] from group's SW3 files and trims.
           Returns None if window degenerates or no SW3 meta is available.
        """
        sw3_times = []
        for fid in g.file_ids:
            fr = self.files.get(fid)
            if fr and fr.kind == "sw3":
                ft = fr.meta.get("first_ts"); lt = fr.meta.get("last_ts")
                if isinstance(ft, (int, float)) and isinstance(lt, (int, float)):
                    sw3_times.append((float(ft), float(lt)))
        if not sw3_times:
            return None
        base_start = min(a for a, _ in sw3_times)
        base_end   = max(b for _, b in sw3_times)
        tmin = base_start + max(0, int(getattr(g, "trim_start_s", 0)))
        tmax = base_end   - max(0, int(getattr(g, "trim_end_s", 0)))
        if tmax <= tmin:
            return None
        return (tmin, tmax)

    def _filter_rows_by_window(self, rows: List[Tuple[float, float]], tmin: Optional[float], tmax: Optional[float]):
        """Rows is a list of (timestamp, value). Return rows within [tmin, tmax]."""
        if tmin is None or tmax is None:
            return rows
        return [rv for rv in rows if (rv[0] >= tmin and rv[0] <= tmax)]

    def on_set_group_trim(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        # Prompt for start and end trim (HH:MM:SS)
        top = tk.Toplevel(self); top.title(f"Set Trim — {g.name}")
        ttk.Label(top, text="Start trim (HH:MM:SS):").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        e_start = ttk.Entry(top, width=16)
        e_start.insert(0, fmt_hhmmss(max(0, int(getattr(g, "trim_start_s", 0)))))
        e_start.grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(top, text="End trim (HH:MM:SS):").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        e_end = ttk.Entry(top, width=16)
        e_end.insert(0, fmt_hhmmss(max(0, int(getattr(g, "trim_end_s", 0)))))
        e_end.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        btns = ttk.Frame(top); btns.grid(row=2, column=0, columnspan=2, sticky="e", padx=6, pady=6)
        def save():
            try:
                g.trim_start_s = max(0, parse_hhmmss(e_start.get()))
                g.trim_end_s   = max(0, parse_hhmmss(e_end.get()))
            except Exception as ex:
                messagebox.showwarning("Invalid", str(ex)); return
            top.destroy()
            self._refresh_groups_tv()
        ttk.Button(btns, text="OK", command=save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=4)

    def on_clear_group_trim(self):
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        g.trim_start_s = 0; g.trim_end_s = 0
        self._refresh_groups_tv()

    def _ensure_group_selected_or_prompt(self) -> Optional[str]:
        sel = self.tv_groups.selection()
        if sel: return sel[0]
        if not self.groups:
            messagebox.showinfo("Groups", "Create a group first."); return None
        choices = list(self.groups.keys()); labels = [self.groups[c].name for c in choices]
        idx = simpledialog.askinteger("Select Group", "Enter group number:\n"+ "\n".join(f"{i+1}. {labels[i]}" for i in range(len(labels))),
                                      minvalue=1, maxvalue=len(labels), parent=self)
        return choices[idx-1] if idx else None

    def _on_group_selection_changed(self):
        pass  # selection is used via friendly comboboxes

    # ---- Settings ----
    def on_set_power_columns(self):
        top = tk.Toplevel(self); top.title("Power Columns Map"); top.resizable(True, False)
        ttk.Label(top, text="Candidate names for timestamp column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        e1 = ttk.Entry(top, width=60); e1.insert(0, ", ".join(self.power_column_map["timestamp"])); e1.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(top, text="Candidate names for watts column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        e2 = ttk.Entry(top, width=60); e2.insert(0, ", ".join(self.power_column_map["watts"])); e2.pack(fill=tk.X, padx=6, pady=3)
        def save():
            self.power_column_map["timestamp"] = [s.strip() for s in e1.get().split(",") if s.strip()]
            self.power_column_map["watts"] = [s.strip() for s in e2.get().split(",") if s.strip()]
            top.destroy()
        b = ttk.Frame(top); b.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(b, text="OK", command=save).pack(side=tk.RIGHT, padx=3)
        ttk.Button(b, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=3)

    def on_add_module_path(self):
        folder = filedialog.askdirectory(title="Add folder to Python module search path (eegbin/util)")
        if not folder: return
        _add_module_path(folder); self._set_status(f"Added module path: {folder}")

    def on_list_module_paths(self):
        if not _EXTRA_MODULE_PATHS:
            messagebox.showinfo("Module Paths", "No extra module paths added."); return
        msg = "Extra module search paths:\n\n" + "\n".join("• "+p for p in _EXTRA_MODULE_PATHS)
        messagebox.showinfo("Module Paths", msg)

    # ---- Reload ----
    def on_reload_all_files(self, only_selected: bool=False):
        targets = list(self.tv_files.selection()) if only_selected else list(self.files.keys())
        if only_selected and not targets:
            messagebox.showinfo("Reload", "No files selected."); return
        ensure_eegbin_imported()
        ok=err=0
        for fid in targets:
            rec = self.files.get(fid)
            if not rec: continue
            try:
                if rec.kind=="sw3":
                    with open(rec.path, "rb") as fd: buf = fd.read()
                    try: sweep = eegbin.load_eegbin3(buf, from_path=rec.path)
                    except Exception: sweep = eegbin.load_eegbin2(buf, from_path=rec.path)
                    rec.meta = self._meta_from_sweep(sweep)
                else:
                    _, rec.meta = self._load_power_csv_with_meta(rec.path)
                ok += 1
            except Exception:
                err += 1
        self._refresh_files_tv(); self._refresh_display_mappings()
        self._set_status(f"Reloaded {ok} file(s){'; errors: '+str(err) if err else ''}.")

    # ---- Session I/O ----
    def on_save_session(self):
        path = filedialog.asksaveasfilename(title="Save Session", defaultextension=".json",
                                            filetypes=[("Session JSON","*.json")])
        if not path: return
        data = {
            "power_column_map": self.power_column_map,
            "module_paths": list(_EXTRA_MODULE_PATHS),
            "files": [asdict(fr) for fr in self.files.values()],
            "groups": [{
                "group_id": g.group_id, "name": g.name,
                "trim_start_s": getattr(g, "trim_start_s", 0), "trim_end_s": getattr(g, "trim_end_s", 0),
                "file_ids": list(g.file_ids),
                "associations": {k: list(v) for k, v in g.associations.items()},
            } for g in self.groups.values()],
            "controls": {
                "intensity_ema_span": self.intensity_ema_span.get(),
                "power_ema_span": self.power_ema_span.get(),
                "overlay_ema_span": self.overlay_ema_span.get(),
                "trim_start_s": self.trim_start_s.get(),
                "trim_end_s": self.trim_end_s.get(),
                "align_tolerance_s": self.align_tolerance_s.get(),
                "ccf_max_lag_s": self.ccf_max_lag_s.get(),
                "resample_seconds": self.resample_seconds.get(),
                "normalize_to_1m": self.normalize_to_1m.get(),
                "only_yaw_roll_zero": self.only_yaw_roll_zero.get(),
                "ivt_show_points": self.ivt_show_points.get(),
                "ivt_show_ema": self.ivt_show_ema.get(),
                "ivt_point_alpha": float(self.ivt_point_alpha.get()),
                "ivt_line_alpha": float(self.ivt_line_alpha.get()),
                "ovp_show_points": self.ovp_show_points.get(),
                "ovp_show_int_ema": self.ovp_show_int_ema.get(),
                "ovp_show_pow_ema": self.ovp_show_pow_ema.get(),
                "ovp_point_alpha": float(self.ovp_point_alpha.get()),
                "ovp_line_alpha": float(self.ovp_line_alpha.get()),
                "gdo_show_points": self.gdo_show_points.get(),
                "gdo_show_ema": self.gdo_show_ema.get(),
                "gdo_point_alpha": float(self.gdo_point_alpha.get()),
                "gdo_line_alpha": float(self.gdo_line_alpha.get()),
                "source_mode": self.source_mode.get(),
                "combine_group_sw3": self.combine_group_sw3.get(),
                "ivt_sw3_display": self.ivt_sw3_display.get(),
                "ivt_group_display": self.ivt_group_display.get(),
                "ovp_group_display": self.ovp_group_display.get(),
                "controls_visible": self.controls_visible.get(),
            }
        }
        with open(path, "w") as fd: json.dump(data, fd, indent=2)
        self._set_status(f"Saved session to {path}")

    def on_load_session(self):
        path = filedialog.askopenfilename(title="Load Session",
                                          filetypes=[("Session JSON","*.json"),("All files","*.*")])
        if not path: return
        with open(path, "r") as fd: data = json.load(fd)
        _EXTRA_MODULE_PATHS.clear()
        for p in data.get("module_paths", []): _add_module_path(p)
        self.power_column_map = data.get("power_column_map", self.power_column_map)
        self.files = {fr["file_id"]: FileRecord(**fr) for fr in data.get("files", [])}
        self.groups = {}
        for g in data.get("groups", []):
            gr = GroupRecord(group_id=g["group_id"], name=g["name"],
                             trim_start_s=g.get("trim_start_s", 0), trim_end_s=g.get("trim_end_s", 0))
            gr.file_ids = set(g.get("file_ids", []))
            gr.associations = {k: set(v) for k,v in g.get("associations", {}).items()}
            self.groups[gr.group_id] = gr
        c = data.get("controls", {})
        def set_if(k,var):
            if k in c: var.set(c[k])
        for k,var in [
            ("intensity_ema_span", self.intensity_ema_span),
            ("power_ema_span", self.power_ema_span),
            ("overlay_ema_span", self.overlay_ema_span),
            ("trim_start_s", self.trim_start_s),
            ("trim_end_s", self.trim_end_s),
            ("align_tolerance_s", self.align_tolerance_s),
            ("ccf_max_lag_s", self.ccf_max_lag_s),
            ("resample_seconds", self.resample_seconds),
            ("normalize_to_1m", self.normalize_to_1m),
            ("only_yaw_roll_zero", self.only_yaw_roll_zero),
            ("ivt_show_points", self.ivt_show_points),
            ("ivt_show_ema", self.ivt_show_ema),
            ("ivt_point_alpha", self.ivt_point_alpha),
            ("ivt_line_alpha", self.ivt_line_alpha),
            ("ovp_show_points", self.ovp_show_points),
            ("ovp_show_int_ema", self.ovp_show_int_ema),
            ("ovp_show_pow_ema", self.ovp_show_pow_ema),
            ("ovp_point_alpha", self.ovp_point_alpha),
            ("ovp_line_alpha", self.ovp_line_alpha),
            ("gdo_show_points", self.gdo_show_points),
            ("gdo_show_ema", self.gdo_show_ema),
            ("gdo_point_alpha", self.gdo_point_alpha),
            ("gdo_line_alpha", self.gdo_line_alpha),
            ("source_mode", self.source_mode),
            ("combine_group_sw3", self.combine_group_sw3),
            ("ivt_sw3_display", self.ivt_sw3_display),
            ("ivt_group_display", self.ivt_group_display),
            ("ovp_group_display", self.ovp_group_display),
            ("controls_visible", self.controls_visible),
        ]: set_if(k,var)

        self._refresh_files_tv()
        self._refresh_groups_tv()
        self._refresh_display_mappings()
        self.on_toggle_controls()
        self._set_status(f"Loaded session from {path}")

    # ------------------------------------------------------------------
    # Analysis: Intensity vs Time
    # ------------------------------------------------------------------
    def on_plot_intensity_vs_time(self):
        # Replace any existing IVT plot(s)
        self._close_plots('ivt')

        mode = self.source_mode.get()
        ema_span = int(self.intensity_ema_span.get())
        trim_start = int(self.trim_start_s.get())
        trim_end   = int(self.trim_end_s.get())
        only_zero  = self.only_yaw_roll_zero.get()
        norm_1m    = self.normalize_to_1m.get()
        show_pts   = self.ivt_show_points.get()
        show_line  = self.ivt_show_ema.get()
        p_alpha    = float(self.ivt_point_alpha.get())
        l_alpha    = float(self.ivt_line_alpha.get())

        def rows_from_sw3(fid: str):
            rec = self.files.get(fid)
            if not rec or rec.kind != "sw3": return []
            try:
                with open(rec.path, "rb") as fd: buf = fd.read()
                ensure_eegbin_imported()
                try: sweep = eegbin.load_eegbin3(buf, from_path=rec.path)
                except Exception: sweep = eegbin.load_eegbin2(buf, from_path=rec.path)
            except Exception as e:
                messagebox.showwarning("Load error", f"Failed to load {rec.path}\n{e}"); return []
            out = []
            for r in sweep.rows:
                if hasattr(r, "valid") and not r.valid: continue
                if only_zero and not (getattr(r.coords,"yaw_deg",None)==0 and getattr(r.coords,"roll_deg",None)==0): continue
                if getattr(r, "timestamp", None) is None: continue
                I_wm2 = float(getattr(r.capture, "integral_result", 0.0) or 0.0)
                if norm_1m:
                    dist_m = float(getattr(r.coords,"lin_mm",0.0) or 0.0)/1000.0
                    if dist_m > 0: I_wm2 = I_wm2*(dist_m**2)
                I_uW_cm2 = I_wm2*100.0
                out.append((float(r.timestamp), I_uW_cm2))
            out.sort(key=lambda x: x[0])
            return out

        def rows_from_sw3_bands(fid: str):
            """
            Returns three time series (lists of (timestamp, uW/cm^2)):
              - total (existing integral_result path, with 1 m normalization if selected)
              - band_200_300 (integrated 200–300 nm)
              - band_200_230 (integrated 200–230 nm)
            """
            rec = self.files.get(fid)
            if not rec or rec.kind != "sw3": return ([], [], [])
            try:
                with open(rec.path, "rb") as fd: buf = fd.read()
                ensure_eegbin_imported()
                try: sweep = eegbin.load_eegbin3(buf, from_path=rec.path)
                except Exception: sweep = eegbin.load_eegbin2(buf, from_path=rec.path)
            except Exception as e:
                messagebox.showwarning("Load error", f"Failed to load {rec.path}\n{e}"); return ([], [], [])

            tot, b200_300, b200_230 = [], [], []
            for r in getattr(sweep, "rows", []):
                if hasattr(r, "valid") and not r.valid: continue
                if only_zero and not (getattr(r.coords,"yaw_deg",None)==0 and getattr(r.coords,"roll_deg",None)==0): continue
                if getattr(r, "timestamp", None) is None: continue
                ts = float(r.timestamp)
                # distance normalization factor (to 1 m) if enabled
                scale = 1.0
                if norm_1m:
                    dist_m = float(getattr(r.coords, "lin_mm", 0.0) or 0.0)/1000.0
                    if dist_m > 0:
                        scale = (dist_m**2)
                # total (existing: integral_result)
                I_wm2 = float(getattr(r.capture, "integral_result", 0.0) or 0.0) * scale
                tot.append((ts, I_wm2 * 100.0))  # W/m^2 -> uW/cm^2
                # band integrations if spectrum is present
                spec = getattr(r.capture, "spectral_result", None)
                if spec and len(spec) > 3:
                    try:
                        i_200_300 = float(sweep.integrate_spectral(spec, 200.0, 300.0)) * scale
                        i_200_230 = float(sweep.integrate_spectral(spec, 200.0, 230.0)) * scale
                        b200_300.append((ts, i_200_300 * 100.0))
                        b200_230.append((ts, i_200_230 * 100.0))
                    except Exception:
                        # If integration fails on a row, just skip that row for bands.
                        pass
            tot.sort(key=lambda x: x[0]); b200_300.sort(key=lambda x: x[0]); b200_230.sort(key=lambda x: x[0])
            return (tot, b200_300, b200_230)

        fig = plt.figure(); ax = fig.add_subplot(111)

        if mode == "file":
            disp = self.ivt_sw3_display.get()
            fid = self._display_to_sw3_id(disp)
            if not fid:
                messagebox.showinfo("Plot", "Select an SW3 file."); return
            rows, rows_b20300, rows_b20230 = rows_from_sw3_bands(fid)

            # Apply per-action trim (seconds) relative to series start
            if rows:
                t = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
                t0 = t[0]
                if trim_start>0: mask = (t - t0) >= trim_start; t,y = t[mask], y[mask]
                if trim_end>0:   tend = t[-1]; mask = (tend - t) >= trim_end; t,y = t[mask], y[mask]
                # If the file belongs to a trimmed group, clip to group window as well
                for g in self.groups.values():
                    if fid in g.file_ids:
                        win = self._group_time_window(g)
                        if win is not None:
                            tmin, tmax = win
                            mask = (t >= tmin) & (t <= tmax)
                            t, y = t[mask], y[mask]
                            # Apply same clipping to band series
                            if 'rows_b20300' in locals() and rows_b20300:
                                tb, yb = np.array([r[0] for r in rows_b20300]), np.array([r[1] for r in rows_b20300])
                                mb = (tb >= tmin) & (tb <= tmax)
                                rows_b20300 = list(zip(tb[mb], yb[mb]))
                            if 'rows_b20230' in locals() and rows_b20230:
                                tc, yc = np.array([r[0] for r in rows_b20230]), np.array([r[1] for r in rows_b20230])
                                mc = (tc >= tmin) & (tc <= tmax)
                                rows_b20230 = list(zip(tc[mc], yc[mc]))
                        break
            else:
                messagebox.showinfo("Plot", "No rows after filtering."); return

            if t.size==0:
                messagebox.showinfo("Plot", "No data to plot after trims."); return
            y_ema = ema(y, ema_span)
            th = (t - t[0]) / 3600.0
            if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{self.files[fid].label} (raw)")
            if show_line: ax.plot(th, y_ema, linewidth=2, alpha=l_alpha, label=f"{self.files[fid].label} EMA")
            # --- Add 200–300 nm and 200–230 nm lines ---
            if 'rows_b20300' in locals() and rows_b20300:
                t2 = np.array([r[0] for r in rows_b20300]); y2 = np.array([r[1] for r in rows_b20300])
                # Apply time trims relative to this series' start
                if t2.size:
                    t2_0 = t2[0]
                    if trim_start>0: m2 = (t2 - t2_0) >= trim_start; t2,y2 = t2[m2], y2[m2]
                    if trim_end>0:   t2end = t2[-1]; m2 = (t2end - t2) >= trim_end; t2,y2 = t2[m2], y2[m2]
                if t2.size:
                    y2_ema = ema(y2, ema_span)
                    th2 = (t2 - t[0]) / 3600.0  # align to main series start for readability
                    if show_line: ax.plot(th2, y2_ema, linewidth=2, alpha=l_alpha, linestyle="--",
                                          label=f"{self.files[fid].label} 200–300 nm EMA")
            if 'rows_b20230' in locals() and rows_b20230:
                t3 = np.array([r[0] for r in rows_b20230]); y3 = np.array([r[1] for r in rows_b20230])
                if t3.size:
                    t3_0 = t3[0]
                    if trim_start>0: m3 = (t3 - t3_0) >= trim_start; t3,y3 = t3[m3], y3[m3]
                    if trim_end>0:   t3end = t3[-1]; m3 = (t3end - t3) >= trim_end; t3,y3 = t3[m3], y3[m3]
                if t3.size:
                    y3_ema = ema(y3, ema_span)
                    th3 = (t3 - t[0]) / 3600.0
                    if show_line: ax.plot(th3, y3_ema, linewidth=2, alpha=l_alpha, linestyle=":",
                                          label=f"{self.files[fid].label} 200–230 nm EMA")
        else:
            gdisp = self.ivt_group_display.get()
            gid = self._display_to_group_id(gdisp)
            if not gid or gid not in self.groups: messagebox.showinfo("Plot", "Select a Group."); return
            g = self.groups[gid]
            win = self._group_time_window(g)
            sw3_ids = [fid for fid in g.file_ids if self.files.get(fid) and self.files[fid].kind=="sw3"]
            if not sw3_ids: messagebox.showinfo("Plot", "No SW3 files in this group."); return
            colors = get_cmap_colors(len(sw3_ids), "viridis")
            if self.combine_group_sw3.get():
                all_rows = []; all_b20300 = []; all_b20230 = []
                for fid in sw3_ids:
                    r0, r20300, r20230 = rows_from_sw3_bands(fid)
                    all_rows.extend(r0); all_b20300.extend(r20300); all_b20230.extend(r20230)
                if win is not None and all_rows:
                    tmin, tmax = win
                    all_rows = self._filter_rows_by_window(all_rows, tmin, tmax)
                    if all_b20300: all_b20300 = self._filter_rows_by_window(all_b20300, tmin, tmax)
                    if all_b20230: all_b20230 = self._filter_rows_by_window(all_b20230, tmin, tmax)
                if not all_rows: messagebox.showinfo("Plot", "No rows after filtering."); return
                all_rows.sort(key=lambda x: x[0])
                t = np.array([r[0] for r in all_rows]); y = np.array([r[1] for r in all_rows])
                y_ema = ema(y, ema_span); th = (t - t[0]) / 3600.0
                if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{g.name} (raw)")
                if show_line: ax.plot(th, y_ema, linewidth=2, alpha=l_alpha, label=f"{g.name} EMA")
                # Group-combined band lines (same color; different styles)
                if 'all_b20300' in locals() and all_b20300:
                    all_b20300.sort(key=lambda x:x[0])
                    t2 = np.array([r[0] for r in all_b20300]); y2 = np.array([r[1] for r in all_b20300])
                    if t2.size:
                        y2_ema = ema(y2, ema_span); th2 = (t2 - t[0]) / 3600.0
                        if show_line: ax.plot(th2, y2_ema, linewidth=2, alpha=l_alpha, linestyle="--",
                                          label=f"{g.name} 200–300 nm EMA")
                if 'all_b20230' in locals() and all_b20230:
                    all_b20230.sort(key=lambda x:x[0])
                    t3 = np.array([r[0] for r in all_b20230]); y3 = np.array([r[1] for r in all_b20230])
                    if t3.size:
                        y3_ema = ema(y3, ema_span); th3 = (t3 - t[0]) / 3600.0
                        if show_line: ax.plot(th3, y3_ema, linewidth=2, alpha=l_alpha, linestyle=":",
                                          label=f"{g.name} 200–230 nm EMA")
            else:
                for i, fid in enumerate(sw3_ids):
                    rows, rows_b20300, rows_b20230 = rows_from_sw3_bands(fid)
                    if win is not None and rows:
                        tmin, tmax = win
                        rows = self._filter_rows_by_window(rows, tmin, tmax)
                    if rows_b20300: rows_b20300 = self._filter_rows_by_window(rows_b20300, tmin, tmax)
                    if rows_b20230: rows_b20230 = self._filter_rows_by_window(rows_b20230, tmin, tmax)
                    if not rows: continue
                    t = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
                    y_ema = ema(y, ema_span); th = (t - t[0]) / 3600.0
                    c = colors[i]
                    if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{self.files[fid].label} (raw)", color=c)
                    if show_line: ax.plot(th, y_ema, linewidth=2, alpha=l_alpha, label=f"{self.files[fid].label} EMA", color=c)

                    # Per-file band lines (same color; dashed/dotted)
                    if 'rows_b20300' in locals() and rows_b20300:
                        rows_b20300.sort(key=lambda x:x[0])
                        t2 = np.array([r[0] for r in rows_b20300]); y2 = np.array([r[1] for r in rows_b20300])
                        if t2.size:
                            y2_ema = ema(y2, ema_span); th2 = (t2 - t[0]) / 3600.0
                            if show_line: ax.plot(th2, y2_ema, linewidth=2, alpha=l_alpha, linestyle="--",
                                                  label=f"{self.files[fid].label} 200–300 nm EMA", color=c)
                    if 'rows_b20230' in locals() and rows_b20230:
                        rows_b20230.sort(key=lambda x:x[0])
                        t3 = np.array([r[0] for r in rows_b20230]); y3 = np.array([r[1] for r in rows_b20230])
                        if t3.size:
                            y3_ema = ema(y3, ema_span); th3 = (t3 - t[0]) / 3600.0
                            if show_line: ax.plot(th3, y3_ema, linewidth=2, alpha=l_alpha, linestyle=":",
                                                  label=f"{self.files[fid].label} 200–230 nm EMA", color=c)
        ax.set_title("Measured Light Intensity vs Time")
        ax.set_xlabel("Time since start (hours)")
        ax.set_ylabel("Normalized Intensity at 1 m (µW/cm²)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        self._register_fig('ivt', fig)
        plt.show(block=False)

    # ------------------------------------------------------------------
    # Analysis: Build aligned frame for a group (applies group trims)
    # ------------------------------------------------------------------
    def _build_aligned_for_group(self, gid: str) -> Optional[pd.DataFrame]:
        g = self.groups.get(gid)
        if not g or not g.associations:
            return None
        ema_int = int(self.intensity_ema_span.get())
        ema_pow = int(self.power_ema_span.get())
        tol_s   = int(self.align_tolerance_s.get())
        only_zero = self.only_yaw_roll_zero.get()
        norm_1m   = self.normalize_to_1m.get()

        frames = []
        win = self._group_time_window(g)

        for sw3_id, pow_ids in g.associations.items():
            sw3_rec = self.files.get(sw3_id)
            if not sw3_rec or sw3_rec.kind!="sw3":
                continue
            try:
                with open(sw3_rec.path, "rb") as fd: buf = fd.read()
                ensure_eegbin_imported()
                try: sweep = eegbin.load_eegbin3(buf, from_path=sw3_rec.path)
                except Exception: sweep = eegbin.load_eegbin2(buf, from_path=sw3_rec.path)
            except Exception as e:
                messagebox.showwarning("Load error", f"Failed to load {sw3_rec.path}\n{e}"); continue

            sw_rows = []
            for r in sweep.rows:
                if hasattr(r, "valid") and not r.valid: continue
                if only_zero and not (getattr(r.coords,"yaw_deg",None)==0 and getattr(r.coords,"roll_deg",None)==0): continue
                if getattr(r, "timestamp", None) is None: continue
                I_wm2 = float(getattr(r.capture, "integral_result", 0.0) or 0.0)
                if norm_1m:
                    dist_m = float(getattr(r.coords,"lin_mm",0.0) or 0.0)/1000.0
                    if dist_m>0: I_wm2 = I_wm2*(dist_m**2)
                I_uW_cm2 = I_wm2*100.0
                sw_rows.append((float(r.timestamp), I_uW_cm2))

            if win is not None and sw_rows:
                tmin, tmax = win
                sw_rows = self._filter_rows_by_window(sw_rows, tmin, tmax)
            if not sw_rows:
                continue

            t_sw = np.array([r[0] for r in sw_rows], dtype=float)
            y_sw = np.array([r[1] for r in sw_rows], dtype=float)
            df_sw = pd.DataFrame({"timestamp": t_sw, "intensity_ema": ema(y_sw, ema_int)})

            # Build per power file
            for pid in pow_ids:
                pow_rec = self.files.get(pid); 
                if not pow_rec or pow_rec.kind!="power": continue
                try:
                    power_df, _ = self._load_power_csv_with_meta(pow_rec.path)
                except Exception as e:
                    messagebox.showwarning("Load error", f"Failed to load CSV {pow_rec.path}\n{e}"); continue

                # Determine power window from trimmed sw window ± tol
                sw_tmin = t_sw.min(); sw_tmax = t_sw.max()
                if win is not None:
                    sw_tmin, sw_tmax = win
                tmin = sw_tmin - tol_s; tmax = sw_tmax + tol_s

                pdf = _prepare_power_series(power_df, tmin, tmax, ema_pow, int(self.resample_seconds.get()))
                if pdf is None: 
                    continue

                df_join = merge_asof_seconds(df_sw, pdf, tol_s).dropna(subset=["power_ema"])
                if df_join.empty: 
                    continue
                df_join["group_id"] = gid
                df_join["sw3_id"]   = sw3_id
                df_join["power_id"] = pid
                df_join["sw3_label"] = sw3_rec.label
                df_join["power_label"] = pow_rec.label
                frames.append(df_join)

        if not frames:
            return None
        aligned = pd.concat(frames, ignore_index=True).sort_values("timestamp_s")
        return aligned

    # ------------------------------------------------------------------
    # Analysis: OVP (time series) and correlation/scatter
    # ------------------------------------------------------------------
    def on_analyze_group(self):
        # Replace any existing OVP plot(s)
        self._close_plots('ovp')

        gdisp = self.ovp_group_display.get()
        gid = self._display_to_group_id(gdisp)
        if not gid or gid not in self.groups:
            messagebox.showinfo("Analyze", "Select a Group."); return
        aligned = self._build_aligned_for_group(gid)
        if aligned is None or aligned.empty:
            messagebox.showinfo("Analyze", "No aligned data could be formed."); return
        self._aligned_cache = aligned

        show_pts = self.ovp_show_points.get()
        show_int = self.ovp_show_int_ema.get()
        show_pow = self.ovp_show_pow_ema.get()
        p_alpha  = float(self.ovp_point_alpha.get())
        l_alpha  = float(self.ovp_line_alpha.get())

        # Build paired colors per (sw3_id, power_id)
        pair_keys = sorted(set(zip(aligned["sw3_id"], aligned["power_id"])))
        base_colors = get_cmap_colors(len(pair_keys), "viridis")
        fig = plt.figure(); ax1 = fig.add_subplot(111); ax2 = ax1.twinx()

        # global origin for readability
        t0_global = aligned["timestamp_s"].min()

        for i, (sw_id, pow_id) in enumerate(pair_keys):
            dfp = aligned[(aligned["sw3_id"]==sw_id) & (aligned["power_id"]==pow_id)]
            if dfp.empty: continue
            th = (dfp["timestamp_s"].to_numpy() - t0_global) / 3600.0
            base = base_colors[i]
            c_int = darken(base, 0.35)   # darker for intensity
            c_pow = lighten(base, 0.60)  # lighter for power
            sw_label = dfp["sw3_label"].iloc[0]
            pw_label = dfp["power_label"].iloc[0]

            if show_int:
                ax1.plot(th, dfp["intensity_ema"].to_numpy(), label=f"{sw_label} (optical irradiance)", alpha=l_alpha, color=c_int, linewidth=2)
            if show_pow:
                ax2.plot(th, dfp["power_ema"].to_numpy(), label=f"{pw_label} (power)", alpha=l_alpha, color=c_pow, linewidth=2)
            if show_pts:
                ax1.scatter(th, dfp["intensity_ema"].to_numpy(), s=8, alpha=p_alpha, color=c_int)

        ax1.set_xlabel("Time since earliest pair start (hours)")
        ax1.set_ylabel("Intensity (µW/cm²)")
        ax2.set_ylabel("Power (W)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # deduplicate legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        seen = {}
        for h, lab in zip(lines1+lines2, labels1+labels2):
            if lab not in seen:
                seen[lab] = h
        ax2.legend(list(seen.values()), list(seen.keys()), loc="best")

        ax1.set_title(f"Optical irradiance and power vs time — {self.groups[gid].name}")
        fig.tight_layout()
        self._register_fig('ovp', fig)
        plt.show(block=False)
        self._set_status(f"Analyzed group '{self.groups[gid].name}' — time series plotted.")

    def on_corr_and_scatter(self):
        # Replace any existing correlation/scatter plots
        self._close_plots('corr')

        if self._aligned_cache is None or self._aligned_cache.empty:
            # Try to build from selected group
            gdisp = self.ovp_group_display.get()
            gid = self._display_to_group_id(gdisp)
            if gid and gid in self.groups:
                self._aligned_cache = self._build_aligned_for_group(gid)
        aligned = self._aligned_cache
        if aligned is None or aligned.empty:
            messagebox.showinfo("Correlation", "Run 'Analyze Group' first (or ensure aligned data exists)."); return

        x = aligned["intensity_ema"].to_numpy(dtype=float)
        y = aligned["power_ema"].to_numpy(dtype=float)

        # Cross-correlation scan
        lags = np.arange(-int(self.ccf_max_lag_s.get()), int(self.ccf_max_lag_s.get())+1, 1, dtype=int)
        ccf_vals = []
        for lag in lags:
            if lag < 0:
                x_lag = x[-lag:]; y_lag = y[:len(x_lag)]
            elif lag > 0:
                y_lag = y[lag:];  x_lag = x[:len(y_lag)]
            else:
                x_lag = x; y_lag = y
            if x_lag.size==0 or y_lag.size==0 or np.std(x_lag)==0 or np.std(y_lag)==0:
                ccf_vals.append(np.nan)
            else:
                ccf_vals.append(float(np.corrcoef(x_lag, y_lag)[0,1]))
        ccf_vals = np.array(ccf_vals, dtype=float)
        best_idx = int(np.nanargmax(np.abs(ccf_vals))) if np.isfinite(ccf_vals).any() else None
        best_lag = int(lags[best_idx]) if best_idx is not None else None
        best_ccf = float(ccf_vals[best_idx]) if best_idx is not None else float("nan")

        # 1) CCF plot
        fig_ccf = plt.figure(); axc = fig_ccf.add_subplot(111)
        axc.plot(lags, ccf_vals); axc.axvline(0, linestyle="--", alpha=0.5)
        if best_idx is not None:
            axc.axvline(best_lag, linestyle=":")
            y_text = np.nanmax(ccf_vals) if np.isfinite(ccf_vals).any() else 0.0
            axc.text(best_lag, y_text, f"best lag={best_lag}s\nr={best_ccf:.3f}", ha="center", va="bottom")
        axc.set_title("Cross‑correlation")
        axc.set_xlabel("Lag (s) [positive = intensity lags power]")
        axc.set_ylabel("Correlation")
        axc.grid(True, linestyle="--", alpha=0.5)
        fig_ccf.tight_layout()
        self._register_fig('corr', fig_ccf)
        plt.show(block=False)

        # 2) Scatter + regression
        slope, intercept, r_lin = linear_regression(aligned["power_ema"].to_numpy(), aligned["intensity_ema"].to_numpy())
        fig_sc = plt.figure(); axs = fig_sc.add_subplot(111)
        # color per pair (use friendly labels)
        pair_keys = sorted(set(zip(aligned["sw3_id"], aligned["power_id"])))
        base_colors = get_cmap_colors(len(pair_keys), "viridis")
        for i, (sw_id, pow_id) in enumerate(pair_keys):
            dfp = aligned[(aligned["sw3_id"]==sw_id) & (aligned["power_id"]==pow_id)]
            axs.scatter(dfp["power_ema"], dfp["intensity_ema"], s=10, alpha=0.25,
                        label=f"{dfp['sw3_label'].iloc[0]} vs {dfp['power_label'].iloc[0]}", color=base_colors[i])
        if np.isfinite(slope) and np.isfinite(intercept):
            xs = np.linspace(aligned["power_ema"].min(), aligned["power_ema"].max(), 200)
            ys = slope*xs + intercept
            axs.plot(xs, ys, linewidth=2, alpha=0.9, label=f"Fit: y={slope:.3f}x+{intercept:.3f}  r={r_lin:.3f}")
        axs.set_xlabel("Power (W) [EMA]"); axs.set_ylabel("Intensity (µW/cm²) [EMA]"); axs.grid(True, linestyle="--", alpha=0.5)
        axs.legend(); fig_sc.tight_layout(); self._register_fig('corr', fig_sc); plt.show(block=False)

    def on_export_aligned_csv(self):
        if self._aligned_cache is None or self._aligned_cache.empty:
            messagebox.showinfo("Export", "No aligned data to export. Run 'Analyze Group' first."); return
        path = filedialog.asksaveasfilename(title="Export Aligned CSV", defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")])
        if not path: return
        df = self._aligned_cache.rename(columns={
            "timestamp_s":"Timestamp", "intensity_ema":"Intensity_EMA_uW_cm2", "power_ema":"Power_EMA_W"
        }).copy()
        keep = ["Timestamp","Intensity_EMA_uW_cm2","Power_EMA_W","group_id","sw3_id","power_id","sw3_label","power_label"]
        df[keep].to_csv(path, index=False)
        self._set_status(f"Exported aligned CSV to {path}")

    # ------------------------------------------------------------------
    # Analysis: Group Decay Overlay (applies trim)
    # ------------------------------------------------------------------
    def on_plot_group_decay(self, selected_only: bool):
        # Replace any existing GDO plot(s)
        self._close_plots('gdo')

        ema_span = int(self.overlay_ema_span.get())
        only_zero = self.only_yaw_roll_zero.get()
        norm_1m   = self.normalize_to_1m.get()
        show_pts  = self.gdo_show_points.get()
        show_line = self.gdo_show_ema.get()
        p_alpha   = float(self.gdo_point_alpha.get())
        l_alpha   = float(self.gdo_line_alpha.get())

        if selected_only:
            idxs = self.lb_groups_select.curselection()
            gids = [self.lb_groups_select.get(i).split("  ")[0] for i in idxs]
        else:
            gids = list(self.groups.keys())
        if not gids:
            messagebox.showinfo("Plot", "No groups selected."); return

        base_colors = get_cmap_colors(len(gids), "viridis")
        fig = plt.figure(); ax = fig.add_subplot(111)

        for i, gid in enumerate(gids):
            g = self.groups.get(gid); 
            if not g: continue
            win = self._group_time_window(g)
            sw3_ids = [fid for fid in g.file_ids if self.files.get(fid) and self.files[fid].kind=="sw3"]
            rows_all = []
            for sid in sw3_ids:
                rec = self.files[sid]
                try:
                    with open(rec.path, "rb") as fd: buf = fd.read()
                    ensure_eegbin_imported()
                    try: sweep = eegbin.load_eegbin3(buf, from_path=rec.path)
                    except Exception: sweep = eegbin.load_eegbin2(buf, from_path=rec.path)
                except Exception as e:
                    messagebox.showwarning("Load error", f"Failed to load {rec.path}\n{e}"); continue
                for r in sweep.rows:
                    if hasattr(r, "valid") and not r.valid: continue
                    if only_zero and not (getattr(r.coords,"yaw_deg",None)==0 and getattr(r.coords,"roll_deg",None)==0): continue
                    if getattr(r, "timestamp", None) is None: continue
                    I_wm2 = float(getattr(r.capture,"integral_result",0.0) or 0.0)
                    if norm_1m:
                        dist_m = float(getattr(r.coords,"lin_mm",0.0) or 0.0)/1000.0
                        if dist_m>0: I_wm2 = I_wm2*(dist_m**2)
                    I_uW_cm2 = I_wm2*100.0
                    rows_all.append((float(r.timestamp), I_uW_cm2))
            if win is not None and rows_all:
                tmin, tmax = win
                rows_all = self._filter_rows_by_window(rows_all, tmin, tmax)
            if not rows_all: continue
            rows_all.sort(key=lambda x:x[0])
            t = np.array([r[0] for r in rows_all]); y = np.array([r[1] for r in rows_all])
            y_ema = ema(y, ema_span)
            y_pct = (y_ema/y_ema.max())*100.0 if y_ema.max()>0 else np.zeros_like(y_ema)
            th = (t - t[0]) / 3600.0
            c = base_colors[i]
            if show_pts: ax.scatter(th, y_pct, s=8, alpha=p_alpha, color=c, label=f"{g.name} (pts)")
            if show_line: ax.plot(th, y_pct, linewidth=2, alpha=l_alpha, color=c, label=g.name)

        ax.set_title("Decay Overlay by Group (EMA, normalized to each group’s peak)")
        ax.set_xlabel("Hours since group start"); ax.set_ylabel("Intensity (% of peak)")
        ax.grid(True, linestyle="--", alpha=0.5); ax.legend()
        fig.tight_layout(); self._register_fig('gdo', fig); plt.show(block=False)

    # ---- Plot utils ----
    def _register_fig(self, cat: str, fig: plt.Figure):
        self._figs.setdefault(cat, []).append(fig)

    def _close_plots(self, cat: str, also: Tuple[str, ...]=()):
        cats = (cat,) + tuple(also)
        for c in cats:
            figs = self._figs.get(c, [])
            for f in figs:
                try: plt.close(f)
                except Exception: pass
            self._figs[c] = []
        self._set_status(f"Closed {', '.join(cats)} plot windows.")

    def _select_all_groups(self):
        self.lb_groups_select.selection_set(0, tk.END)

    def on_save_last_figure(self):
        for cat in ("ivt","ovp","gdo","corr"):
            if self._figs.get(cat):
                last_fig = self._figs[cat][-1]
                path = filedialog.asksaveasfilename(title="Save Figure", defaultextension=".png",
                                                    filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
                if not path: return
                last_fig.savefig(path, dpi=150, bbox_inches="tight")
                self._set_status(f"Saved figure to {path}")
                return
        messagebox.showinfo("Save Figure", "No figure to save yet.")

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _next_file_id(self) -> str:
        i=1
        while f"F{i:03d}" in self.files: i+=1
        return f"F{i:03d}"

    def _next_group_id(self) -> str:
        i=1
        while f"G{i:03d}" in self.groups: i+=1
        return f"G{i:03d}"

# ------------------------------------------------------------------
# Association dialog (friendly labels; shows full mapping)
# ------------------------------------------------------------------

class AssociationDialog(tk.Toplevel):
    def __init__(self, master: App, group: GroupRecord, sw3_list, power_list):
        super().__init__(master)
        self.title(f"Associations for '{group.name}'"); self.resizable(True, True)
        self.group = group; self.updated = False
        self.master = master

        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL); pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        left = ttk.Frame(pane); pane.add(left, weight=1)
        right = ttk.Frame(pane); pane.add(right, weight=1)

        ttk.Label(left, text="SW3 files").pack(anchor="w")
        self.lb_sw3 = tk.Listbox(left, selectmode=tk.SINGLE, exportselection=False)
        for fid, fr in sw3_list: self.lb_sw3.insert(tk.END, f"{fr.label} ({fid})")
        self.sw3_ids = [fid for fid,_ in sw3_list]; self.lb_sw3.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        ttk.Label(right, text="Power files").pack(anchor="w")
        self.lb_pow = tk.Listbox(right, selectmode=tk.MULTIPLE, exportselection=False)
        for fid, fr in power_list: self.lb_pow.insert(tk.END, f"{fr.label} ({fid})")
        self.pow_ids = [fid for fid,_ in power_list]; self.lb_pow.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        btns = ttk.Frame(self); btns.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btns, text="Map selected SW3 → selected Power", command=self._map_selected).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Remove mapping", command=self._remove_mapping).pack(side=tk.LEFT, padx=3)
        ttk.Button(btns, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=3)

        self.tv_map = ttk.Treeview(self, columns=("sw3", "powers"), show="headings", height=8)
        self.tv_map.heading("sw3", text="SW3"); self.tv_map.heading("powers", text="Power")
        self.tv_map.column("sw3", stretch=True, width=260); self.tv_map.column("powers", stretch=True, width=480)
        self.tv_map.pack(fill=tk.BOTH, expand=True, padx=6, pady=6); self._refresh_mapping()
        self.lb_sw3.bind("<<ListboxSelect>>", lambda e: self._refresh_mapping())

    def _map_selected(self):
        idx_sw = self.lb_sw3.curselection(); idx_pw = self.lb_pow.curselection()
        if not idx_sw or not idx_pw: return
        sw_id = self.sw3_ids[idx_sw[0]]; pow_ids = [self.pow_ids[i] for i in idx_pw]
        s = self.group.associations.get(sw_id, set()); s |= set(pow_ids)
        self.group.associations[sw_id] = s; self.updated = True; self._refresh_mapping()

    def _remove_mapping(self):
        idx_sw = self.lb_sw3.curselection(); 
        if not idx_sw: return
        sw_id = self.sw3_ids[idx_sw[0]]; idx_pw = self.lb_pow.curselection()
        if not idx_pw: self.group.associations.pop(sw_id, None)
        else:
            s = self.group.associations.get(sw_id, set())
            for i in idx_pw: s.discard(self.pow_ids[i])
            if s: self.group.associations[sw_id] = s
            else: self.group.associations.pop(sw_id, None)
        self.updated = True; self._refresh_mapping()

    def _refresh_mapping(self):
        for item in self.tv_map.get_children(): self.tv_map.delete(item)
        # Always list full mapping with friendly labels
        for sw_id, pset in sorted(self.group.associations.items()):
            try: sw_label = self.master.files[sw_id].label
            except Exception: sw_label = sw_id
            power_labels = []
            for pid in sorted(list(pset)):
                try: power_labels.append(self.master.files[pid].label)
                except Exception: power_labels.append(pid)
            self.tv_map.insert("", "end", values=(sw_label, ", ".join(power_labels)))

# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
