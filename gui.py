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
import math
import os
import sys
import threading
import time
import traceback
from queue import Empty, Queue
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib import colormaps as mpl_colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

# ------------------------------------------------------------------
# Optional util shim (so importing eegbin doesn't require imgui)
# ------------------------------------------------------------------

def _ensure_util_shim():
    """Install a minimal util shim so eegbin can import without imgui."""
    import types
    if 'util' in sys.modules:
        return
    util = types.ModuleType('util')
    def inclusive_range(start, stop, step):
        """Fallback inclusive_range used when util.py is unavailable."""
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
        """No-op placeholder for imgui editable input."""
        return (False, value)
    def do_editable(preamble, value, units="", width=100, enable=True):
        """No-op placeholder for imgui editable input."""
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
    """Return a Matplotlib colormap by name with fallback for older versions."""
    try:
        return mpl_colormaps.get_cmap(name)
    except Exception:
        return plt.get_cmap(name)

def human_datetime(epoch_s: float) -> str:
    """Format epoch seconds into a local-time human string."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch_s))
    except Exception:
        return ""

def parse_hhmmss(s: str) -> int:
    """Parse seconds or HH:MM:SS into total seconds (supports leading '-')."""
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
    """Format seconds as HH:MM:SS (supports negative values)."""
    neg = seconds < 0
    s = abs(int(seconds))
    hh, rem = divmod(s, 3600)
    mm, ss = divmod(rem, 60)
    out = f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"-{out}" if neg else out

def _add_module_path(path: str):
    """Add a folder to sys.path and remember it for the session."""
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
    """Return an exponential moving average (EMA) with the given span."""
    if span <= 1:
        return np.asarray(arr, dtype=float)
    s = pd.Series(arr, dtype="float64")
    return s.ewm(span=int(span), adjust=False).mean().to_numpy()

def _median_positive_step_s(timestamps_s: np.ndarray) -> float:
    """Return the median positive delta between timestamps in seconds."""
    t = np.asarray(timestamps_s, dtype=float)
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 1.0
    return float(np.median(dt))

def _is_nearly_regular_cadence(
    timestamps_s: np.ndarray,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
    max_ratio: float = 1.25,
) -> bool:
    """Return True when timestamp deltas are close to a fixed cadence."""
    t = np.asarray(timestamps_s, dtype=float)
    if t.size < 4:
        return True
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 3:
        return True
    lo = float(np.percentile(dt, p_lo))
    hi = float(np.percentile(dt, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0:
        return False
    return (hi / lo) <= float(max_ratio)

def detect_cadence_cliff_by_samples(
    timestamps_s: np.ndarray,
    threshold_ratio: float = 3.0,
) -> Tuple[float, Optional[int]]:
    """Detect cadence cliff in sample-count domain.

    Returns:
    - baseline_dt_s: robust early-file baseline sampling interval (seconds)
    - cliff_sample_idx: sample index where sustained slowdown starts, or None
    """
    t = np.asarray(timestamps_s, dtype=float)
    if t.size < 4:
        return 1.0, None

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 8:
        return _median_positive_step_s(t), None

    baseline_n = int(np.clip(dt.size // 5, 300, 6000))
    baseline_n = min(baseline_n, dt.size)
    baseline_dt = float(np.median(dt[:baseline_n]))
    if not np.isfinite(baseline_dt) or baseline_dt <= 0:
        baseline_dt = _median_positive_step_s(t)
        return baseline_dt, None

    window = int(np.clip(baseline_n // 5, 31, 401))
    rolling = pd.Series(dt, dtype="float64").rolling(window=window, min_periods=window).median().to_numpy()
    ratio = rolling / baseline_dt

    sustain_run = int(np.clip(window // 6, 6, 40))
    start_idx = min(max(baseline_n, window), dt.size - 1)
    hits = 0
    for j in range(start_idx, dt.size):
        r = ratio[j]
        if np.isfinite(r) and r >= float(threshold_ratio):
            hits += 1
        else:
            hits = 0
        if hits >= sustain_run:
            # dt[j] is between samples j and j+1.
            return baseline_dt, int(j + 1)
    return baseline_dt, None

def ema_timeaware(timestamps_s: np.ndarray, values: np.ndarray, span: int) -> np.ndarray:
    """EMA on irregular samples using elapsed time instead of sample count.

    `span` is interpreted as "roughly this many median samples of memory", then
    converted to a time half-life from the observed median sample interval.

    This version is cadence-cliff aware:
    - detects sustained slowdown by sample index (not file percent),
    - increases post-cliff smoothing memory,
    - caps per-step decay for alpha calculation so sparse late samples do not
      become excessively jumpy.
    """
    y = np.asarray(values, dtype=float)
    t = np.asarray(timestamps_s, dtype=float)
    if y.size == 0:
        return y
    if y.size == 1 or span <= 1:
        return y.copy()

    dt_ref = _median_positive_step_s(t)
    baseline_dt, cliff_idx = detect_cadence_cliff_by_samples(t)
    halflife_s = max(1.0, float(max(1, int(span))) * dt_ref)
    gap_reset_s = 10.0 * halflife_s
    dt_alpha_cap_s = max(baseline_dt * 2.5, dt_ref)
    post_cliff_halflife_mult = 1.75
    ln2 = np.log(2.0)

    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, y.size):
        raw_dt = t[i] - t[i - 1]
        if not np.isfinite(raw_dt) or raw_dt <= 0:
            raw_dt = dt_ref
        if raw_dt > gap_reset_s:
            # Hard reset across very large gaps to avoid stale-memory carryover.
            out[i] = y[i]
            continue

        dt_for_alpha = min(raw_dt, dt_alpha_cap_s)
        halflife_i = halflife_s
        if cliff_idx is not None and i >= cliff_idx:
            halflife_i = halflife_s * post_cliff_halflife_mult

        alpha = 1.0 - np.exp(-(ln2 * dt_for_alpha) / halflife_i)
        out[i] = out[i - 1] + alpha * (y[i] - out[i - 1])
    return out

def ema_adaptive(timestamps_s: np.ndarray, values: np.ndarray, span: int, timeaware: bool) -> np.ndarray:
    """Compute EMA in sample-count mode or time-aware mode."""
    if not timeaware:
        return ema(values, span)
    return ema_timeaware(timestamps_s, values, span)

def _insert_nan_gaps(timestamps_s: np.ndarray, values: np.ndarray, min_gap_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Insert NaNs so lines break across large sampling/data gaps."""
    t = np.asarray(timestamps_s, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size <= 1:
        return t, y
    out_t = [t[0]]
    out_y = [y[0]]
    for i in range(1, t.size):
        if (t[i] - t[i - 1]) > float(min_gap_s):
            out_t.append(np.nan)
            out_y.append(np.nan)
        out_t.append(t[i])
        out_y.append(y[i])
    return np.asarray(out_t, dtype=float), np.asarray(out_y, dtype=float)

def _gap_threshold_s(timestamps_s: np.ndarray, floor_s: float) -> float:
    """Estimate a practical line-break threshold from observed cadence."""
    cadence = _median_positive_step_s(np.asarray(timestamps_s, dtype=float))
    return max(float(floor_s), 8.0 * cadence)

def _ellipsize(label: str, max_len: int = 56) -> str:
    """Shorten long labels for legends."""
    s = str(label)
    if len(s) <= max_len:
        return s
    return f"{s[:max_len-3]}..."

def _apply_smart_legend(ax, handles, labels):
    """Place legend inside for small sets, outside for dense legends."""
    if not labels:
        return False
    labels = [_ellipsize(lab) for lab in labels]
    outside = len(labels) > 10 or max(len(x) for x in labels) > 42
    if outside:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            frameon=True,
        )
    else:
        ax.legend(handles, labels, loc="best", fontsize=9)
    return outside

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, intercept, r) for y ~ x using scipy if available."""
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
    """Merge two time series by nearest timestamp within a tolerance (seconds)."""
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
                          ema_span: int, resample_seconds: int, timeaware_ema: bool) -> Optional[pd.DataFrame]:
    """Select window, resample, compute EMA; return columns ['timestamp','power_ema'] or None if empty."""
    ema_span = max(1, int(ema_span))
    resample_seconds = max(1, int(resample_seconds))
    # Vectorized numeric path for speed on very large logs.
    ts_all = power_df["Timestamp"].to_numpy(dtype=float)
    w_all = power_df["W_Active"].to_numpy(dtype=float)
    valid = np.isfinite(ts_all) & np.isfinite(w_all)
    if not valid.any():
        return None
    ts = ts_all[valid]
    w = w_all[valid]

    # Time window clip.
    in_win = (ts >= float(tmin)) & (ts <= float(tmax))
    if not in_win.any():
        return None
    ts = ts[in_win]
    w = w[in_win]
    if ts.size == 0:
        return None

    # Ensure sorted timestamps.
    if np.any(np.diff(ts) < 0):
        order = np.argsort(ts)
        ts = ts[order]
        w = w[order]

    # Aggregate onto integer resample buckets without creating full time grids.
    bucket = np.floor(ts / float(resample_seconds)).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(bucket) != 0) + 1]
    sums = np.add.reduceat(w, starts)
    counts = np.diff(np.r_[starts, w.size])
    power = sums / counts
    ts_resampled = bucket[starts].astype(np.float64) * float(resample_seconds)

    # Split segments by large outages and EMA each segment independently.
    gap_split_s = max(float(10 * resample_seconds), 60.0)
    split_idx = np.flatnonzero(np.diff(ts_resampled) > gap_split_s) + 1
    bounds = np.r_[0, split_idx, ts_resampled.size]

    out_ts_parts = []
    out_ema_parts = []
    for i in range(bounds.size - 1):
        a = int(bounds[i])
        b = int(bounds[i + 1])
        if b - a <= 0:
            continue
        seg_ts = ts_resampled[a:b]
        seg_power = power[a:b]
        if timeaware_ema and _is_nearly_regular_cadence(seg_ts):
            # Resampled power is typically near-uniform; fast EMA avoids very
            # long Python loops for million-point runs while preserving shape.
            seg_ema = ema(seg_power, ema_span)
        else:
            seg_ema = ema_adaptive(seg_ts, seg_power, ema_span, timeaware_ema)
        out_ts_parts.append(seg_ts)
        out_ema_parts.append(seg_ema)

    if not out_ts_parts:
        return None
    out_ts = np.concatenate(out_ts_parts)
    out_ema = np.concatenate(out_ema_parts)
    return pd.DataFrame({"timestamp": out_ts.astype(np.int64), "power_ema": out_ema})

def _slice_time_window(df: pd.DataFrame, tmin: float, tmax: float, ts_col: str = "timestamp") -> pd.DataFrame:
    """Fast slice for a dataframe sorted by integer timestamp column."""
    if df is None or df.empty:
        return df
    ts = df[ts_col].to_numpy(dtype=float)
    lo = int(np.searchsorted(ts, tmin, side="left"))
    hi = int(np.searchsorted(ts, tmax, side="right"))
    return df.iloc[lo:hi]

def _estimate_step_s_from_meta(meta: Optional[Dict[str, object]]) -> Optional[float]:
    """Estimate mean sampling step from file metadata if available."""
    if not meta:
        return None
    first = meta.get("first_ts")
    last = meta.get("last_ts")
    count = meta.get("count")
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        return None
    if not isinstance(count, (int, float)) or count is None:
        return None
    n = int(count)
    if n <= 1:
        return None
    duration = float(last) - float(first)
    if not np.isfinite(duration) or duration <= 0:
        return None
    return duration / float(n - 1)

def recommend_power_ema_span(
    intensity_ema_span: int,
    sw3_median_step_s: float,
    resample_seconds: int,
    power_step_s: Optional[float] = None,
) -> int:
    """Recommend power EMA span from optics EMA and observed cadence.

    The goal is to keep smoothing in comparable *time* units between optical
    and power channels.
    """
    intensity_ema_span = max(1, int(intensity_ema_span))
    sw3_median_step_s = float(sw3_median_step_s) if np.isfinite(sw3_median_step_s) else 1.0
    sw3_median_step_s = max(1.0, sw3_median_step_s)
    resample_seconds = max(1, int(resample_seconds))
    if power_step_s is None or not np.isfinite(power_step_s) or power_step_s <= 0:
        effective_power_step_s = float(resample_seconds)
    else:
        effective_power_step_s = max(float(power_step_s), float(resample_seconds))

    target_smoothing_s = float(intensity_ema_span) * sw3_median_step_s
    span = int(round(target_smoothing_s / effective_power_step_s))
    return int(np.clip(span, 5, 4000))

def choose_effective_resample_seconds(
    duration_s: float,
    requested_seconds: int,
    max_points: int = 600_000,
) -> int:
    """Upscale resample interval for very long windows to bound point count."""
    requested = max(1, int(requested_seconds))
    if not np.isfinite(duration_s) or duration_s <= 0:
        return requested
    max_points = max(50_000, int(max_points))
    auto_seconds = int(np.ceil(float(duration_s) / float(max_points)))
    return max(requested, auto_seconds, 1)

def get_cmap_colors(n: int, cmap_name: str) -> List[Tuple[float, float, float, float]]:
    """Return N colors sampled from a colormap."""
    cmap = _get_cmap(cmap_name)
    xs = np.linspace(0.1, 0.9, max(1, n))
    return [cmap(x) for x in xs]

def get_solarized_colors(n: int) -> List[Tuple[float, float, float, float]]:
    """Return N colors from a Solarized-like palette (cycled)."""
    palette = [
        "#6c71c4",  # violet
        "#268bd2",  # blue
        "#2aa198",  # cyan
        "#859900",  # green
        "#b58900",  # yellow
        "#cb4b16",  # orange
        "#d33682",  # magenta
        "#586e75",  # base01
        "#073642",  # base02
    ]
    return [mpl_colors.to_rgba(palette[i % len(palette)]) for i in range(max(1, n))]

def _to_rgb(c):
    """Convert a color to an RGB numpy array."""
    return np.array(mpl_colors.to_rgb(c), dtype=float)

def lighten(color, amount=0.6):
    """Lighten color by mixing with white; amount in [0..1]."""
    c = _to_rgb(color)
    return tuple(np.clip(1 - (1 - c) * (1 - amount), 0, 1))

def darken(color, amount=0.35):
    """Darken color by scaling toward black; amount in [0..1]."""
    c = _to_rgb(color)
    return tuple(np.clip(c * (1 - (1 - amount)), 0, 1))


# Unit conversion: 1 W/m^2 = 100 µW/cm^2
WM2_TO_UW_CM2 = 100.0
APP_NAME = "SW3 + Power Analyzer"
APP_VERSION = "3.0.0"
SESSION_SCHEMA_VERSION = 2

# Report Builder roles/tags
REPORT_SCAN_ROLES = [
    "Unassigned",
    "Complete Dataset",
    "Main Scan",
    "Spectrum Scan",
    "Startup Scan",
    "Composite Multi-phase",
    "Burn-in Scan",
    "Warm-up Run",
    "R2 Validation Scan",
    "Other",
    "Ignore",
]
REPORT_PHASE_TAGS = [
    "Unassigned",
    "Warm-up",
    "Spectrum Point",
    "R2 Pullback",
    "Loose Spectrometer Scan",
    "Tight Spectrometer Scan",
    "Spectral Web",
    "Burn-in",
    "Ignore",
]
REPORT_CSV_ROLES = [
    "Power Log",
    "Power Waveform",
    "Power Factor/Current",
    "Other",
]
REPORT_SEGMENT_TYPES: List[Tuple[str, str]] = [
    ("warmup", "Warm-up"),
    ("spectrum_point", "Spectrum Point"),
    ("r2_pullback", "R2 Pullback"),
    ("loose_web", "Loose Spectrometer Scan"),
    ("tight_web", "Tight Spectrometer Scan"),
]
WARMUP_FULL_HOUR_MIN_H = 0.90
WARMUP_FULL_HOUR_MAX_H = 1.20
REPORT_METADATA_FIELDS: List[Tuple[str, str, str]] = [
    ("reporting_name", "Reporting Name", ""),
    ("reporting_subtitle", "Reporting Subtitle", ""),
    ("catalog_id", "Catalog ID", ""),
    ("revision_name", "Revision Name", "R00"),
    ("revision_date", "Revision Date", ""),
    ("revision_notes", "Revision Notes", "Initial Version"),
    ("acq_name", "Article Name", ""),
    ("acq_manufacturer", "Manufacturer", ""),
    ("acq_model_number", "Model Number", ""),
    ("acq_serial_number", "Article Serial Number", ""),
    ("acq_production_date", "Production Date", ""),
    ("osluv_serial_number", "OSLUV Control Number", ""),
    ("acq_date", "Acquisition Date", ""),
    ("acq_price", "Acquisition Price", ""),
    ("acq_method", "Acquisition Method", ""),
    ("supply_style", "Supply Style", ""),
    ("wall_power_type", "Wall Power Type", ""),
    ("wall_power_w", "Wall Power (W)", ""),
    ("consumer_cost", "Consumer Cost", ""),
    ("psu_desc", "PSU Description", ""),
    ("psu_manuf", "PSU Manufacturer", ""),
    ("psu_model", "PSU Model", ""),
    ("psu_serial", "PSU Serial Number", ""),
    ("psu_osluv_serial", "PSU OSLUV Control Number", ""),
]
REPORT_METADATA_DEFAULTS: Dict[str, str] = {k: d for k, _label, d in REPORT_METADATA_FIELDS}
REPORT_METADATA_LABELS: Dict[str, str] = {k: label for k, label, _d in REPORT_METADATA_FIELDS}
REPORT_METADATA_SECTIONS: List[Tuple[str, List[str]]] = [
    ("Report Identity", ["reporting_name", "reporting_subtitle", "catalog_id", "revision_name", "revision_date", "revision_notes"]),
    (
        "Article Details",
        [
            "acq_name",
            "acq_manufacturer",
            "acq_model_number",
            "acq_serial_number",
            "acq_production_date",
            "osluv_serial_number",
            "acq_date",
            "acq_price",
            "acq_method",
        ],
    ),
    ("Electrical Setup", ["supply_style", "wall_power_type", "wall_power_w", "consumer_cost"]),
    ("Power Supply", ["psu_desc", "psu_manuf", "psu_model", "psu_serial", "psu_osluv_serial"]),
]
REPORT_COMMENT_FIELDS: List[Tuple[str, str]] = [
    ("general_notes", "General Notes"),
    ("warmup_note", "Warm-Up Note"),
    ("burnin_note", "Burn-In Note"),
    ("power_note", "Electrical Notes"),
    ("apparatus_note", "Measurement Apparatus Notes"),
]
REPORT_EXPOSURE_MODES: List[str] = [
    "Unweighted",
    "IES/ANSI Eye",
    "IES/ANSI Skin",
    "IEC",
]
REPORT_OPTICAL_POWER_MODES: List[str] = [
    "Band",
    "Full spectrum",
]
REPORT_SPECTRAL_POWER_BINS: List[Tuple[str, Tuple[float, float]]] = [
    ("Far UVC", (200.0, 240.0)),
    ("High-S(lambda) UVC", (240.0, 300.0)),
    ("UVC", (200.0, 280.0)),
    ("UVB", (280.0, 315.0)),
    ("UVA", (315.0, 400.0)),
    ("Total UV", (200.0, 400.0)),
]
# Coarse fallback S(lambda) anchors from legacy reporting standards tables.
REPORT_EXPOSURE_WEIGHT_CURVES: Dict[str, Dict[str, List[float]]] = {
    "IES/ANSI Eye": {
        "nm": [200, 210, 220, 222, 230, 240, 250, 260, 270, 280, 290, 300, 305, 310, 315, 320, 330, 340, 360, 380, 400],
        "val": [0.001845, 0.002930, 0.013712, 0.018668, 0.064134, 0.300000, 0.430000, 0.650000, 1.000000, 0.880000, 0.640000, 0.300000, 0.060000, 0.015000, 0.003000, 0.001000, 0.000410, 0.000280, 0.000130, 0.000064, 0.000030],
    },
    "IES/ANSI Skin": {
        "nm": [200, 210, 220, 222, 230, 240, 250, 260, 270, 280, 290, 300, 305, 310, 315, 320, 330, 340, 360, 380, 400],
        "val": [0.000300, 0.001196, 0.004758, 0.006270, 0.018919, 0.075232, 0.300000, 0.300000, 0.300000, 0.300000, 0.300000, 0.300000, 0.060000, 0.015000, 0.003000, 0.001000, 0.000410, 0.000280, 0.000130, 0.000064, 0.000030],
    },
    "IEC": {
        "nm": [200, 210, 220, 222, 230, 240, 250, 260, 270, 280, 290, 300, 305, 310, 315, 320, 330, 340, 360, 380, 400],
        "val": [0.030000, 0.075000, 0.120000, 0.131203, 0.190000, 0.300000, 0.430000, 0.650000, 1.000000, 0.880000, 0.640000, 0.300000, 0.060000, 0.015000, 0.003000, 0.001000, 0.000410, 0.000280, 0.000130, 0.000064, 0.000030],
    },
}


def suggest_report_phase_tag(phase: Dict[str, object]) -> str:
    """Return a suggested phase tag from procedure naming/type conventions."""
    name = str(phase.get("name", "") or "").strip().lower()
    ptype = str(phase.get("phase_type", "") or "").strip().upper()
    axis = str(phase.get("axis", "") or "").strip().upper()

    if "ignore" in name or "discard" in name:
        return "Ignore"
    if "burn" in name:
        return "Burn-in"
    if "warm-up" in name or "warmup" in name or ptype == "WARMUP_WAIT":
        return "Warm-up"
    if "spectrum point" in name or ptype == "SPECTRUM_POINT":
        return "Spectrum Point"
    if "r2" in name or "pullback" in name or ptype == "LINEAR_PULLBACK":
        return "R2 Pullback"
    if ptype in ("SPECTRAL_WEB", "INTEGRAL_WEB"):
        if "tight" in name:
            return "Tight Spectrometer Scan"
        if "loose" in name:
            return "Loose Spectrometer Scan"
        return "Spectral Web"
    if axis == "LINEAR":
        return "R2 Pullback"
    return "Unassigned"


def report_segment_key_for_phase(phase: Dict[str, object], tag: Optional[str] = None) -> Optional[str]:
    """Map a phase into a canonical segment key used by report selection UI."""
    name = str(phase.get("name", "") or "").strip().lower()
    ptype = str(phase.get("phase_type", "") or "").strip().upper()
    use_tag = (tag or suggest_report_phase_tag(phase) or "").strip()

    if use_tag == "Warm-up":
        return "warmup"
    if use_tag == "Spectrum Point":
        return "spectrum_point"
    if use_tag == "R2 Pullback":
        return "r2_pullback"
    if use_tag == "Loose Spectrometer Scan":
        return "loose_web"
    if use_tag == "Tight Spectrometer Scan":
        return "tight_web"
    if use_tag == "Spectral Web" or ptype in ("SPECTRAL_WEB", "INTEGRAL_WEB"):
        if "tight" in name:
            return "tight_web"
        if "loose" in name:
            return "loose_web"
    return None


def choose_latest_full_hour_warmup_segment_id(candidates: List[Dict[str, object]]) -> Optional[str]:
    """Pick latest warm-up candidate with ~1 hour duration, falling back to latest available."""
    if not candidates:
        return None

    full_hour = [
        c for c in candidates
        if isinstance(c.get("duration_h"), (int, float))
        and (WARMUP_FULL_HOUR_MIN_H <= float(c.get("duration_h")) <= WARMUP_FULL_HOUR_MAX_H)
    ]
    pool = full_hour if full_hour else candidates
    best = max(pool, key=lambda c: (float(c.get("start_ts", -1.0)), int(c.get("phase_idx", -1))))
    seg_id = best.get("segment_id")
    return str(seg_id) if seg_id else None


def choose_pattern_display_planes(
    roll_values: List[float],
    selected_index: int,
) -> Tuple[List[float], List[float], int]:
    """Return normalized roll list plus plotted planes (0deg, 90deg, and selected)."""
    normalized: List[float] = []
    for value in roll_values:
        try:
            fv = float(value)
        except Exception:
            continue
        if not np.isfinite(fv):
            continue
        normalized.append(fv)
    normalized = sorted(set(normalized))
    if not normalized:
        return [], [], 0

    idx = max(0, min(int(selected_index), len(normalized) - 1))
    selected_plane = normalized[idx]

    planes: List[float] = []
    for target in (0.0, 90.0):
        nearest = min(normalized, key=lambda v: abs(v - target))
        if nearest not in planes:
            planes.append(nearest)
    if selected_plane not in planes:
        planes.append(selected_plane)
    if not planes:
        planes = [normalized[0]]
    return normalized, planes, idx


def log_interp_clamped(
    x: float,
    xp: np.ndarray,
    fp: np.ndarray,
    *,
    zero_at_or_above: Optional[float] = None,
) -> float:
    """Log-space interpolate fp(xp) with endpoint clamping and optional high-end cutoff."""
    xv = float(x)
    if zero_at_or_above is not None and xv >= float(zero_at_or_above):
        return 0.0
    xs = np.asarray(xp, dtype=float)
    ys = np.asarray(fp, dtype=float)
    if xs.size == 0 or ys.size == 0:
        return 0.0
    n = min(xs.size, ys.size)
    xs = xs[:n]
    ys = ys[:n]
    mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return 0.0
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if xv <= float(xs[0]):
        return float(ys[0])
    if xv >= float(xs[-1]):
        return float(ys[-1])
    log_y = np.log10(np.clip(ys, 1e-12, None))
    return float(np.power(10.0, np.interp(xv, xs, log_y)))


def suggest_report_scan_role(phases: List[Dict[str, object]]) -> str:
    """Infer a scan-level role from phase tags and warm-up duration."""
    if not phases:
        return "Unassigned"

    tags = [suggest_report_phase_tag(ph) for ph in phases]
    names = [str(ph.get("name", "") or "").strip().lower() for ph in phases]
    warmup_h = 0.0
    warmup_phase_count = 0
    for ph, tag in zip(phases, tags):
        if tag != "Warm-up":
            continue
        warmup_phase_count += 1
        start = ph.get("start_ts")
        end = ph.get("end_ts")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            warmup_h += (float(end) - float(start)) / 3600.0

    has_scan_pattern = any(
        t in ("Spectrum Point", "R2 Pullback", "Loose Spectrometer Scan", "Tight Spectrometer Scan", "Spectral Web")
        for t in tags
    )
    has_spectrum_point = any(t == "Spectrum Point" for t in tags)
    has_warmup = any(t == "Warm-up" for t in tags)
    has_r2 = any(t == "R2 Pullback" for t in tags)
    has_loose = any(t == "Loose Spectrometer Scan" for t in tags)
    has_tight = any(t == "Tight Spectrometer Scan" for t in tags)
    has_full_run = has_spectrum_point and has_r2 and has_loose and has_tight
    has_burnin_marker = any(t == "Burn-in" for t in tags) or any("burn" in n for n in names)

    if has_burnin_marker:
        return "Burn-in Scan"
    if has_full_run and (warmup_h >= 12 or warmup_phase_count >= 10):
        return "Complete Dataset"
    if has_scan_pattern:
        return "Main Scan"
    if has_warmup and has_spectrum_point:
        return "Startup Scan"
    if has_warmup:
        return "Warm-up Run"
    if has_spectrum_point:
        return "Spectrum Scan"
    if len(phases) > 1:
        return "Composite Multi-phase"
    return "Unassigned"

# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclass
class FileRecord:
    """Project-level metadata for a loaded file (SW3 or Power CSV)."""
    file_id: str
    kind: str   # 'sw3' or 'power'
    path: str
    label: str
    meta: Dict[str, object] = field(default_factory=dict)

@dataclass
class GroupRecord:
    """Grouping of files plus per-group trimming and associations."""
    group_id: str
    name: str
    trim_start_s: int = 0
    trim_end_s: int = 0
    file_ids: Set[str] = field(default_factory=set)
    associations: Dict[str, Set[str]] = field(default_factory=dict)  # sw3_id -> set(power_id)

@dataclass
class ReportScanRecord:
    """Report Builder SW3 scan input with optional phase tagging."""
    scan_id: str
    path: str
    label: str
    role: str = "Unassigned"
    meta: Dict[str, object] = field(default_factory=dict)
    phases: List[Dict[str, object]] = field(default_factory=list)
    phase_tags: Dict[int, str] = field(default_factory=dict)

@dataclass
class ReportCsvRecord:
    """Report Builder CSV input with role tag."""
    csv_id: str
    path: str
    label: str
    role: str = "Power Log"
    meta: Dict[str, object] = field(default_factory=dict)

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

class App(tk.Tk):
    """Main Tkinter application for SW3 + Power Analyzer."""
    def __init__(self):
        """Initialize UI, state, and plotting caches."""
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self._set_initial_geometry()
        self.minsize(920, 620)
        self.option_add("*tearOff", False)
        self._configure_styles()

        self.files: Dict[str, FileRecord] = {}
        self.groups: Dict[str, GroupRecord] = {}
        self.report_scans: Dict[str, ReportScanRecord] = {}
        self.report_csvs: Dict[str, ReportCsvRecord] = {}
        self._report_selected_scan_id: Optional[str] = None
        self._report_phase_scan_display_to_id: Dict[str, str] = {}
        self._report_phase_scan_id_to_display: Dict[str, str] = {}
        self.report_segment_selection: Dict[str, str] = {}
        self._report_segment_display_to_id: Dict[str, Dict[str, str]] = {
            key: {} for key, _ in REPORT_SEGMENT_TYPES
        }
        self._report_segment_combos: Dict[str, ttk.Combobox] = {}
        self._report_sweep_cache: Dict[str, object] = {}
        self._report_preview_plots: Dict[str, Dict[str, object]] = {}
        self._report_comment_widgets: Dict[str, scrolledtext.ScrolledText] = {}
        self._report_overview_text: Optional[scrolledtext.ScrolledText] = None
        self._report_weight_tables: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._report_weight_table_source: Optional[str] = None
        self._report_spectral_cache: Dict[str, object] = {}
        self._report_pattern_cache: Dict[str, object] = {}
        self._report_pattern_slider: Optional[tk.Scale] = None
        self._report_pattern_slider_syncing = False

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
        self._aligned_cache_gid: Optional[str] = None
        self._aligned_cache_signature: Optional[Tuple] = None
        self._last_analyzed_gid: Optional[str] = None
        self._power_csv_cache: Dict[Tuple, Tuple[pd.DataFrame, Dict[str, object]]] = {}
        self._analysis_job_id: int = 0
        self._analysis_running_gid: Optional[str] = None
        self._analysis_running_signature: Optional[Tuple] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_queue: Queue = Queue()
        self._pending_reprocess_gid: Optional[str] = None
        self._ovp_series_cache: Optional[Dict[str, object]] = None
        self._sw3_plot_cache: Dict[Tuple[str, bool, bool], List[Tuple[float, float]]] = {}
        self._sw3_preprocess_job_id: int = 0
        self._sw3_preprocess_thread: Optional[threading.Thread] = None
        self._sw3_preprocess_running_signature: Optional[Tuple] = None
        self._sw3_preprocess_queue: Queue = Queue()
        self._reload_job_id: int = 0
        self._reload_thread: Optional[threading.Thread] = None
        self._reload_queue: Queue = Queue()

    def _set_initial_geometry(self):
        """Fit startup window to screen so bottom controls/status remain visible."""
        self.update_idletasks()
        screen_w = int(self.winfo_screenwidth() or 1440)
        screen_h = int(self.winfo_screenheight() or 900)
        width = max(1024, min(1320, screen_w - 120))
        height = max(680, min(880, screen_h - 140))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2 - 10)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _configure_styles(self):
        """Apply a cohesive ttk style pass for a cleaner, denser UI."""
        style = ttk.Style(self)
        # Keep the OS-native ttk theme so controls match platform styling.
        style.configure(".", font=("TkDefaultFont", 10))
        style.configure("Treeview", rowheight=22)
        style.configure("Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
        style.configure("TNotebook.Tab", padding=(12, 5))
        style.configure("Section.TLabelframe", padding=(7, 5))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Toolbar.TButton", padding=(8, 4))
        style.configure("Primary.TButton", padding=(10, 5))
        style.configure("Status.TLabel", padding=(8, 4))
        style.configure("Hint.TLabel", foreground="#586e75")

    # ---- control vars ----
    def _init_vars(self):
        """Initialize Tkinter variables for controls and state."""
        # analysis params
        # Tuned on Open Excimer long-run dataset (see docs/open_excimer_recommendations.json)
        self.intensity_ema_span = tk.IntVar(self, value=15)
        self.power_ema_span     = tk.IntVar(self, value=86)
        self.overlay_ema_span   = tk.IntVar(self, value=15)
        self.trim_start_s       = tk.IntVar(self, value=0)   # per-IVT extra trim
        self.trim_end_s         = tk.IntVar(self, value=0)
        self.align_tolerance_s  = tk.IntVar(self, value=6)
        self.ccf_max_lag_s      = tk.IntVar(self, value=180)
        self.resample_seconds   = tk.IntVar(self, value=3)
        self.time_weighted_ema  = tk.BooleanVar(self, value=True)
        self.auto_power_ema     = tk.BooleanVar(self, value=True)

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

        # report builder
        self.report_scan_role_var = tk.StringVar(self, value="Unassigned")
        self.report_phase_tag_var = tk.StringVar(self, value="Unassigned")
        self.report_csv_role_var  = tk.StringVar(self, value="Power Log")
        self.report_phase_scan_display_var = tk.StringVar(self, value="")
        self.report_scan_info_var = tk.StringVar(self, value="No scan selected.")
        self.report_lamp_image_path_var = tk.StringVar(self, value="")
        self.report_axes_image_path_var = tk.StringVar(self, value="")
        self.report_segment_choice_vars = {
            key: tk.StringVar(self, value="")
            for key, _ in REPORT_SEGMENT_TYPES
        }
        self.report_meta_vars: Dict[str, tk.StringVar] = {
            key: tk.StringVar(self, value=default)
            for key, _label, default in REPORT_METADATA_FIELDS
        }
        self.report_exposure_mode_var = tk.StringVar(self, value="IEC")
        self.report_optical_power_mode_var = tk.StringVar(self, value="Band")
        self.report_optical_power_min_nm_var = tk.StringVar(self, value="200")
        self.report_optical_power_max_nm_var = tk.StringVar(self, value="230")
        self.report_pattern_plane_idx_var = tk.IntVar(self, value=0)
        self.report_pattern_plane_label_var = tk.StringVar(self, value="No plane rows available.")
        self.report_preview_status_var = tk.StringVar(self, value="Preview not generated.")
        self.report_electrical_summary_var = tk.StringVar(self, value="No electrical summary yet.")

    # ---- UI ----
    def _build_ui(self):
        """Construct the main window layout."""
        self._build_menu()

        # Main top-level tabs: Analyzer + Report Builder
        self._main_nb = ttk.Notebook(self)
        self._main_nb.pack(fill=tk.BOTH, expand=True)

        tab_analyzer = ttk.Frame(self._main_nb)
        tab_report = ttk.Frame(self._main_nb)
        self._main_nb.add(tab_analyzer, text="Analyzer")
        self._main_nb.add(tab_report, text="Report Builder")

        # Analyzer tab: Single vertical panedwindow with three panes
        self._paned = ttk.Panedwindow(tab_analyzer, orient=tk.VERTICAL)
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

        # Report Builder tab: full-page layout
        self._build_report_builder_tab(tab_report)

        # status / busy indicator (explicit bottom strip so it stays visible)
        status_shell = tk.Frame(self, bg="#fdf6e3", bd=1, relief=tk.SUNKEN, height=30)
        status_shell.pack(fill=tk.X, side=tk.BOTTOM)
        status_shell.pack_propagate(False)
        status_bar = tk.Frame(status_shell, bg="#fdf6e3")
        status_bar.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        self.status_var = tk.StringVar(self, value="Ready.")
        self._status_title = tk.Label(
            status_bar,
            text="Status:",
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
            bg="#fdf6e3",
            fg="#586e75",
        )
        self._status_title.pack(side=tk.LEFT, padx=(0, 6))
        self._status_label = tk.Label(
            status_bar,
            textvariable=self.status_var,
            anchor="w",
            bg="#fdf6e3",
            fg="#586e75",
        )
        self._status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

        busy_shell = tk.Frame(status_bar, bg="#fdf6e3")
        busy_shell.pack(side=tk.RIGHT)
        self._busy_var = tk.StringVar(self, value="Background: idle")
        self._busy_label = tk.Label(
            busy_shell,
            textvariable=self._busy_var,
            anchor="e",
            bg="#fdf6e3",
            fg="#657b83",
        )
        self._busy_label.pack(side=tk.RIGHT, padx=(0, 8))
        self._busy_pb = ttk.Progressbar(busy_shell, orient=tk.HORIZONTAL, mode="indeterminate", length=130)
        self._busy_pb.stop()
        self._busy_pb.pack_forget()
        self._busy_active = False

        # initial vertical sash positions (favor controls) with minimum Files height
        self.after(120, self._set_initial_sashes)

    def _build_menu(self):
        """Create the menu bar and bind menu actions."""
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
            "About",
            f"{APP_NAME} v{APP_VERSION}\n"
            "Aligns SW3/eegbin optical data with power logs by epoch time.\n"
            "Includes a Report Builder tab for scan/CSV/segment metadata setup.",
        ))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_files_frame(self, parent):
        """Build the Files pane (file list and actions)."""
        frm = ttk.LabelFrame(parent, text="Files")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        toolbar = ttk.Frame(frm)
        toolbar.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Label(toolbar, text="Actions:", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="Add SW3", style="Toolbar.TButton", command=self.on_add_sw3_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Add Power CSV", style="Toolbar.TButton", command=self.on_add_power_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Assign Group", style="Toolbar.TButton", command=self.on_assign_files_to_group).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(toolbar, text="Rename", style="Toolbar.TButton", command=self.on_rename_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Remove", style="Toolbar.TButton", command=self.on_remove_files).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(
            toolbar,
            text="Reload Selected",
            style="Toolbar.TButton",
            command=lambda: self.on_reload_all_files(only_selected=True),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reload All", style="Toolbar.TButton", command=self.on_reload_all_files).pack(side=tk.LEFT, padx=2)

        # --- Treeview container BELOW the button bar ---
        container = ttk.Frame(frm)
        container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 6))

        cols = ("label", "kind", "first", "last", "count", "path")
        self.tv_files = ttk.Treeview(container, columns=cols, show="headings", selectmode="extended")
        heading = {
            "label": "Label",
            "kind": "Kind",
            "first": "First Seen",
            "last": "Last Seen",
            "count": "Rows",
            "path": "Path",
        }
        for c in cols:
            self.tv_files.heading(c, text=heading[c])
            if c == "label":
                w = 220
            elif c == "path":
                w = 360
            elif c == "kind":
                w = 80
            elif c == "count":
                w = 90
            else:
                w = 170
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
        """Build the Groups pane (groups list and associations)."""
        frm = ttk.LabelFrame(parent, text="Groups & Associations")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        top = ttk.Frame(frm)
        top.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Label(top, text="Actions:", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="New Group", style="Toolbar.TButton", command=self.on_new_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Rename", style="Toolbar.TButton", command=self.on_rename_group).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Delete", style="Toolbar.TButton", command=self.on_delete_group).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(top, text="Associations", style="Toolbar.TButton", command=self.on_edit_associations).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Set Trim", style="Toolbar.TButton", command=self.on_set_group_trim).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Clear Trim", style="Toolbar.TButton", command=self.on_clear_group_trim).pack(side=tk.LEFT, padx=2)

        container = ttk.Frame(frm)
        container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 6))
        cols = ("name", "trim", "sw3_count", "power_count")
        self.tv_groups = ttk.Treeview(container, columns=cols, show="headings", selectmode="browse")
        heading = {"name": "Group", "trim": "Trim", "sw3_count": "SW3", "power_count": "Power"}
        for c in cols:
            self.tv_groups.heading(c, text=heading[c])
            if c == "name":
                w = 260
            elif c == "trim":
                w = 220
            else:
                w = 90
            self.tv_groups.column(c, width=w, anchor="w", stretch=True)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tv_groups.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.tv_groups.xview)
        self.tv_groups.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tv_groups.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1); container.columnconfigure(0, weight=1)
        self.tv_groups.bind("<<TreeviewSelect>>", lambda e: self._on_group_selection_changed())

    def _build_controls(self, parent):
        """Build the Controls pane (analysis controls and plots)."""
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def make_section(tab, title: str):
            sec = ttk.LabelFrame(tab, text=title, style="Section.TLabelframe")
            sec.pack(fill=tk.X, pady=(0, 5))
            return sec

        def make_row(sec, pady=(0, 2)):
            row = ttk.Frame(sec)
            row.pack(fill=tk.X, padx=4, pady=pady)
            return row

        # ---- Intensity vs Time ----
        tab_ivt = ttk.Frame(nb, padding=(8, 7))
        nb.add(tab_ivt, text="Intensity vs Time")

        ivt_src = make_section(tab_ivt, "Source")
        r0 = make_row(ivt_src)
        ttk.Label(r0, text="Mode").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(r0, text="Single file", value="file", variable=self.source_mode).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(r0, text="Group", value="group", variable=self.source_mode).pack(side=tk.LEFT)
        r0b = make_row(ivt_src)
        ttk.Label(r0b, text="SW3").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_ivt_sw3 = ttk.Combobox(r0b, state="readonly", width=34, textvariable=self.ivt_sw3_display)
        self.cb_ivt_sw3.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(r0b, text="Group").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_ivt_group = ttk.Combobox(r0b, state="readonly", width=30, textvariable=self.ivt_group_display)
        self.cb_ivt_group.pack(side=tk.LEFT)
        r0c = make_row(ivt_src, pady=(0, 0))
        ttk.Checkbutton(r0c, text="Combine all SW3 files in selected group", variable=self.combine_group_sw3).pack(
            side=tk.LEFT
        )

        ivt_filter = make_section(tab_ivt, "Filtering & Window")
        r1 = make_row(ivt_filter)
        ttk.Label(r1, text="Optics EMA span").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r1, textvariable=self.intensity_ema_span, width=7).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r1, text="Time-aware EMA", variable=self.time_weighted_ema).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r1, text="Normalize to 1 m", variable=self.normalize_to_1m).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r1, text="Yaw=0, Roll=0 only", variable=self.only_yaw_roll_zero).pack(side=tk.LEFT)
        r1b = make_row(ivt_filter, pady=(0, 0))
        ttk.Label(r1b, text="Trim start").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r1b, textvariable=self.trim_start_s, width=9).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(r1b, text="s").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(r1b, text="Trim end").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(r1b, textvariable=self.trim_end_s, width=9).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(r1b, text="s").pack(side=tk.LEFT)

        ivt_display = make_section(tab_ivt, "Display")
        r2 = make_row(ivt_display)
        ttk.Checkbutton(r2, text="Show points", variable=self.ivt_show_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(r2, text="Show EMA", variable=self.ivt_show_ema).pack(side=tk.LEFT)
        r2b = make_row(ivt_display, pady=(0, 0))
        ttk.Label(r2b, text="Points alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(r2b, variable=self.ivt_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Label(r2b, text="EMA alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(r2b, variable=self.ivt_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT
        )

        ivt_actions = ttk.Frame(tab_ivt)
        ivt_actions.pack(fill=tk.X)
        ttk.Button(ivt_actions, text="Plot", style="Primary.TButton", command=self.on_plot_intensity_vs_time).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(ivt_actions, text="Save Figure...", style="Toolbar.TButton", command=self.on_save_last_figure).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            ivt_actions,
            text="Close IVT/Spectrum Plots",
            style="Toolbar.TButton",
            command=lambda: self._close_plots('ivt', also=('spec',))
        ).pack(side=tk.LEFT, padx=10)
        ttk.Label(
            tab_ivt,
            text="Trim accepts seconds (90) or HH:MM:SS (00:01:30).",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(8, 0))

        # ---- Optics vs Power ----
        tab_ovp = ttk.Frame(nb, padding=(8, 7))
        nb.add(tab_ovp, text="Optics vs Power")
        ovp_cfg = make_section(tab_ovp, "Group & Alignment")
        o1 = make_row(ovp_cfg)
        ttk.Label(o1, text="Group").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_group = ttk.Combobox(o1, state="readonly", width=42, textvariable=self.ovp_group_display)
        self.cb_group.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(o1, text="Optics EMA comes from Intensity tab", style="Hint.TLabel").pack(side=tk.LEFT)
        o1b = make_row(ovp_cfg, pady=(0, 0))
        ttk.Label(o1b, text="Power EMA").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(o1b, textvariable=self.power_ema_span, width=7).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(o1b, text="Auto from optics + cadence", variable=self.auto_power_ema).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Label(o1b, text="Resample").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(o1b, textvariable=self.resample_seconds, width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(o1b, text="s").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(o1b, text="Align tol").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(o1b, textvariable=self.align_tolerance_s, width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(o1b, text="s").pack(side=tk.LEFT)

        ovp_display = make_section(tab_ovp, "Display")
        o2 = make_row(ovp_display)
        ttk.Checkbutton(o2, text="Show points", variable=self.ovp_show_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(o2, text="Show optics EMA", variable=self.ovp_show_int_ema).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(o2, text="Show power EMA", variable=self.ovp_show_pow_ema).pack(side=tk.LEFT)
        o2b = make_row(ovp_display, pady=(0, 0))
        ttk.Label(o2b, text="Points alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(o2b, variable=self.ovp_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Label(o2b, text="Lines alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(o2b, variable=self.ovp_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT
        )

        ovp_actions = ttk.Frame(tab_ovp)
        ovp_actions.pack(fill=tk.X)
        ttk.Button(ovp_actions, text="Analyze Group", style="Primary.TButton", command=self.on_analyze_group).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            ovp_actions,
            text="Correlation & Scatter",
            style="Toolbar.TButton",
            command=self.on_corr_and_scatter,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(ovp_actions, text="Export Aligned CSV...", style="Toolbar.TButton", command=self.on_export_aligned_csv).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(ovp_actions, text="Save Figure...", style="Toolbar.TButton", command=self.on_save_last_figure).pack(
            side=tk.LEFT, padx=3
        )
        ttk.Button(
            ovp_actions,
            text="Close OVP Plots",
            style="Toolbar.TButton",
            command=lambda: self._close_plots('ovp', also=('corr',))
        ).pack(side=tk.LEFT, padx=10)

        # ---- Group Decay Overlay ----
        tab_gdo = ttk.Frame(nb, padding=(8, 7))
        nb.add(tab_gdo, text="Group Decay Overlay")
        gdo_display = make_section(tab_gdo, "Display")
        g1 = make_row(gdo_display)
        ttk.Label(g1, text="Optics EMA comes from Intensity tab", style="Hint.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(g1, text="Show points", variable=self.gdo_show_points).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(g1, text="Show EMA", variable=self.gdo_show_ema).pack(side=tk.LEFT)
        g1b = make_row(gdo_display, pady=(0, 0))
        ttk.Label(g1b, text="Points alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(g1b, variable=self.gdo_point_alpha, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Label(g1b, text="Lines alpha").pack(side=tk.LEFT, padx=(0, 4))
        tk.Scale(g1b, variable=self.gdo_line_alpha, from_=0.1, to=1, resolution=0.05, orient=tk.HORIZONTAL, length=140).pack(
            side=tk.LEFT
        )

        gdo_actions = ttk.Frame(tab_gdo)
        gdo_actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            gdo_actions,
            text="Plot Selected Groups",
            style="Primary.TButton",
            command=lambda: self.on_plot_group_decay(selected_only=True),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            gdo_actions,
            text="Plot All Groups",
            style="Toolbar.TButton",
            command=lambda: self.on_plot_group_decay(selected_only=False),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(gdo_actions, text="Close GDO Plots", style="Toolbar.TButton", command=lambda: self._close_plots('gdo')).pack(side=tk.LEFT, padx=10)

        ttk.Label(tab_gdo, text="Pick groups below (default: all groups).", style="Hint.TLabel").pack(anchor="w", pady=(0, 4))
        frame_list = ttk.LabelFrame(tab_gdo, text="Group Selection", style="Section.TLabelframe")
        frame_list.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        self.lb_groups_select = tk.Listbox(frame_list, selectmode=tk.EXTENDED, exportselection=False)
        vs = ttk.Scrollbar(frame_list, orient="vertical", command=self.lb_groups_select.yview)
        self.lb_groups_select.configure(yscroll=vs.set)
        self.lb_groups_select.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        frame_list.rowconfigure(0, weight=1)
        frame_list.columnconfigure(0, weight=1)
        bsel = ttk.Frame(tab_gdo)
        bsel.pack(fill=tk.X, pady=2)
        ttk.Button(bsel, text="Select All", style="Toolbar.TButton", command=self._select_all_groups).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            bsel,
            text="Select None",
            style="Toolbar.TButton",
            command=lambda: self.lb_groups_select.selection_clear(0, tk.END),
        ).pack(side=tk.LEFT, padx=3)



    def _build_report_builder_tab(self, parent):
        """Build the Report Builder UI (scan/CSV inputs + phase tagging)."""
        tab = ttk.Frame(parent, padding=(8, 7))
        tab.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            tab,
            text="Import SW3 scans and CSV inputs, then tag scan phases for report generation.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        report_nb = ttk.Notebook(tab)
        report_nb.pack(fill=tk.BOTH, expand=True)

        tab_inputs = ttk.Frame(report_nb, padding=(4, 4))
        tab_phase = ttk.Frame(report_nb, padding=(4, 4))
        tab_meta = ttk.Frame(report_nb, padding=(4, 4))
        tab_preview = ttk.Frame(report_nb, padding=(4, 4))
        report_nb.add(tab_inputs, text="Inputs")
        report_nb.add(tab_phase, text="Phases & Segments")
        report_nb.add(tab_meta, text="Metadata")
        report_nb.add(tab_preview, text="Preview")

        h_pane = ttk.Panedwindow(tab_inputs, orient=tk.HORIZONTAL)
        h_pane.pack(fill=tk.BOTH, expand=True)

        # ---- SW3 Scans ----
        scan_frame = ttk.LabelFrame(h_pane, text="SW3 Scans")
        h_pane.add(scan_frame, weight=1)
        try:
            h_pane.pane(scan_frame, minsize=320, weight=1)
        except Exception:
            pass
        scan_toolbar = ttk.Frame(scan_frame)
        scan_toolbar.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Button(scan_toolbar, text="Add SW3", style="Toolbar.TButton", command=self.on_add_report_sw3_files).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(scan_toolbar, text="Remove", style="Toolbar.TButton", command=self.on_remove_report_scans).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Separator(scan_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(scan_toolbar, text="Rename", style="Toolbar.TButton", command=self.on_rename_report_scan).pack(
            side=tk.LEFT, padx=2
        )

        scan_container = ttk.Frame(scan_frame)
        scan_container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 6))
        scan_cols = ("label", "role", "phases", "first", "last", "path")
        self.tv_report_scans = ttk.Treeview(scan_container, columns=scan_cols, show="headings", selectmode="browse")
        scan_heading = {
            "label": "Label",
            "role": "Role",
            "phases": "Phases",
            "first": "First Seen",
            "last": "Last Seen",
            "path": "Path",
        }
        for c in scan_cols:
            self.tv_report_scans.heading(c, text=scan_heading[c])
            if c == "label":
                w = 160
            elif c == "role":
                w = 120
            elif c == "phases":
                w = 60
            elif c in ("first", "last"):
                w = 125
            else:
                w = 220
            self.tv_report_scans.column(c, width=w, anchor="w", stretch=True)
        scan_vsb = ttk.Scrollbar(scan_container, orient="vertical", command=self.tv_report_scans.yview)
        scan_hsb = ttk.Scrollbar(scan_container, orient="horizontal", command=self.tv_report_scans.xview)
        self.tv_report_scans.configure(yscroll=scan_vsb.set, xscroll=scan_hsb.set)
        self.tv_report_scans.grid(row=0, column=0, sticky="nsew")
        scan_vsb.grid(row=0, column=1, sticky="ns")
        scan_hsb.grid(row=1, column=0, sticky="ew")
        scan_container.rowconfigure(0, weight=1)
        scan_container.columnconfigure(0, weight=1)
        self.tv_report_scans.bind("<<TreeviewSelect>>", lambda e: self._on_report_scan_selected())
        self.tv_report_scans.bind("<Double-1>", lambda e: self.on_rename_report_scan())

        # ---- CSV Inputs ----
        csv_frame = ttk.LabelFrame(h_pane, text="CSV Inputs")
        h_pane.add(csv_frame, weight=1)
        try:
            h_pane.pane(csv_frame, minsize=320, weight=1)
        except Exception:
            pass
        csv_toolbar = ttk.Frame(csv_frame)
        csv_toolbar.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Button(csv_toolbar, text="Add CSV", style="Toolbar.TButton", command=self.on_add_report_csv_files).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(csv_toolbar, text="Remove", style="Toolbar.TButton", command=self.on_remove_report_csvs).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Separator(csv_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(csv_toolbar, text="Rename", style="Toolbar.TButton", command=self.on_rename_report_csv).pack(
            side=tk.LEFT, padx=2
        )

        csv_container = ttk.Frame(csv_frame)
        csv_container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 6))
        csv_cols = ("label", "role", "columns", "size", "path")
        self.tv_report_csvs = ttk.Treeview(csv_container, columns=csv_cols, show="headings", selectmode="browse")
        csv_heading = {
            "label": "Label",
            "role": "Role",
            "columns": "Columns",
            "size": "Size",
            "path": "Path",
        }
        for c in csv_cols:
            self.tv_report_csvs.heading(c, text=csv_heading[c])
            if c == "label":
                w = 150
            elif c == "role":
                w = 130
            elif c == "columns":
                w = 170
            elif c == "size":
                w = 80
            else:
                w = 220
            self.tv_report_csvs.column(c, width=w, anchor="w", stretch=True)
        csv_vsb = ttk.Scrollbar(csv_container, orient="vertical", command=self.tv_report_csvs.yview)
        csv_hsb = ttk.Scrollbar(csv_container, orient="horizontal", command=self.tv_report_csvs.xview)
        self.tv_report_csvs.configure(yscroll=csv_vsb.set, xscroll=csv_hsb.set)
        self.tv_report_csvs.grid(row=0, column=0, sticky="nsew")
        csv_vsb.grid(row=0, column=1, sticky="ns")
        csv_hsb.grid(row=1, column=0, sticky="ew")
        csv_container.rowconfigure(0, weight=1)
        csv_container.columnconfigure(0, weight=1)
        self.tv_report_csvs.bind("<<TreeviewSelect>>", lambda e: self._on_report_csv_selected())
        self.tv_report_csvs.bind("<Double-1>", lambda e: self.on_rename_report_csv())

        csv_footer = ttk.Frame(csv_frame)
        csv_footer.pack(fill=tk.X, padx=4, pady=(0, 6))
        ttk.Label(csv_footer, text="Selected CSV role:").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_report_csv_role = ttk.Combobox(
            csv_footer,
            state="readonly",
            width=22,
            textvariable=self.report_csv_role_var,
            values=REPORT_CSV_ROLES,
        )
        self.cb_report_csv_role.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(csv_footer, text="Set Role", style="Toolbar.TButton", command=self.on_report_set_csv_role).pack(
            side=tk.LEFT, padx=2
        )

        image_frame = ttk.LabelFrame(csv_frame, text="Report Images")
        image_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        img_grid = ttk.Frame(image_frame)
        img_grid.pack(fill=tk.X, padx=4, pady=(6, 6))
        img_grid.columnconfigure(1, weight=1)
        img_grid.columnconfigure(2, minsize=90)
        img_grid.columnconfigure(3, minsize=66)

        ttk.Label(img_grid, text="Lamp photo").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 4))
        ttk.Entry(
            img_grid,
            textvariable=self.report_lamp_image_path_var,
            state="readonly",
            width=38,
        ).grid(row=0, column=1, sticky="ew", pady=(0, 4))
        ttk.Button(
            img_grid,
            text="Browse...",
            style="Toolbar.TButton",
            command=self.on_report_set_lamp_image,
        ).grid(row=0, column=2, sticky="ew", padx=(6, 2), pady=(0, 4))
        ttk.Button(
            img_grid,
            text="Clear",
            style="Toolbar.TButton",
            command=lambda: self._set_report_image_path("lamp", ""),
        ).grid(row=0, column=3, sticky="ew", padx=(2, 0), pady=(0, 4))

        ttk.Label(img_grid, text="Axes photo").grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(
            img_grid,
            textvariable=self.report_axes_image_path_var,
            state="readonly",
            width=38,
        ).grid(row=1, column=1, sticky="ew")
        ttk.Button(
            img_grid,
            text="Browse...",
            style="Toolbar.TButton",
            command=self.on_report_set_axes_image,
        ).grid(row=1, column=2, sticky="ew", padx=(6, 2))
        ttk.Button(
            img_grid,
            text="Clear",
            style="Toolbar.TButton",
            command=lambda: self._set_report_image_path("axes", ""),
        ).grid(row=1, column=3, sticky="ew", padx=(2, 0))

        # ---- Phase Tagging ----
        phase_frame = ttk.LabelFrame(tab_phase, text="Phase Tagging")
        phase_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        row0 = ttk.Frame(phase_frame)
        row0.pack(fill=tk.X, padx=4, pady=(6, 2))
        ttk.Label(row0, text="Scan").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_report_phase_scan = ttk.Combobox(
            row0,
            state="readonly",
            width=46,
            textvariable=self.report_phase_scan_display_var,
        )
        self.cb_report_phase_scan.pack(side=tk.LEFT, padx=(0, 8))
        self.cb_report_phase_scan.bind("<<ComboboxSelected>>", lambda _e: self._on_report_phase_scan_combo_selected())
        ttk.Label(row0, textvariable=self.report_scan_info_var, style="Hint.TLabel").pack(side=tk.LEFT)

        row1 = ttk.Frame(phase_frame)
        row1.pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Label(row1, text="Scan role").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_report_scan_role = ttk.Combobox(
            row1,
            state="readonly",
            width=18,
            textvariable=self.report_scan_role_var,
            values=REPORT_SCAN_ROLES,
        )
        self.cb_report_scan_role.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row1, text="Set Scan Role", style="Toolbar.TButton", command=self.on_report_set_scan_role).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            row1,
            text="Apply Role",
            style="Toolbar.TButton",
            command=self.on_report_apply_role_to_phases,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Auto-tag", style="Toolbar.TButton", command=self.on_report_auto_tag_phases).pack(
            side=tk.LEFT, padx=2
        )

        phase_container = ttk.Frame(phase_frame)
        phase_container.pack(fill=tk.BOTH, expand=True, padx=3, pady=(2, 4))
        phase_cols = ("index", "name", "type", "axis", "start", "end", "duration_h", "count", "tag")
        self.tv_report_phases = ttk.Treeview(
            phase_container,
            columns=phase_cols,
            show="headings",
            selectmode="extended",
            height=5,
        )
        phase_heading = {
            "index": "#",
            "name": "Phase",
            "type": "Type",
            "axis": "Axis",
            "start": "Start",
            "end": "End",
            "duration_h": "Duration (h)",
            "count": "Rows",
            "tag": "Tag",
        }
        for c in phase_cols:
            self.tv_report_phases.heading(c, text=phase_heading[c])
            if c == "index":
                w = 40
            elif c == "name":
                w = 150
            elif c == "type":
                w = 95
            elif c == "axis":
                w = 70
            elif c in ("start", "end"):
                w = 120
            elif c == "duration_h":
                w = 90
            elif c == "count":
                w = 60
            else:
                w = 130
            self.tv_report_phases.column(c, width=w, anchor="w", stretch=True)
        phase_vsb = ttk.Scrollbar(phase_container, orient="vertical", command=self.tv_report_phases.yview)
        phase_hsb = ttk.Scrollbar(phase_container, orient="horizontal", command=self.tv_report_phases.xview)
        self.tv_report_phases.configure(yscroll=phase_vsb.set, xscroll=phase_hsb.set)
        self.tv_report_phases.grid(row=0, column=0, sticky="nsew")
        phase_vsb.grid(row=0, column=1, sticky="ns")
        phase_hsb.grid(row=1, column=0, sticky="ew")
        phase_container.rowconfigure(0, weight=1)
        phase_container.columnconfigure(0, weight=1)

        row2 = ttk.Frame(phase_frame)
        row2.pack(fill=tk.X, padx=4, pady=(2, 6))
        ttk.Label(row2, text="Phase tag").pack(side=tk.LEFT, padx=(0, 4))
        self.cb_report_phase_tag = ttk.Combobox(
            row2,
            state="readonly",
            width=20,
            textvariable=self.report_phase_tag_var,
            values=REPORT_PHASE_TAGS,
        )
        self.cb_report_phase_tag.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            row2,
            text="Set Tag for Selected Phases",
            style="Primary.TButton",
            command=self.on_report_set_phase_tag,
        ).pack(side=tk.LEFT, padx=2)

        seg_frame = ttk.LabelFrame(phase_frame, text="Segment Selection (Choose Which Duplicate Phase To Use)")
        seg_frame.pack(fill=tk.X, padx=4, pady=(2, 4))
        seg_toolbar = ttk.Frame(seg_frame)
        seg_toolbar.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Button(
            seg_toolbar,
            text="Auto Latest",
            style="Toolbar.TButton",
            command=self.on_report_autoselect_segments_latest,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            seg_toolbar,
            text="Auto Selected Scan",
            style="Toolbar.TButton",
            command=self.on_report_autoselect_segments_from_selected_scan,
        ).pack(side=tk.LEFT, padx=2)

        seg_grid = ttk.Frame(seg_frame)
        seg_grid.pack(fill=tk.X, padx=4, pady=(2, 6))
        for row_idx, (seg_key, seg_label) in enumerate(REPORT_SEGMENT_TYPES):
            ttk.Label(seg_grid, text=f"{seg_label}:").grid(
                row=row_idx, column=0, sticky="w", padx=(0, 6), pady=2
            )
            cb = ttk.Combobox(
                seg_grid,
                state="readonly",
                width=56,
                textvariable=self.report_segment_choice_vars[seg_key],
            )
            cb.grid(row=row_idx, column=1, sticky="ew", pady=2)
            cb.bind("<<ComboboxSelected>>", lambda _e, key=seg_key: self._on_report_segment_choice(key))
            self._report_segment_combos[seg_key] = cb
        seg_grid.columnconfigure(1, weight=1)

        ttk.Label(
            phase_frame,
            text="Tip: warm-up auto-selection uses the latest ~1 hour warm-up segment in the chosen SW3 run.",
            style="Hint.TLabel",
        ).pack(anchor="w", padx=4, pady=(0, 6))
        self._refresh_report_segment_selectors()
        self._build_report_metadata_tab(tab_meta)
        self._build_report_preview_tab(tab_preview)

    def _build_report_metadata_tab(self, parent):
        """Build report metadata/notes editor as its own top-level tab."""
        meta_frame = ttk.LabelFrame(parent, text="Report Metadata & Notes")
        meta_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        meta_toolbar = ttk.Frame(meta_frame)
        meta_toolbar.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Button(
            meta_toolbar,
            text="Use Selected Scan Label",
            style="Toolbar.TButton",
            command=self.on_report_seed_metadata_from_selected_scan,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(
            meta_toolbar,
            text="Edit metadata and section notes here. Analysis controls and plots are in the Preview tab.",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT)

        meta_container = ttk.Frame(meta_frame)
        meta_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        meta_canvas = tk.Canvas(meta_container, highlightthickness=0)
        meta_vsb = ttk.Scrollbar(meta_container, orient="vertical", command=meta_canvas.yview)
        meta_canvas.configure(yscrollcommand=meta_vsb.set)
        meta_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        meta_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        form = ttk.Frame(meta_canvas)
        form_window = meta_canvas.create_window((0, 0), window=form, anchor="nw")

        def _sync_meta_scroll(_event=None):
            meta_canvas.configure(scrollregion=meta_canvas.bbox("all"))

        def _sync_form_width(event):
            try:
                meta_canvas.itemconfigure(form_window, width=event.width)
            except Exception:
                pass

        form.bind("<Configure>", _sync_meta_scroll)
        meta_canvas.bind("<Configure>", _sync_form_width)

        self._report_comment_widgets = {}
        row = 0
        for section_name, field_keys in REPORT_METADATA_SECTIONS:
            ttk.Label(form, text=section_name, font=("TkDefaultFont", 10, "bold")).grid(
                row=row,
                column=0,
                columnspan=2,
                sticky="w",
                padx=2,
                pady=(8, 4),
            )
            row += 1
            for key in field_keys:
                ttk.Label(form, text=f"{REPORT_METADATA_LABELS.get(key, key)}:").grid(
                    row=row,
                    column=0,
                    sticky="w",
                    padx=(4, 6),
                    pady=1,
                )
                ttk.Entry(
                    form,
                    textvariable=self.report_meta_vars[key],
                ).grid(row=row, column=1, sticky="ew", padx=(0, 2), pady=1)
                row += 1

        ttk.Label(form, text="Section Notes", font=("TkDefaultFont", 10, "bold")).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            padx=2,
            pady=(10, 4),
        )
        row += 1
        for key, label in REPORT_COMMENT_FIELDS:
            ttk.Label(form, text=f"{label}:").grid(
                row=row,
                column=0,
                sticky="nw",
                padx=(4, 6),
                pady=(2, 2),
            )
            txt = scrolledtext.ScrolledText(form, height=3, wrap=tk.WORD)
            txt.grid(row=row, column=1, sticky="ew", padx=(0, 2), pady=(2, 2))
            self._report_comment_widgets[key] = txt
            row += 1
        form.columnconfigure(1, weight=1)

    def _build_report_preview_tab(self, parent):
        """Build report previews as a dedicated top-level tab."""
        preview_frame = ttk.LabelFrame(parent, text="Report Analysis Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        preview_toolbar = ttk.Frame(preview_frame)
        preview_toolbar.pack(fill=tk.X, padx=4, pady=(6, 4))
        ttk.Button(
            preview_toolbar,
            text="Refresh Previews",
            style="Primary.TButton",
            command=self.on_report_refresh_previews,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(preview_toolbar, text="Exposure weighting:").pack(side=tk.LEFT, padx=(10, 4))
        self.cb_report_exposure_mode = ttk.Combobox(
            preview_toolbar,
            state="readonly",
            width=16,
            values=REPORT_EXPOSURE_MODES,
            textvariable=self.report_exposure_mode_var,
        )
        self.cb_report_exposure_mode.pack(side=tk.LEFT, padx=(0, 6))
        self.cb_report_exposure_mode.bind("<<ComboboxSelected>>", lambda _e: self.on_report_exposure_mode_changed())
        ttk.Label(preview_toolbar, text="Optical power band (nm):").pack(side=tk.LEFT, padx=(10, 4))
        self.cb_report_optical_power_mode = ttk.Combobox(
            preview_toolbar,
            state="readonly",
            width=13,
            values=REPORT_OPTICAL_POWER_MODES,
            textvariable=self.report_optical_power_mode_var,
        )
        self.cb_report_optical_power_mode.pack(side=tk.LEFT, padx=(0, 4))
        self.cb_report_optical_power_mode.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.on_report_optical_power_settings_changed(),
        )
        self.ent_report_optical_power_min = ttk.Entry(
            preview_toolbar,
            textvariable=self.report_optical_power_min_nm_var,
            width=6,
        )
        self.ent_report_optical_power_min.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(preview_toolbar, text="to").pack(side=tk.LEFT, padx=(0, 2))
        self.ent_report_optical_power_max = ttk.Entry(
            preview_toolbar,
            textvariable=self.report_optical_power_max_nm_var,
            width=6,
        )
        self.ent_report_optical_power_max.pack(side=tk.LEFT, padx=(0, 4))
        self.ent_report_optical_power_min.bind("<Return>", lambda _e: self.on_report_optical_power_settings_changed())
        self.ent_report_optical_power_max.bind("<Return>", lambda _e: self.on_report_optical_power_settings_changed())
        ttk.Button(
            preview_toolbar,
            text="200-230",
            style="Toolbar.TButton",
            command=lambda: self.on_report_set_optical_power_preset(200.0, 230.0),
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(
            preview_toolbar,
            text="200-300",
            style="Toolbar.TButton",
            command=lambda: self.on_report_set_optical_power_preset(200.0, 300.0),
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Separator(preview_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(
            preview_toolbar,
            text="Open Spectral Table",
            style="Toolbar.TButton",
            command=self.on_report_open_spectral_table_popout,
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(
            preview_toolbar,
            text="Open Spectral Linear",
            style="Toolbar.TButton",
            command=self.on_report_open_spectral_linear_popout,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            preview_toolbar,
            text="Open Spectral Log",
            style="Toolbar.TButton",
            command=self.on_report_open_spectral_log_popout,
        ).pack(side=tk.LEFT, padx=2)
        self._sync_report_optical_power_controls()
        ttk.Label(preview_toolbar, textvariable=self.report_preview_status_var, style="Hint.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        preview_nb = ttk.Notebook(preview_frame)
        preview_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 6))

        tab_overview = ttk.Frame(preview_nb, padding=(4, 4))
        preview_nb.add(tab_overview, text="Overview")
        self._report_overview_text = scrolledtext.ScrolledText(tab_overview, wrap=tk.WORD, height=14)
        self._report_overview_text.pack(fill=tk.BOTH, expand=True)
        self._set_readonly_text_widget(self._report_overview_text, "Preview not generated yet.")

        preview_nb.add(self._build_report_preview_plot_tab(preview_nb, "pattern", "Radiant Intensity Pattern"), text="Pattern")
        preview_nb.add(self._build_report_preview_plot_tab(preview_nb, "warmup", "Warm-Up Behavior"), text="Warm-Up")
        preview_nb.add(self._build_report_preview_plot_tab(preview_nb, "burnin", "Burn-In Behavior"), text="Burn-In")
        preview_nb.add(self._build_report_preview_plot_tab(preview_nb, "spectral", "Spectral Output"), text="Spectral")
        preview_nb.add(self._build_report_preview_plot_tab(preview_nb, "exposure", "Exposure Limits (Unweighted)"), text="Exposure")
        preview_nb.add(self._build_report_preview_plot_tab(preview_nb, "roll", "Roll-Dependence"), text="Roll")
        preview_nb.add(self._build_report_preview_plot_tab(preview_nb, "r2", "R² Validation"), text="R²")

        tab_electrical = self._build_report_preview_plot_tab(preview_nb, "electrical", "Electrical Characteristics")
        ttk.Label(
            tab_electrical,
            textvariable=self.report_electrical_summary_var,
            style="Hint.TLabel",
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, expand=False, pady=(2, 0))
        preview_nb.add(tab_electrical, text="Electrical")

    def _build_report_preview_plot_tab(self, parent, key: str, title: str):
        """Create one preview tab with a Matplotlib canvas and message line."""
        tab = ttk.Frame(parent, padding=(4, 4))
        if key == "spectral":
            fig_size = (8.0, 6.2)
        elif key == "electrical":
            fig_size = (7.6, 4.8)
        else:
            fig_size = (7.0, 3.3)
        fig = Figure(figsize=fig_size, dpi=100, tight_layout=True)
        toolbar_shell = ttk.Frame(tab, height=36)
        toolbar_shell.pack_propagate(False)
        toolbar_shell.pack(fill=tk.X, expand=False, pady=(0, 2))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(2, 2))
        toolbar_host = ttk.Frame(toolbar_shell)
        toolbar_host.pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_host, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)
        if key == "spectral":
            ttk.Button(
                toolbar_host,
                text="Open Table",
                style="Toolbar.TButton",
                command=self.on_report_open_spectral_table_popout,
            ).pack(side=tk.LEFT, padx=(10, 2))
            ttk.Button(
                toolbar_host,
                text="Open Linear",
                style="Toolbar.TButton",
                command=self.on_report_open_spectral_linear_popout,
            ).pack(side=tk.LEFT, padx=2)
            ttk.Button(
                toolbar_host,
                text="Open Log",
                style="Toolbar.TButton",
                command=self.on_report_open_spectral_log_popout,
            ).pack(side=tk.LEFT, padx=2)
        if key == "pattern":
            ttk.Label(toolbar_host, text="Custom plane:").pack(side=tk.LEFT, padx=(10, 4))
            self._report_pattern_slider = tk.Scale(
                toolbar_host,
                from_=0,
                to=0,
                variable=self.report_pattern_plane_idx_var,
                orient=tk.HORIZONTAL,
                resolution=1,
                length=170,
                showvalue=False,
                state=tk.DISABLED,
                highlightthickness=0,
                command=self.on_report_pattern_plane_slider_changed,
            )
            self._report_pattern_slider.pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(
                toolbar_host,
                textvariable=self.report_pattern_plane_label_var,
                style="Hint.TLabel",
                width=18,
            ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(
            toolbar_host,
            text="Use toolbar controls to pan, box-zoom, reset view, and save preview images.",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT, padx=(8, 0))
        msg_var = tk.StringVar(self, value="No preview data yet.")
        ttk.Label(tab, textvariable=msg_var, style="Hint.TLabel", justify=tk.LEFT, wraplength=1000).pack(anchor="w")
        self._report_preview_plots[key] = {
            "figure": fig,
            "canvas": canvas,
            "toolbar": toolbar,
            "message_var": msg_var,
        }
        return tab

    def _set_readonly_text_widget(self, widget: Optional[scrolledtext.ScrolledText], text: str):
        """Replace text in a read-only text widget."""
        if widget is None:
            return
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    def _get_report_comment_text(self, key: str) -> str:
        """Read report note text for a given note key."""
        widget = self._report_comment_widgets.get(key)
        if widget is None:
            return ""
        return widget.get("1.0", tk.END).strip()

    def _set_report_comment_text(self, key: str, text: str):
        """Set report note text for a given note key."""
        widget = self._report_comment_widgets.get(key)
        if widget is None:
            return
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text or "")

    def _collect_report_comments(self) -> Dict[str, str]:
        """Snapshot comment fields from the metadata editor."""
        return {key: self._get_report_comment_text(key) for key, _label in REPORT_COMMENT_FIELDS}

    def _apply_report_comments(self, data: Dict[str, object]):
        """Load comment fields into metadata editor widgets."""
        for key, _label in REPORT_COMMENT_FIELDS:
            self._set_report_comment_text(key, str(data.get(key, "") or ""))

    def _collect_report_metadata(self) -> Dict[str, str]:
        """Snapshot metadata entry fields."""
        return {key: var.get().strip() for key, var in self.report_meta_vars.items()}

    def _apply_report_metadata(self, data: Dict[str, object]):
        """Load metadata entry fields from persisted data."""
        for key, _label, default in REPORT_METADATA_FIELDS:
            self.report_meta_vars[key].set(str(data.get(key, default) or ""))

    def on_report_seed_metadata_from_selected_scan(self):
        """Populate key metadata fields from the selected scan label if empty."""
        sid = self._current_report_scan_id()
        if not sid:
            messagebox.showinfo("Report Metadata", "Select a scan first.")
            return
        rec = self.report_scans.get(sid)
        if rec is None:
            return
        if not self.report_meta_vars["reporting_name"].get().strip():
            self.report_meta_vars["reporting_name"].set(rec.label)
        if not self.report_meta_vars["acq_name"].get().strip():
            self.report_meta_vars["acq_name"].set(rec.label)
        if not self.report_meta_vars["catalog_id"].get().strip():
            self.report_meta_vars["catalog_id"].set("UNASSIGNED")
        self._set_status("Seeded report metadata from selected scan.")

    def on_report_exposure_mode_changed(self):
        """Handle exposure weighting mode changes."""
        mode = self.report_exposure_mode_var.get().strip() or "Unweighted"
        self._set_status(f"Exposure weighting set to {mode}.")
        if self._report_preview_plots:
            self.on_report_refresh_previews()

    def _sync_report_optical_power_controls(self):
        """Enable/disable optical band entry fields based on integration mode."""
        mode = self.report_optical_power_mode_var.get().strip() or "Band"
        state = "normal" if mode == "Band" else "disabled"
        try:
            self.ent_report_optical_power_min.configure(state=state)
            self.ent_report_optical_power_max.configure(state=state)
        except Exception:
            pass

    def on_report_optical_power_settings_changed(self):
        """Handle optical power integration control updates."""
        mode = self.report_optical_power_mode_var.get().strip() or "Band"
        self._sync_report_optical_power_controls()
        if mode == "Full spectrum":
            self._set_status("Optical power integration set to full spectrum.")
        else:
            self._set_status(
                f"Optical power integration band set to "
                f"{self.report_optical_power_min_nm_var.get().strip()}-"
                f"{self.report_optical_power_max_nm_var.get().strip()} nm."
            )
        if self._report_preview_plots:
            self.on_report_refresh_previews()

    def on_report_set_optical_power_preset(self, lo_nm: float, hi_nm: float):
        """Apply a wavelength-band preset for optical power integration."""
        self.report_optical_power_mode_var.set("Band")
        self.report_optical_power_min_nm_var.set(f"{float(lo_nm):.1f}".rstrip("0").rstrip("."))
        self.report_optical_power_max_nm_var.set(f"{float(hi_nm):.1f}".rstrip("0").rstrip("."))
        self.on_report_optical_power_settings_changed()

    def _report_pattern_cache_signature(
        self,
        rows: List[object],
        used: List[str],
    ) -> Tuple[Tuple[str, ...], int, Optional[float], Optional[float]]:
        """Return a compact signature for cached pattern preprocessing."""
        used_sig = tuple(sorted(set(str(x) for x in used)))
        if not rows:
            return used_sig, 0, None, None
        first_ts = float(getattr(rows[0], "timestamp", 0.0) or 0.0)
        last_ts = float(getattr(rows[-1], "timestamp", 0.0) or 0.0)
        return used_sig, len(rows), first_ts, last_ts

    def _build_report_pattern_cache(
        self,
        rows: List[object],
        used: List[str],
    ) -> Dict[str, object]:
        """Precompute roll/yaw traces for responsive custom-plane slider updates."""
        sig = self._report_pattern_cache_signature(rows, used)
        roll_yaw_map: Dict[float, Dict[float, List[float]]] = {}
        for row in rows:
            coords = getattr(row, "coords", None)
            if coords is None:
                continue
            roll = getattr(coords, "roll_deg", None)
            yaw = getattr(coords, "yaw_deg", None)
            if roll is None or yaw is None:
                continue
            roll_f = float(roll)
            yaw_f = float(yaw)
            if not (np.isfinite(roll_f) and np.isfinite(yaw_f)):
                continue
            val = self._row_intensity_uW_cm2(row, normalize_to_1m=True)
            if not np.isfinite(val):
                continue
            yaw_bins = roll_yaw_map.setdefault(roll_f, {})
            yaw_bins.setdefault(yaw_f, []).append(float(val))

        series_by_roll: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        roll_values: List[float] = []
        all_vals: List[float] = []
        for roll_f in sorted(roll_yaw_map.keys()):
            yaw_bins = roll_yaw_map[roll_f]
            if not yaw_bins:
                continue
            yaws = np.asarray(sorted(yaw_bins.keys()), dtype=float)
            ys = np.asarray([float(np.mean(yaw_bins[yy])) for yy in yaws], dtype=float)
            if ys.size == 0:
                continue
            series_by_roll[roll_f] = (yaws, ys)
            roll_values.append(roll_f)
            all_vals.extend(ys.tolist())

        return {
            "signature": sig,
            "roll_values": roll_values,
            "series_by_roll": series_by_roll,
            "rows_count": len(rows),
            "segment_count": len(set(str(x) for x in used)),
            "peak": float(max(all_vals)) if all_vals else None,
        }

    def _ensure_report_pattern_cache(
        self,
        rows: List[object],
        used: List[str],
    ) -> Dict[str, object]:
        """Return cached preprocessed pattern traces; rebuild when inputs changed."""
        sig = self._report_pattern_cache_signature(rows, used)
        cache = self._report_pattern_cache
        if cache and cache.get("signature") == sig:
            return cache
        cache = self._build_report_pattern_cache(rows, used)
        self._report_pattern_cache = cache
        return cache

    def _draw_report_pattern_from_cache(self, ax, cache: Dict[str, object]) -> str:
        """Draw pattern plot from precomputed cache and selected slider index."""
        roll_values = [float(x) for x in list(cache.get("roll_values", []) or [])]
        series_by_roll = cache.get("series_by_roll", {})
        if not roll_values or not isinstance(series_by_roll, dict):
            ax.set_axis_off()
            msg = "No usable rows for pattern preview."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            return msg

        roll_values, planes, selected_idx = choose_pattern_display_planes(
            roll_values,
            int(self.report_pattern_plane_idx_var.get()),
        )
        self._sync_report_pattern_plane_controls(roll_values, selected_idx)
        selected_plane = roll_values[selected_idx] if roll_values else None

        plotted_vals: List[float] = []
        for plane in planes:
            pair = series_by_roll.get(plane)
            if pair is None:
                continue
            yaws, ys = pair
            yaws_arr = np.asarray(yaws, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
            if yaws_arr.size == 0 or ys_arr.size == 0:
                continue
            ax.plot(
                yaws_arr,
                ys_arr,
                marker="o",
                markersize=2.5,
                linewidth=1.1,
                label=f"Roll {plane:.1f}°",
            )
            plotted_vals.extend(ys_arr.tolist())

        if not plotted_vals:
            ax.set_axis_off()
            msg = "No usable rows for pattern preview."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            return msg

        ax.set_xlabel("Yaw (deg)")
        ax.set_ylabel("Irradiance at 1m (uW/cm^2)")
        ax.grid(True, alpha=0.3)
        if len(planes) > 1:
            ax.legend(loc="best", fontsize=8)
        peak = float(max(plotted_vals))
        rows_count = int(cache.get("rows_count", 0) or 0)
        seg_count = int(cache.get("segment_count", 0) or 0)
        if selected_plane is not None:
            return (
                f"Used {rows_count} web rows from {seg_count} selected segment(s). "
                f"Peak {peak:.1f} uW/cm^2 @1m. "
                f"Custom plane: roll {selected_plane:.1f} deg."
            )
        return f"Used {rows_count} web rows from {seg_count} selected segment(s). Peak {peak:.1f} uW/cm^2 @1m."

    def _sync_report_pattern_plane_controls(
        self,
        roll_values: List[float],
        selected_index: Optional[int] = None,
    ):
        """Sync pattern custom-plane slider limits and selected-value label."""
        wanted_idx = (
            int(selected_index)
            if selected_index is not None
            else int(self.report_pattern_plane_idx_var.get())
        )
        normalized, _planes, idx = choose_pattern_display_planes(roll_values, wanted_idx)

        self._report_pattern_slider_syncing = True
        try:
            if not normalized:
                self.report_pattern_plane_idx_var.set(0)
                self.report_pattern_plane_label_var.set("No plane rows available.")
                if self._report_pattern_slider is not None:
                    self._report_pattern_slider.configure(from_=0, to=0, state=tk.DISABLED)
                return
            self.report_pattern_plane_idx_var.set(idx)
            self.report_pattern_plane_label_var.set(f"Roll {normalized[idx]:.1f} deg")
            if self._report_pattern_slider is not None:
                state = tk.NORMAL if len(normalized) > 1 else tk.DISABLED
                self._report_pattern_slider.configure(
                    from_=0,
                    to=max(0, len(normalized) - 1),
                    state=state,
                )
        finally:
            self._report_pattern_slider_syncing = False

    def on_report_pattern_plane_slider_changed(self, raw_value: str):
        """Update pattern preview when custom roll-plane slider changes."""
        if self._report_pattern_slider_syncing:
            return
        cache = self._report_pattern_cache
        roll_values = list(cache.get("roll_values", []) or [])
        if not roll_values:
            return
        try:
            requested_idx = int(round(float(raw_value)))
        except Exception:
            requested_idx = int(self.report_pattern_plane_idx_var.get())
        normalized, _planes, idx = choose_pattern_display_planes(
            [float(x) for x in roll_values],
            requested_idx,
        )
        if not normalized:
            return
        self._sync_report_pattern_plane_controls(normalized, idx)
        slot = self._report_preview_slot("pattern")
        if slot:
            fig: Figure = slot["figure"]  # type: ignore[assignment]
            fig.clear()
            ax = fig.add_subplot(111)
            msg = self._draw_report_pattern_from_cache(ax, cache)
            self._preview_set_message("pattern", msg)
            self._preview_draw("pattern")

    def _report_preview_slot(self, key: str) -> Optional[Dict[str, object]]:
        """Return preview slot dict for a tab key."""
        return self._report_preview_plots.get(key)

    def _segment_id_parts(self, segment_id: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse segment id format `<scan_id>:<phase_idx>`."""
        if ":" not in segment_id:
            return None, None
        sid, idx_txt = segment_id.split(":", 1)
        try:
            return sid, int(idx_txt)
        except Exception:
            return sid, None

    def _valid_sorted_rows(self, rows: List[object]) -> List[object]:
        """Filter invalid rows and sort by timestamp."""
        out: List[object] = []
        for row in rows:
            ts = getattr(row, "timestamp", None)
            if ts is None:
                continue
            if hasattr(row, "valid") and not bool(getattr(row, "valid")):
                continue
            out.append(row)
        out.sort(key=lambda r: float(getattr(r, "timestamp", 0.0)))
        return out

    def _get_report_sweep(self, scan_id: str):
        """Get cached sweep for report scan id; load if needed."""
        if scan_id in self._report_sweep_cache:
            return self._report_sweep_cache[scan_id]
        rec = self.report_scans.get(scan_id)
        if rec is None:
            return None
        sweep = self._load_sweep_from_path(rec.path, warn=False, trace=False)
        if sweep is not None:
            self._report_sweep_cache[scan_id] = sweep
        return sweep

    def _phase_rows_from_summary(self, sweep, phase_summary: Dict[str, object]) -> List[object]:
        """Fallback rows for a phase summary when `phase.members` is unavailable."""
        start = phase_summary.get("start_ts")
        end = phase_summary.get("end_ts")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return []
        rows = []
        for row in getattr(sweep, "rows", []) or []:
            ts = getattr(row, "timestamp", None)
            if ts is None:
                continue
            if float(start) <= float(ts) <= float(end):
                rows.append(row)
        return self._valid_sorted_rows(rows)

    def _resolve_report_segment(self, seg_key: str) -> Optional[Dict[str, object]]:
        """Resolve selected segment key to rows and owning scan info."""
        segment_id = self.report_segment_selection.get(seg_key)
        if not segment_id:
            return None
        scan_id, phase_idx = self._segment_id_parts(segment_id)
        if not scan_id or phase_idx is None:
            return None
        rec = self.report_scans.get(scan_id)
        sweep = self._get_report_sweep(scan_id)
        if rec is None or sweep is None:
            return None

        phase_summary = rec.phases[phase_idx] if 0 <= phase_idx < len(rec.phases) else {}
        rows: List[object] = []
        scan_phases = getattr(sweep, "phases", []) or []
        if 0 <= phase_idx < len(scan_phases):
            rows = self._valid_sorted_rows(list(getattr(scan_phases[phase_idx], "members", []) or []))
        if not rows and phase_summary:
            rows = self._phase_rows_from_summary(sweep, phase_summary)
        return {
            "segment_id": segment_id,
            "scan_id": scan_id,
            "phase_idx": phase_idx,
            "scan_record": rec,
            "phase_summary": phase_summary,
            "sweep": sweep,
            "rows": rows,
        }

    def _collect_rows_from_segment_keys(self, seg_keys: List[str]) -> Tuple[List[object], List[str]]:
        """Collect rows from selected segment keys."""
        rows: List[object] = []
        used: List[str] = []
        for key in seg_keys:
            seg = self._resolve_report_segment(key)
            if not seg or not seg["rows"]:
                continue
            rows.extend(seg["rows"])
            used.append(str(seg["segment_id"]))
        return self._valid_sorted_rows(rows), used

    def _select_burnin_rows(self) -> Tuple[List[object], Optional[str]]:
        """Pick long-run warm-up rows for burn-in preview."""
        best_rows: List[object] = []
        best_sid: Optional[str] = None
        best_score = (-1.0, -1.0)
        for sid, rec in self.report_scans.items():
            if rec.role not in ("Burn-in Scan", "Complete Dataset", "Main Scan"):
                continue
            sweep = self._get_report_sweep(sid)
            if sweep is None:
                continue
            rows: List[object] = []
            phases = getattr(sweep, "phases", []) or []
            for ph in rec.phases:
                idx = int(ph.get("index", 0))
                tag = self._phase_tag_for_record(rec, idx, ph)
                if tag != "Warm-up":
                    continue
                if 0 <= idx < len(phases):
                    rows.extend(list(getattr(phases[idx], "members", []) or []))
            rows = self._valid_sorted_rows(rows)
            if len(rows) < 2:
                continue
            duration_h = (float(rows[-1].timestamp) - float(rows[0].timestamp)) / 3600.0
            first_seen = float(rec.meta.get("first_ts")) if isinstance(rec.meta.get("first_ts"), (int, float)) else -1.0
            score = (duration_h, first_seen)
            if score > best_score:
                best_score = score
                best_rows = rows
                best_sid = sid
        return best_rows, best_sid

    def _nearest_value(self, values: List[float], target: float) -> Optional[float]:
        """Return nearest value to target from a non-empty float list."""
        if not values:
            return None
        return min(values, key=lambda v: abs(v - target))

    def _load_weight_tables_from_py_data(self) -> bool:
        """Load vendored full-fidelity IES/IEC weighting tables from this repository."""
        if self._report_weight_tables:
            return True
        import importlib.util

        def _import_module(path: str, module_name: str):
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        py_data_dir = os.path.join(os.path.dirname(__file__), "reporting_data")
        ies_py = os.path.join(py_data_dir, "iestable.py")
        iec_py = os.path.join(py_data_dir, "iectable.py")
        if not (os.path.isfile(ies_py) and os.path.isfile(iec_py)):
            return False
        try:
            ies_mod = _import_module(ies_py, "_sweep_iestable")
            iec_mod = _import_module(iec_py, "_sweep_iectable")
            if ies_mod is None or iec_mod is None:
                return False
            raw_iestable = list(getattr(ies_mod, "iestab", []) or [])
            eye_rows: List[Tuple[float, float]] = []
            skin_rows: List[Tuple[float, float]] = []
            for row in raw_iestable:
                if not isinstance(row, (tuple, list)) or len(row) < 3:
                    continue
                try:
                    wvl = float(row[0])
                    eye = float(row[1])
                    skin = float(row[2])
                except Exception:
                    continue
                if not (np.isfinite(wvl) and np.isfinite(eye) and np.isfinite(skin)):
                    continue
                if eye <= 0 or skin <= 0:
                    continue
                eye_rows.append((wvl, eye))
                skin_rows.append((wvl, skin))
            if len(eye_rows) < 8 or len(skin_rows) < 8:
                return False
            eye_rows.sort(key=lambda p: p[0])
            skin_rows.sort(key=lambda p: p[0])
            eye_x = np.asarray([p[0] for p in eye_rows], dtype=float)
            eye_y = np.asarray([p[1] for p in eye_rows], dtype=float)
            skin_x = np.asarray([p[0] for p in skin_rows], dtype=float)
            skin_y = np.asarray([p[1] for p in skin_rows], dtype=float)

            iec_data = getattr(iec_mod, "iec_slambda", None)
            iec_x = np.asarray([], dtype=float)
            iec_y = np.asarray([], dtype=float)
            if isinstance(iec_data, (tuple, list)) and len(iec_data) >= 2:
                iec_x = np.asarray(iec_data[0], dtype=float)
                iec_y = np.asarray(iec_data[1], dtype=float)
            if iec_x.size == 0 or iec_y.size == 0:
                raw_iec = list(getattr(iec_mod, "d", []) or [])
                iec_rows: List[Tuple[float, float]] = []
                for row in raw_iec:
                    if not isinstance(row, (tuple, list)) or len(row) < 2:
                        continue
                    try:
                        wvl = float(row[0])
                        val = float(row[1])
                    except Exception:
                        continue
                    if np.isfinite(wvl) and np.isfinite(val) and val > 0:
                        iec_rows.append((wvl, val))
                iec_rows.sort(key=lambda p: p[0])
                iec_x = np.asarray([p[0] for p in iec_rows], dtype=float)
                iec_y = np.asarray([p[1] for p in iec_rows], dtype=float)
            if iec_x.size < 8 or iec_y.size < 8:
                return False
            order = np.argsort(iec_x)
            iec_x = iec_x[order]
            iec_y = iec_y[order]
            keep = np.isfinite(iec_x) & np.isfinite(iec_y) & (iec_y > 0)
            iec_x = iec_x[keep]
            iec_y = iec_y[keep]
            if iec_x.size < 8:
                return False

            self._report_weight_tables = {
                "IES/ANSI Eye": (eye_x, eye_y),
                "IES/ANSI Skin": (skin_x, skin_y),
                "IEC": (iec_x, iec_y),
            }
            self._report_weight_table_source = py_data_dir
            return True
        except Exception:
            return False

    def _weighting_curve_source(self, mode: str) -> str:
        """Describe which weighting curve source is active for a mode."""
        if mode in self._report_weight_tables:
            src = self._report_weight_table_source or "reporting_data"
            return f"vendored tables ({src})"
        return "embedded fallback anchors"

    def _lookup_slambda_value(self, mode: str, wavelength_nm: float) -> float:
        """Lookup S(lambda) for selected standard mode."""
        wvl = float(wavelength_nm)
        if mode == "Unweighted":
            return 1.0 if 200.0 <= wvl <= 400.0 else 0.0

        # Preferred source: vendored full-fidelity py_data tables.
        self._load_weight_tables_from_py_data()
        table = self._report_weight_tables.get(mode)
        if table is not None:
            xs, ys = table
            return float(max(0.0, log_interp_clamped(wvl, xs, ys, zero_at_or_above=400.0)))

        curve = REPORT_EXPOSURE_WEIGHT_CURVES.get(mode)
        if not curve:
            return 1.0
        xs = np.asarray(curve["nm"], dtype=float)
        ys = np.asarray(curve["val"], dtype=float)
        if xs.size == 0:
            return 1.0
        return float(max(0.0, log_interp_clamped(wvl, xs, ys, zero_at_or_above=400.0)))

    def _selected_spectral_point_data(self) -> Optional[Dict[str, object]]:
        """Resolve selected spectrum point into sweep, row, wavelengths, and spectrum arrays."""
        seg = self._resolve_report_segment("spectrum_point")
        if not seg or not seg["rows"]:
            return None
        sweep = seg["sweep"]
        rows = sorted(
            seg["rows"],
            key=lambda r: (
                float(getattr(getattr(r, "coords", None), "lin_mm", 0.0) or 0.0),
                float(getattr(r, "timestamp", 0.0) or 0.0),
            ),
        )
        spec_row = None
        for row in rows:
            spec = getattr(getattr(row, "capture", None), "spectral_result", None)
            if spec is not None and len(spec):
                spec_row = row
                break
        if spec_row is None:
            return None
        wavelengths = np.asarray(getattr(sweep, "spectral_wavelengths", []) or [], dtype=float)
        values = np.asarray(getattr(spec_row.capture, "spectral_result", []) or [], dtype=float)
        if values.size == 0:
            return None
        if wavelengths.size != values.size or wavelengths.size == 0:
            wavelengths = np.arange(values.size, dtype=float)
        return {"segment": seg, "sweep": sweep, "row": spec_row, "wavelengths": wavelengths, "values": values}

    def _compute_exposure_weighting_factor(self, mode: str, wavelengths: np.ndarray, values: np.ndarray) -> float:
        """Compute spectrum-averaged weighting factor for selected exposure mode."""
        if mode == "Unweighted":
            return 1.0
        mask = (wavelengths >= 200.0) & (wavelengths <= 400.0)
        xs = wavelengths[mask]
        ys = values[mask]
        if xs.size < 2:
            return 1.0
        if hasattr(np, "trapezoid"):
            denom = float(np.trapezoid(ys, xs))
        else:
            denom = float(np.trapz(ys, xs))
        if not np.isfinite(denom) or denom <= 0:
            return 1.0
        weights = np.asarray([self._lookup_slambda_value(mode, float(x)) for x in xs], dtype=float)
        if hasattr(np, "trapezoid"):
            numer = float(np.trapezoid(ys * weights, xs))
        else:
            numer = float(np.trapz(ys * weights, xs))
        if not np.isfinite(numer) or numer <= 0:
            return 1.0
        return numer / denom

    def _current_optical_power_band(self) -> Tuple[Optional[Tuple[float, float]], str, Optional[str]]:
        """Return selected optical power band setting and display label."""
        mode = self.report_optical_power_mode_var.get().strip() or "Band"
        if mode == "Full spectrum":
            return None, "full spectrum", None
        lo_txt = self.report_optical_power_min_nm_var.get().strip()
        hi_txt = self.report_optical_power_max_nm_var.get().strip()
        try:
            lo = float(lo_txt)
            hi = float(hi_txt)
        except Exception:
            return None, "band", "Optical power band must be numeric."
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None, "band", "Optical power band must be finite numbers."
        if hi <= lo:
            return None, "band", "Optical power range requires max > min."
        return (float(lo), float(hi)), f"{lo:g}-{hi:g} nm", None

    def _integrate_array_band(self, x_nm: np.ndarray, y_vals: np.ndarray, band: Optional[Tuple[float, float]]) -> float:
        """Integrate spectral array over a wavelength band or full available range."""
        xs = np.asarray(x_nm, dtype=float)
        ys = np.asarray(y_vals, dtype=float)
        if xs.size == 0 or ys.size == 0:
            return 0.0
        if xs.size != ys.size:
            n = min(xs.size, ys.size)
            xs = xs[:n]
            ys = ys[:n]
        if band is not None:
            lo, hi = band
            mask = (xs >= float(lo)) & (xs <= float(hi))
            xs = xs[mask]
            ys = ys[mask]
        if xs.size < 2:
            return float(np.sum(ys))
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(ys, xs))
        return float(np.trapz(ys, xs))

    def _row_band_irradiance_wm2(
        self,
        row,
        sweep,
        band: Optional[Tuple[float, float]],
        require_spectral: bool = False,
    ) -> Optional[float]:
        """Return row irradiance in W/m^2 for requested band; None when unavailable."""
        def _to_reference_1m(value_wm2: float) -> float:
            coords = getattr(row, "coords", None)
            lin_mm = getattr(coords, "lin_mm", None) if coords is not None else None
            if isinstance(lin_mm, (int, float)) and float(lin_mm) > 0:
                return float(value_wm2) * ((float(lin_mm) / 1000.0) ** 2)
            return float(value_wm2)

        capture = getattr(row, "capture", None)
        spectral = getattr(capture, "spectral_result", None) if capture is not None else None
        has_spectral = spectral is not None and len(spectral) > 0
        wavelengths = np.asarray(getattr(sweep, "spectral_wavelengths", []) or [], dtype=float)
        values = np.asarray(spectral or [], dtype=float)

        if band is None:
            if has_spectral:
                if wavelengths.size == values.size and values.size > 0:
                    return _to_reference_1m(self._integrate_array_band(wavelengths, values, None))
            return _to_reference_1m(float(getattr(capture, "integral_result", 0.0) or 0.0))

        lo, hi = band
        if has_spectral:
            if wavelengths.size == values.size and values.size > 0:
                return _to_reference_1m(self._integrate_array_band(wavelengths, values, (float(lo), float(hi))))
        if require_spectral:
            return None
        return _to_reference_1m(float(getattr(capture, "integral_result", 0.0) or 0.0))

    def _integrate_total_power_from_web_rows(
        self,
        sweep,
        rows: List[object],
        band: Optional[Tuple[float, float]],
        require_spectral: bool = False,
    ) -> Optional[float]:
        """Integrate total optical power from web rows over hemisphere (returns watts)."""
        valid_rows = self._valid_sorted_rows(rows)
        if not valid_rows:
            return None
        yaw_vals = sorted(
            {
                round(abs(float(getattr(getattr(r, "coords", None), "yaw_deg", 0.0) or 0.0)), 6)
                for r in valid_rows
                if getattr(getattr(r, "coords", None), "yaw_deg", None) is not None
            }
        )
        if len(yaw_vals) < 2:
            return None
        ystep = float(np.median(np.diff(np.asarray(yaw_vals, dtype=float))))
        if not np.isfinite(ystep) or ystep <= 0:
            return None
        tol = max(1e-4, ystep * 0.25)
        total_w = 0.0
        used_bands = 0
        for y in yaw_vals:
            if y <= 0:
                theta_lo = 0.0
                theta_hi = min(90.0, ystep / 2.0)
            else:
                theta_lo = max(0.0, y - (ystep / 2.0))
                theta_hi = min(90.0, y + (ystep / 2.0))
            if theta_hi <= theta_lo:
                continue
            area_sr = 2.0 * math.pi * (
                math.cos(math.radians(theta_lo)) - math.cos(math.radians(theta_hi))
            )
            band_vals: List[float] = []
            for row in valid_rows:
                coords = getattr(row, "coords", None)
                if coords is None or getattr(coords, "yaw_deg", None) is None:
                    continue
                y_abs = abs(float(coords.yaw_deg))
                if abs(y_abs - y) > tol:
                    continue
                v = self._row_band_irradiance_wm2(row, sweep, band, require_spectral=require_spectral)
                if v is None or not np.isfinite(v) or v < 0:
                    continue
                band_vals.append(float(v))
            if not band_vals:
                continue
            used_bands += 1
            total_w += area_sr * float(np.mean(band_vals))
        if used_bands == 0:
            return None
        return float(total_w)

    def _compute_spectrum_fraction_for_band(
        self,
        wavelengths: np.ndarray,
        values: np.ndarray,
        band: Tuple[float, float],
    ) -> Optional[float]:
        """Return fraction of full-spectrum power inside requested band."""
        denom = self._integrate_array_band(wavelengths, values, None)
        if not np.isfinite(denom) or denom <= 0:
            return None
        numer = self._integrate_array_band(wavelengths, values, band)
        if not np.isfinite(numer) or numer < 0:
            return None
        return float(numer / denom)

    def _estimate_total_optical_power_mw(self) -> Tuple[Optional[float], str]:
        """Estimate total optical power (mW) using selected web and wavelength controls."""
        band, band_label, err = self._current_optical_power_band()
        if err:
            return None, err
        return self._estimate_optical_power_mw_for_band(band, band_label)

    def _estimate_optical_power_mw_for_band(
        self,
        band: Optional[Tuple[float, float]],
        band_label: Optional[str] = None,
    ) -> Tuple[Optional[float], str]:
        """Estimate optical power (mW) for a specific band or full spectrum."""
        if band_label is None:
            if band is None:
                band_label = "full spectrum"
            else:
                band_label = f"{float(band[0]):g}-{float(band[1]):g} nm"
        seg = self._resolve_report_segment("tight_web")
        if not seg or not seg["rows"]:
            seg = self._resolve_report_segment("loose_web")
        if not seg or not seg["rows"]:
            return None, "No web segment selected for optical power integration."

        sweep = seg["sweep"]
        rows = list(seg["rows"])
        total_full_w = self._integrate_total_power_from_web_rows(sweep, rows, band=None, require_spectral=False)
        if total_full_w is None:
            return None, "Could not integrate full-spectrum optical power from selected web rows."

        if band is None:
            return total_full_w * 1000.0, f"{band_label} from segment {seg['segment_id']} (hemisphere web integration)."

        # Preferred path: direct spectral integration across web rows.
        direct_band_w = self._integrate_total_power_from_web_rows(
            sweep,
            rows,
            band=band,
            require_spectral=True,
        )
        if direct_band_w is not None:
            return direct_band_w * 1000.0, f"{band_label} from segment {seg['segment_id']} (direct spectral web integration)."

        # Fallback path used by legacy workflow when web rows are integral-only.
        spectral = self._selected_spectral_point_data()
        if not spectral:
            return None, f"{band_label} requires spectrum-point data when web rows are integral-only."
        frac = self._compute_spectrum_fraction_for_band(
            np.asarray(spectral["wavelengths"], dtype=float),
            np.asarray(spectral["values"], dtype=float),
            band,
        )
        if frac is None:
            return None, f"Could not derive spectral fraction for {band_label}."
        estimate_w = total_full_w * frac
        return (
            estimate_w * 1000.0,
            f"{band_label} estimated from segment {seg['segment_id']} using spectrum-point fraction "
            f"({frac * 100.0:.1f}% of full-spectrum web power).",
        )

    def _integrate_spectrum_band(self, sweep, wavelengths: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
        """Integrate spectrum over wavelength band using sweep helper when available."""
        try:
            return float(sweep.integrate_spectral(list(values), float(lo), float(hi)))
        except Exception:
            mask = (wavelengths >= float(lo)) & (wavelengths <= float(hi))
            if not np.any(mask):
                return 0.0
            xs = wavelengths[mask]
            ys = values[mask]
            if xs.size < 2:
                return float(np.sum(ys))
            return float(np.trapz(ys, xs))

    def _preview_set_message(self, key: str, message: str):
        """Set status message line for one preview tab."""
        slot = self._report_preview_slot(key)
        if not slot:
            return
        msg_var = slot.get("message_var")
        if isinstance(msg_var, tk.StringVar):
            msg_var.set(message)

    def _preview_draw(self, key: str):
        """Redraw one preview figure."""
        slot = self._report_preview_slot(key)
        if not slot:
            return
        canvas = slot.get("canvas")
        if canvas is not None:
            canvas.draw_idle()

    def _open_report_figure_popout(
        self,
        title: str,
        render_cb: Callable[[Figure], None],
        *,
        geometry: str = "980x640",
    ):
        """Open a standalone Matplotlib popout window with toolbar controls."""
        win = tk.Toplevel(self)
        win.title(title)
        try:
            win.geometry(geometry)
        except Exception:
            pass
        shell = ttk.Frame(win, padding=(6, 6))
        shell.pack(fill=tk.BOTH, expand=True)
        fig = Figure(figsize=(9.0, 5.6), dpi=100, tight_layout=True)
        render_cb(fig)
        canvas = FigureCanvasTkAgg(fig, master=shell)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_shell = ttk.Frame(shell)
        toolbar_shell.pack(fill=tk.X, expand=False, pady=(2, 0))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_shell, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)
        ttk.Label(
            toolbar_shell,
            text="Use toolbar controls to pan, box-zoom, reset, and save.",
            style="Hint.TLabel",
        ).pack(side=tk.LEFT, padx=(8, 0))
        canvas.draw_idle()

    def _spectral_cache_ready(self) -> bool:
        """Return True when latest spectral preview cache has plot/table data."""
        return bool(self._report_spectral_cache.get("has_data"))

    def on_report_open_spectral_table_popout(self):
        """Open table-only spectral waveband view in a standalone window."""
        if not self._spectral_cache_ready():
            messagebox.showinfo("Spectral Table", "Refresh spectral preview first.")
            return
        table_rows = list(self._report_spectral_cache.get("table_rows", []) or [])
        if not table_rows:
            messagebox.showinfo("Spectral Table", "No spectral table rows available yet.")
            return

        win = tk.Toplevel(self)
        win.title("Spectral Power Bins")
        try:
            win.geometry("920x520")
        except Exception:
            pass
        shell = ttk.Frame(win, padding=(6, 6))
        shell.pack(fill=tk.BOTH, expand=True)
        cols = ("waveband", "power_mw", "pct_total_uv")
        tv = ttk.Treeview(shell, columns=cols, show="headings", selectmode="browse")
        tv.heading("waveband", text="Waveband")
        tv.heading("power_mw", text="Optical Power (mW)")
        tv.heading("pct_total_uv", text="% of Total UV")
        tv.column("waveband", width=450, anchor="w", stretch=True)
        tv.column("power_mw", width=180, anchor="w", stretch=False)
        tv.column("pct_total_uv", width=150, anchor="w", stretch=False)
        ysb = ttk.Scrollbar(shell, orient=tk.VERTICAL, command=tv.yview)
        tv.configure(yscrollcommand=ysb.set)
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        for row in table_rows:
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            tv.insert("", tk.END, values=(str(row[0]), str(row[1]), str(row[2])))

    def _render_cached_spectral_linear(self, fig: Figure):
        """Render standalone linear spectrum from cache."""
        ax = fig.add_subplot(111)
        wavelengths = np.asarray(self._report_spectral_cache.get("wavelengths", []), dtype=float)
        normalized = np.asarray(self._report_spectral_cache.get("normalized", []), dtype=float)
        weighted_curves = self._report_spectral_cache.get("weighted_curves", {})
        ax.plot(wavelengths, normalized, linewidth=1.4, label="Raw")
        if isinstance(weighted_curves, dict):
            for mode, arr in weighted_curves.items():
                ax.plot(wavelengths, np.asarray(arr, dtype=float), linewidth=1.1, alpha=0.9, label=f"{mode} weighted")
        ax.set_title("Source Spectrum (linear + weighted overlays)", fontsize=11)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Relative intensity (%)")
        if wavelengths.size:
            ax.set_xlim(max(190.0, float(np.min(wavelengths))), min(410.0, float(np.max(wavelengths))))
        y_top = float(self._report_spectral_cache.get("linear_ymax", 110.0) or 110.0)
        ax.set_ylim(0.0, max(110.0, y_top))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    def _render_cached_spectral_log(self, fig: Figure):
        """Render standalone log-detail spectrum from cache."""
        ax = fig.add_subplot(111)
        wavelengths = np.asarray(self._report_spectral_cache.get("wavelengths", []), dtype=float)
        mask_log = np.asarray(self._report_spectral_cache.get("mask_log", []), dtype=bool)
        normalized = np.asarray(self._report_spectral_cache.get("normalized", []), dtype=float)
        weighted_curves = self._report_spectral_cache.get("weighted_curves", {})
        if wavelengths.size == 0 or mask_log.size != wavelengths.size:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No cached log-spectrum data.", ha="center", va="center", transform=ax.transAxes)
            return
        x_log = wavelengths[mask_log]
        if x_log.size == 0:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No cached log-spectrum data.", ha="center", va="center", transform=ax.transAxes)
            return
        ax.plot(x_log, np.clip(normalized[mask_log], 1e-4, None), linewidth=1.4, label="Raw")
        if isinstance(weighted_curves, dict):
            for mode, arr in weighted_curves.items():
                aa = np.asarray(arr, dtype=float)
                if aa.size != wavelengths.size:
                    continue
                ax.plot(x_log, np.clip(aa[mask_log], 1e-4, None), linewidth=1.1, alpha=0.9, label=f"{mode} weighted")
        ax.set_title("Source Spectrum (log detail + weighted overlays)", fontsize=11)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Relative intensity (%)")
        ax.set_yscale("log")
        ax.set_xlim(float(np.min(x_log)), float(np.max(x_log)))
        y_min = float(self._report_spectral_cache.get("log_ymin", 1e-4) or 1e-4)
        y_max = float(self._report_spectral_cache.get("log_ymax", 1.0) or 1.0)
        ax.set_ylim(max(1e-4, y_min), max(1.0, y_max))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    def on_report_open_spectral_linear_popout(self):
        """Open standalone linear spectrum view with full toolbar interaction."""
        if not self._spectral_cache_ready():
            messagebox.showinfo("Spectral Plot", "Refresh spectral preview first.")
            return
        self._open_report_figure_popout(
            "Spectral Plot: Linear",
            self._render_cached_spectral_linear,
        )

    def on_report_open_spectral_log_popout(self):
        """Open standalone log-detail spectrum view with full toolbar interaction."""
        if not self._spectral_cache_ready():
            messagebox.showinfo("Spectral Plot", "Refresh spectral preview first.")
            return
        self._open_report_figure_popout(
            "Spectral Plot: Log Detail",
            self._render_cached_spectral_log,
        )

    def _read_report_csv_frame(self, path: str) -> pd.DataFrame:
        """Load report CSV with fallback for headerless files."""
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, header=None)
        if len(df.columns) >= 3 and all(str(c).replace(".", "", 1).isdigit() for c in df.columns):
            # Likely headerless; reload as numeric columns.
            df = pd.read_csv(path, header=None)
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
        return df

    def _find_csv_column(self, columns: List[object], tokens: List[str]) -> Optional[object]:
        """Find first column containing any token."""
        normalized_tokens = [str(tok).strip().lower() for tok in tokens if str(tok).strip()]
        names = [(col, str(col).strip().lower()) for col in columns]
        for tok in normalized_tokens:
            for col, name in names:
                if name == tok:
                    return col
        for tok in normalized_tokens:
            for col, name in names:
                if name.startswith(tok + "_") or name.endswith("_" + tok) or f"_{tok}_" in name:
                    return col
        for tok in normalized_tokens:
            if len(tok) < 3:
                continue
            for col, name in names:
                if tok in name:
                    return col
        return None

    def _series_numeric(self, df: pd.DataFrame, col: Optional[object]) -> Optional[pd.Series]:
        """Return numeric series for a column or None."""
        if col is None or col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.dropna()
        if s.empty:
            return None
        return s

    def on_report_refresh_previews(self):
        """Recompute all report analysis previews from selected segments and CSVs."""
        summaries: Dict[str, str] = {}
        try:
            summaries["pattern"] = self._refresh_report_pattern_preview()
            summaries["warmup"] = self._refresh_report_warmup_preview()
            summaries["burnin"] = self._refresh_report_burnin_preview()
            summaries["spectral"] = self._refresh_report_spectral_preview()
            summaries["exposure"] = self._refresh_report_exposure_preview()
            summaries["roll"] = self._refresh_report_roll_preview()
            summaries["r2"] = self._refresh_report_r2_preview()
            summaries["electrical"] = self._refresh_report_electrical_preview()
            self._refresh_report_overview_preview(summaries)
            self.report_preview_status_var.set(
                f"Refreshed {time.strftime('%Y-%m-%d %H:%M:%S')} from selected report segments."
            )
            self._set_status("Report analysis previews refreshed.")
        except Exception as e:
            traceback.print_exc()
            self.report_preview_status_var.set(f"Preview refresh failed: {e}")
            self._set_status(f"Report preview refresh failed: {e}")

    def _refresh_report_overview_preview(self, summaries: Dict[str, str]):
        """Update overview text with metadata, selected segments, and computed summaries."""
        meta = self._collect_report_metadata()
        lines: List[str] = []
        lines.append("Report Metadata")
        lines.append(f"- Reporting Name: {meta.get('reporting_name', '') or '(unset)'}")
        lines.append(f"- Catalog ID: {meta.get('catalog_id', '') or '(unset)'}")
        lines.append(f"- Revision: {meta.get('revision_name', '') or '(unset)'} ({meta.get('revision_date', '') or 'date unset'})")
        lines.append("")
        lines.append("Selected Segments")
        for key, label in REPORT_SEGMENT_TYPES:
            seg = self.report_segment_selection.get(key)
            lines.append(f"- {label}: {seg or '(not selected)'}")
        lines.append("")
        lines.append("Preview Summaries")
        lines.append(f"- exposure_mode: {self.report_exposure_mode_var.get().strip() or 'Unweighted'}")
        band, band_label, band_err = self._current_optical_power_band()
        if band_err:
            lines.append(f"- optical_power_band: invalid ({band_err})")
        else:
            lines.append(f"- optical_power_band: {band_label if band is not None else 'full spectrum'}")
        for sec in ("pattern", "warmup", "burnin", "spectral", "exposure", "roll", "r2", "electrical"):
            val = summaries.get(sec, "No summary available.")
            lines.append(f"- {sec}: {val}")
        lines.append("")
        lines.append("Notes")
        for key, label in REPORT_COMMENT_FIELDS:
            txt = self._get_report_comment_text(key)
            lines.append(f"- {label}: {txt if txt else '(none)'}")
        self._set_readonly_text_widget(self._report_overview_text, "\n".join(lines))

    def _refresh_report_pattern_preview(self) -> str:
        """Render a pattern preview from selected web segments."""
        slot = self._report_preview_slot("pattern")
        if not slot:
            return "Pattern tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        ax = fig.add_subplot(111)
        rows, used = self._collect_rows_from_segment_keys(["tight_web", "loose_web"])
        if not rows:
            self._report_pattern_cache = {}
            self._sync_report_pattern_plane_controls([], 0)
            ax.set_axis_off()
            msg = "No loose/tight web rows selected."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("pattern", msg)
            self._preview_draw("pattern")
            return msg

        cache = self._ensure_report_pattern_cache(rows, used)
        msg = self._draw_report_pattern_from_cache(ax, cache)
        self._preview_set_message("pattern", msg)
        self._preview_draw("pattern")
        return msg

    def _refresh_report_warmup_preview(self) -> str:
        """Render warm-up preview from selected warm-up segment."""
        slot = self._report_preview_slot("warmup")
        if not slot:
            return "Warm-up tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        ax = fig.add_subplot(111)
        seg = self._resolve_report_segment("warmup")
        if not seg or not seg["rows"]:
            ax.set_axis_off()
            msg = "No warm-up segment selected."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("warmup", msg)
            self._preview_draw("warmup")
            return msg

        rows = seg["rows"]
        t0 = float(rows[0].timestamp)
        xs = [(float(r.timestamp) - t0) / 60.0 for r in rows]
        ys = [self._row_intensity_uW_cm2(r, normalize_to_1m=True) for r in rows]
        ax.plot(xs, ys, linewidth=1.1)
        ax.set_xlabel("Elapsed time (min)")
        ax.set_ylabel("Irradiance at 1m (uW/cm^2)")
        ax.grid(True, alpha=0.3)

        drift_msg = "Drift unavailable"
        if len(rows) > 2:
            t_last = float(rows[-1].timestamp)
            recent = [(float(r.timestamp), self._row_intensity_uW_cm2(r, True)) for r in rows]
            older = [p for p in recent if (t_last - p[0]) >= (10 * 60)]
            if older:
                base = older[-1][1]
                last = recent[-1][1]
                if abs(last) > 1e-9:
                    drift_pct = abs(last - base) / abs(last) * 100.0
                    drift_msg = f"10-min drift {drift_pct:.2f}%"
        duration_h = (float(rows[-1].timestamp) - float(rows[0].timestamp)) / 3600.0 if len(rows) >= 2 else 0.0
        msg = f"Warm-up duration {duration_h:.2f}h from segment {seg['segment_id']}. {drift_msg}."
        self._preview_set_message("warmup", msg)
        self._preview_draw("warmup")
        return msg

    def _refresh_report_burnin_preview(self) -> str:
        """Render long-duration warm-up/burn-in preview from complete dataset scans."""
        slot = self._report_preview_slot("burnin")
        if not slot:
            return "Burn-in tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        ax = fig.add_subplot(111)
        rows, sid = self._select_burnin_rows()
        if len(rows) < 2:
            ax.set_axis_off()
            msg = "No complete burn-in warm-up run found (need a long warm-up sequence)."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("burnin", msg)
            self._preview_draw("burnin")
            return msg

        t0 = float(rows[0].timestamp)
        xs = [(float(r.timestamp) - t0) / 3600.0 for r in rows]
        ys = [self._row_intensity_uW_cm2(r, normalize_to_1m=True) for r in rows]
        ax.plot(xs, ys, linewidth=1.1)
        ax.set_xlabel("Elapsed time (h)")
        ax.set_ylabel("Irradiance at 1m (uW/cm^2)")
        ax.grid(True, alpha=0.3)

        final = ys[-1] if ys else 0.0
        if final > 0:
            for pct in (20, 40, 60, 80, 100):
                ax.axhline(final * (pct / 100.0), linestyle="--", linewidth=0.8, color="gray", alpha=0.4)
        duration_h = xs[-1] if xs else 0.0
        msg = f"Burn-in preview from scan {sid}: {duration_h:.1f}h warm-up trajectory."
        self._preview_set_message("burnin", msg)
        self._preview_draw("burnin")
        return msg

    def _refresh_report_spectral_preview(self) -> str:
        """Render spectral plots with weighted overlays and wavelength-bin power table."""
        slot = self._report_preview_slot("spectral")
        if not slot:
            return "Spectral tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        fig.set_tight_layout(False)
        gspec = fig.add_gridspec(3, 1, height_ratios=[1.75, 1.95, 1.95], hspace=0.58)
        fig.subplots_adjust(top=0.98, bottom=0.08)
        ax_table = fig.add_subplot(gspec[0])
        ax_linear = fig.add_subplot(gspec[1])
        ax_log = fig.add_subplot(gspec[2])
        ax_table.set_navigate(False)

        spectral = self._selected_spectral_point_data()
        if not spectral:
            self._report_spectral_cache = {"has_data": False}
            ax_table.set_axis_off()
            ax_linear.set_axis_off()
            ax_log.set_axis_off()
            msg = "No spectrum-point segment selected."
            ax_linear.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_linear.transAxes)
            self._preview_set_message("spectral", msg)
            self._preview_draw("spectral")
            return msg

        sweep = spectral["sweep"]
        wavelengths = np.asarray(spectral["wavelengths"], dtype=float)
        values = np.asarray(spectral["values"], dtype=float)
        vmax = float(np.max(values)) if values.size else 0.0
        if vmax <= 0:
            self._report_spectral_cache = {"has_data": False}
            ax_table.set_axis_off()
            ax_linear.set_axis_off()
            ax_log.set_axis_off()
            msg = "Spectrum values are empty or zero."
            ax_linear.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_linear.transAxes)
            self._preview_set_message("spectral", msg)
            self._preview_draw("spectral")
            return msg

        normalized = (values / vmax) * 100.0
        weighted_curves: Dict[str, np.ndarray] = {}
        weighting_sources: List[str] = []
        for mode in ("IES/ANSI Eye", "IES/ANSI Skin", "IEC"):
            weights = np.asarray([self._lookup_slambda_value(mode, float(w)) for w in wavelengths], dtype=float)
            ref_222 = max(self._lookup_slambda_value(mode, 222.0), 1e-12)
            weighted_curves[mode] = normalized * (weights / ref_222)
            weighting_sources.append(f"{mode}: {self._weighting_curve_source(mode)}")

        all_curve_arrays = [normalized] + list(weighted_curves.values())
        all_max = max(float(np.nanmax(np.asarray(c, dtype=float))) for c in all_curve_arrays)

        ax_linear.plot(wavelengths, normalized, linewidth=1.25, label="Raw")
        for mode, arr in weighted_curves.items():
            ax_linear.plot(wavelengths, arr, linewidth=1.0, alpha=0.88, label=f"{mode} weighted")
        ax_linear.set_title("Source Spectrum (linear + weighted overlays)", fontsize=10, pad=10)
        ax_linear.set_xlabel("Wavelength (nm)")
        ax_linear.set_ylabel("Relative intensity (%)")
        linear_xmin = max(190.0, float(np.min(wavelengths)))
        linear_xmax = min(410.0, float(np.max(wavelengths)))
        linear_ymax = max(110.0, all_max * 1.08)
        ax_linear.set_xlim(linear_xmin, linear_xmax)
        ax_linear.set_ylim(0.0, linear_ymax)
        ax_linear.grid(True, alpha=0.3)
        ax_linear.legend(loc="upper right", fontsize=7.8, ncol=2)

        mask_log = (wavelengths >= 205.0) & (wavelengths <= 260.0)
        if int(np.sum(mask_log)) < 2:
            mask_log = np.isfinite(wavelengths)
        x_log = wavelengths[mask_log]
        curve_logs: List[np.ndarray] = [np.clip(normalized[mask_log], 1e-4, None)]
        ax_log.plot(x_log, curve_logs[0], linewidth=1.25, label="Raw")
        for mode, arr in weighted_curves.items():
            arr_log = np.clip(arr[mask_log], 1e-4, None)
            curve_logs.append(arr_log)
            ax_log.plot(x_log, arr_log, linewidth=1.0, alpha=0.88, label=f"{mode} weighted")
        y_min = min(float(np.nanmin(c)) for c in curve_logs)
        y_max = max(float(np.nanmax(c)) for c in curve_logs)
        ax_log.set_title("Source Spectrum (log detail + weighted overlays)", fontsize=10, pad=8)
        ax_log.set_xlabel("Wavelength (nm)")
        ax_log.set_ylabel("Relative intensity (%)")
        ax_log.set_yscale("log")
        if x_log.size >= 2:
            ax_log.set_xlim(float(np.min(x_log)), float(np.max(x_log)))
        log_ymin = max(1e-4, y_min * 0.9)
        log_ymax = max(1.0, y_max * 1.2)
        ax_log.set_ylim(log_ymin, log_ymax)
        ax_log.grid(True, which="both", alpha=0.3)
        ax_log.legend(loc="upper right", fontsize=7.8, ncol=2)

        total_uv = self._integrate_spectrum_band(sweep, wavelengths, values, 200, 400)
        far = self._integrate_spectrum_band(sweep, wavelengths, values, 200, 240)
        high_s = self._integrate_spectrum_band(sweep, wavelengths, values, 240, 300)
        uvc = self._integrate_spectrum_band(sweep, wavelengths, values, 200, 280)
        uvb = self._integrate_spectrum_band(sweep, wavelengths, values, 280, 315)
        uva = self._integrate_spectrum_band(sweep, wavelengths, values, 315, 400)

        # Power bins table (legacy-style wavebands + selected/current integration totals).
        ax_table.axis("off")
        bin_cache: Dict[Tuple[float, float], Tuple[Optional[float], str]] = {}
        for _name, band in REPORT_SPECTRAL_POWER_BINS:
            bin_cache[band] = self._estimate_optical_power_mw_for_band(band, f"{band[0]:g}-{band[1]:g} nm")
        total_uv_mw = bin_cache[(200.0, 400.0)][0]
        full_spectrum_mw, full_spectrum_ctx = self._estimate_optical_power_mw_for_band(None, "full spectrum")
        selected_mw, selected_ctx = self._estimate_total_optical_power_mw()
        selected_band, selected_band_label, _selected_err = self._current_optical_power_band()

        table_rows: List[List[str]] = []
        for name, band in REPORT_SPECTRAL_POWER_BINS:
            p_mw, _ctx = bin_cache.get(band, (None, ""))
            band_txt = f"{name} ({band[0]:g}nm-{band[1]:g}nm)"
            p_txt = f"{p_mw:.1f}" if isinstance(p_mw, (int, float)) and np.isfinite(float(p_mw)) else "n/a"
            if (
                isinstance(p_mw, (int, float))
                and isinstance(total_uv_mw, (int, float))
                and np.isfinite(float(total_uv_mw))
                and float(total_uv_mw) > 0
            ):
                pct_txt = f"{(float(p_mw) / float(total_uv_mw)) * 100.0:.1f}"
            else:
                pct_txt = "n/a"
            table_rows.append([band_txt, p_txt, pct_txt])

        if isinstance(full_spectrum_mw, (int, float)) and np.isfinite(float(full_spectrum_mw)):
            if isinstance(total_uv_mw, (int, float)) and np.isfinite(float(total_uv_mw)) and float(total_uv_mw) > 0:
                full_pct_txt = f"{(float(full_spectrum_mw) / float(total_uv_mw)) * 100.0:.1f}"
            else:
                full_pct_txt = "n/a"
            table_rows.append(["Full Spectrum (all wavelengths)", f"{float(full_spectrum_mw):.1f}", full_pct_txt])

        if isinstance(selected_mw, (int, float)) and np.isfinite(float(selected_mw)):
            sel_name = (
                f"Selected Integration ({selected_band_label})"
                if selected_band is not None
                else "Selected Integration (full spectrum)"
            )
            if isinstance(total_uv_mw, (int, float)) and np.isfinite(float(total_uv_mw)) and float(total_uv_mw) > 0:
                sel_pct_txt = f"{(float(selected_mw) / float(total_uv_mw)) * 100.0:.1f}"
            else:
                sel_pct_txt = "n/a"
            table_rows.append([sel_name, f"{float(selected_mw):.1f}", sel_pct_txt])

        tbl = ax_table.table(
            cellText=table_rows,
            colLabels=["Waveband", "Optical Power (mW)", "% of Total UV"],
            cellLoc="left",
            colLoc="left",
            loc="upper center",
            bbox=[0.0, 0.0, 1.0, 0.98],
            colWidths=[0.50, 0.26, 0.24],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.8)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
            if c in (1, 2):
                cell.get_text().set_ha("right")
            cell.PAD = 0.10

        if total_uv > 0:
            msg = (
                f"UV fractions from spectrum point: UVC {((uvc / total_uv) * 100):.1f}%, "
                f"UVB {((uvb / total_uv) * 100):.1f}%, UVA {((uva / total_uv) * 100):.1f}%, "
                f"Far-UVC {((far / total_uv) * 100):.1f}%, High-S(lambda) {((high_s / total_uv) * 100):.1f}%."
            )
        else:
            msg = "Could not compute UV fractions from selected spectrum point."
        if selected_mw is not None:
            msg = f"{msg} Selected optical power: {selected_mw:.1f} mW ({selected_ctx})"
        else:
            msg = f"{msg} Selected optical power unavailable ({selected_ctx})"
        if full_spectrum_mw is not None:
            msg = f"{msg} Full-spectrum optical power: {full_spectrum_mw:.1f} mW ({full_spectrum_ctx})"
        if total_uv_mw is not None:
            msg = f"{msg} Total UV (200-400nm): {float(total_uv_mw):.1f} mW."
        source_line = "; ".join(weighting_sources)
        msg = (
            f"{msg} Weighted overlays shown on linear/log plots: raw + IES/ANSI Eye + "
            "IES/ANSI Skin + IEC (each normalized to S(222nm)). "
            f"Curve source: {source_line}. "
            "Use 'Open Linear' / 'Open Log' for dedicated pan/zoom views."
        )
        self._report_spectral_cache = {
            "has_data": True,
            "wavelengths": np.asarray(wavelengths, dtype=float),
            "normalized": np.asarray(normalized, dtype=float),
            "weighted_curves": {k: np.asarray(v, dtype=float) for k, v in weighted_curves.items()},
            "mask_log": np.asarray(mask_log, dtype=bool),
            "linear_ymax": float(linear_ymax),
            "log_ymin": float(log_ymin),
            "log_ymax": float(log_ymax),
            "table_rows": [list(r) for r in table_rows],
        }
        self._preview_set_message("spectral", msg)
        self._preview_draw("spectral")
        return msg

    def _refresh_report_exposure_preview(self) -> str:
        """Render exposure-distance preview with selectable weighting mode."""
        slot = self._report_preview_slot("exposure")
        if not slot:
            return "Exposure tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        ax = fig.add_subplot(111)
        mode = self.report_exposure_mode_var.get().strip() or "Unweighted"
        rows, _used = self._collect_rows_from_segment_keys(["tight_web", "loose_web"])
        if not rows:
            ax.set_axis_off()
            msg = "No web segments selected for exposure preview."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("exposure", msg)
            self._preview_draw("exposure")
            return msg

        peak_1m = max(self._row_intensity_uW_cm2(r, normalize_to_1m=True) for r in rows)
        if peak_1m <= 0:
            ax.set_axis_off()
            msg = "Peak intensity unavailable."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("exposure", msg)
            self._preview_draw("exposure")
            return msg

        weight_factor = 1.0
        source = "unweighted baseline"
        if mode != "Unweighted":
            spectral = self._selected_spectral_point_data()
            if spectral:
                weight_factor = self._compute_exposure_weighting_factor(
                    mode,
                    np.asarray(spectral["wavelengths"], dtype=float),
                    np.asarray(spectral["values"], dtype=float),
                )
                source = self._weighting_curve_source(mode)
            else:
                source = "no spectrum point selected; fallback factor=1.0"

        weighted_peak_1m = peak_1m * weight_factor
        exposure_limit_uj_cm2 = 3000.0
        horizon_h = 8.0
        allowed_rate = exposure_limit_uj_cm2 / (horizon_h * 3600.0)
        fail_safe_m = math.sqrt(weighted_peak_1m / allowed_rate) if weighted_peak_1m > 0 else float("nan")
        base_distances = np.array([0.5, 1.0, 2.0, 3.0], dtype=float)
        base_irradiances = weighted_peak_1m / (base_distances ** 2)
        base_times_h = exposure_limit_uj_cm2 / (base_irradiances * 3600.0)

        ax.plot(base_distances, base_irradiances, marker="o", linewidth=1.1)
        if np.isfinite(fail_safe_m):
            ax.axvline(fail_safe_m, linestyle="--", color="gray", alpha=0.5)
            fail_safe_irr = weighted_peak_1m / (fail_safe_m ** 2)
            ax.scatter([fail_safe_m], [fail_safe_irr], marker="D", s=32, color="#1f77b4", zorder=4)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(f"Worst-case irradiance ({mode}, uW/cm^2)")
        ax.set_title("Exposure Distance Model: E(d) = E(1m)/d^2", fontsize=10)
        ax.grid(True, alpha=0.3)

        text_rows = []
        for d, irr, t_h in zip(base_distances, base_irradiances, base_times_h):
            label = f"{d:.1f}m"
            if t_h < horizon_h:
                t_txt = f"{t_h:.2f}h"
            else:
                t_txt = ">=8h"
            text_rows.append(f"{label}: {irr:.2f} uW/cm^2, limit in {t_txt}")
        fail_safe_text = f"Fail-safe distance={fail_safe_m:.3f}m." if np.isfinite(fail_safe_m) else "Fail-safe distance unavailable."
        msg = (
            f"{mode} exposure preview using 3 mJ/cm^2 (8h) threshold. "
            f"Spectrum-weight factor={weight_factor:.3f} ({source}). "
            f"Weighted peak at 1m={weighted_peak_1m:.2f} uW/cm^2. "
            + f"{fail_safe_text} "
            + f"Examples: {' | '.join(text_rows[:3])}. "
            + "Interpretation: distances at or beyond fail-safe are expected to remain within an 8-hour limit."
        )
        self._preview_set_message("exposure", msg)
        self._preview_draw("exposure")
        return msg

    def _refresh_report_roll_preview(self) -> str:
        """Render roll-dependence preview from selected web rows."""
        slot = self._report_preview_slot("roll")
        if not slot:
            return "Roll tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        ax = fig.add_subplot(111)
        rows, _used = self._collect_rows_from_segment_keys(["tight_web", "loose_web"])
        if not rows:
            ax.set_axis_off()
            msg = "No web rows selected for roll-dependence preview."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("roll", msg)
            self._preview_draw("roll")
            return msg

        roll_map: Dict[float, List[float]] = {}
        for row in rows:
            coords = getattr(row, "coords", None)
            if coords is None:
                continue
            yaw = getattr(coords, "yaw_deg", None)
            roll = getattr(coords, "roll_deg", None)
            if yaw is None or roll is None:
                continue
            if abs(float(yaw)) > 1e-3:
                continue
            val = self._row_intensity_uW_cm2(row, normalize_to_1m=True)
            roll_map.setdefault(float(roll), []).append(val)
        if not roll_map:
            ax.set_axis_off()
            msg = "No yaw=0 rows in selected web segments."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("roll", msg)
            self._preview_draw("roll")
            return msg

        xs = sorted(roll_map.keys())
        ys = [float(np.mean(roll_map[x])) for x in xs]
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.1)
        ax.set_xlabel("Roll (deg)")
        ax.set_ylabel("Irradiance at 1m (uW/cm^2)")
        ax.grid(True, alpha=0.3)

        zero_roll = self._nearest_value(xs, 0.0)
        max_err = 0.0
        if zero_roll is not None:
            zero_v = ys[xs.index(zero_roll)]
            if abs(zero_v) > 1e-9:
                max_err = max(abs(y - zero_v) / abs(zero_v) * 100.0 for y in ys)
        msg = f"Roll-dependence built from {len(xs)} roll points. Max deviation vs 0 deg: {max_err:.1f}%."
        self._preview_set_message("roll", msg)
        self._preview_draw("roll")
        return msg

    def _refresh_report_r2_preview(self) -> str:
        """Render R² pullback preview."""
        slot = self._report_preview_slot("r2")
        if not slot:
            return "R² tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        ax = fig.add_subplot(111)
        seg = self._resolve_report_segment("r2_pullback")
        if not seg or not seg["rows"]:
            ax.set_axis_off()
            msg = "No R² pullback segment selected."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("r2", msg)
            self._preview_draw("r2")
            return msg

        points: Dict[float, float] = {}
        for row in seg["rows"]:
            coords = getattr(row, "coords", None)
            if coords is None:
                continue
            yaw = getattr(coords, "yaw_deg", None)
            roll = getattr(coords, "roll_deg", None)
            dist_mm = getattr(coords, "lin_mm", None)
            if yaw is None or roll is None or dist_mm is None:
                continue
            if abs(float(yaw)) > 1e-3 or abs(float(roll)) > 1e-3:
                continue
            if float(dist_mm) <= 0:
                continue
            points[float(dist_mm)] = float(getattr(row.capture, "integral_result", 0.0) or 0.0) * WM2_TO_UW_CM2
        if len(points) < 2:
            ax.set_axis_off()
            msg = "Need at least two 0 deg yaw/roll pullback points."
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
            self._preview_set_message("r2", msg)
            self._preview_draw("r2")
            return msg

        xs = np.asarray(sorted(points.keys()), dtype=float)
        ys = np.asarray([points[x] for x in xs], dtype=float)
        y_fit = ys[-1] * (xs[-1] / xs) ** 2
        ax.plot(xs, y_fit, linewidth=1.1, label="Ideal 1/r^2")
        ax.scatter(xs, ys, s=18, marker="x", label="Measured")
        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel("Irradiance (uW/cm^2)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        ss_res = float(np.sum((ys - y_fit) ** 2))
        ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        msg = f"R² pullback points: {len(xs)}. Model fit score (R²): {r2:.4f}."
        self._preview_set_message("r2", msg)
        self._preview_draw("r2")
        return msg

    def _refresh_report_electrical_preview(self) -> str:
        """Render electrical preview from selected report CSV roles."""
        slot = self._report_preview_slot("electrical")
        if not slot:
            return "Electrical tab unavailable."
        fig: Figure = slot["figure"]  # type: ignore[assignment]
        fig.clear()
        ax_top = fig.add_subplot(211)
        ax_bottom = fig.add_subplot(212)
        ax_bottom_twin = ax_bottom.twinx()

        power_log = next((rec for rec in self.report_csvs.values() if rec.role == "Power Log"), None)
        power_wfm = next((rec for rec in self.report_csvs.values() if rec.role == "Power Waveform"), None)

        summary_lines: List[str] = []
        plotted = False

        if power_log is not None:
            try:
                df = self._read_report_csv_frame(power_log.path)
                cols = list(df.columns)
                ts_col = self._find_csv_column(cols, ["timestamp", "epoch", "time"])
                active_col = self._find_csv_column(cols, ["w_active", "active"])
                apparent_col = self._find_csv_column(cols, ["w_apparent", "apparent"])
                v_col = self._find_csv_column(cols, ["v_rms", "vrms", "voltage"])
                i_col = self._find_csv_column(cols, ["i_rms", "irms", "current"])
                pf_col = self._find_csv_column(cols, ["pfactor", "power_factor", "pf"])
                freq_col = self._find_csv_column(cols, ["v_freq", "freq", "hz"])
                vthd_col = self._find_csv_column(cols, ["v_thd", "voltage_thd", "v_thd-f"])
                ithd_col = self._find_csv_column(cols, ["i_thd", "current_thd", "i_thd-f"])

                s_ts = self._series_numeric(df, ts_col)
                s_active = self._series_numeric(df, active_col)
                s_apparent = self._series_numeric(df, apparent_col)
                power_series = s_active if s_active is not None else s_apparent
                if s_ts is not None and power_series is not None:
                    n = min(len(s_ts), len(power_series))
                    x = s_ts.iloc[:n].to_numpy(dtype=float)
                    y = power_series.iloc[:n].to_numpy(dtype=float)
                    if n > 1:
                        x = (x - x[0]) / 60.0
                        ax_top.plot(x, y, linewidth=1.0)
                        ax_top.set_xlabel("Elapsed time (min)")
                        ax_top.set_ylabel("Power (W)")
                        ax_top.grid(True, alpha=0.3)
                        plotted = True

                if s_active is not None:
                    summary_lines.append(f"Active power avg: {float(np.mean(s_active)):.2f} W")
                if s_apparent is not None:
                    summary_lines.append(f"Apparent power avg: {float(np.mean(s_apparent)):.2f} W")
                s_v = self._series_numeric(df, v_col)
                s_i = self._series_numeric(df, i_col)
                s_pf = self._series_numeric(df, pf_col)
                s_freq = self._series_numeric(df, freq_col)
                s_vthd = self._series_numeric(df, vthd_col)
                s_ithd = self._series_numeric(df, ithd_col)
                if s_v is not None:
                    summary_lines.append(f"Voltage RMS avg: {float(np.mean(s_v)):.2f} V")
                if s_i is not None:
                    summary_lines.append(f"Current RMS avg: {float(np.mean(s_i)):.3f} A")
                if s_pf is not None:
                    summary_lines.append(f"Power factor avg: {float(np.mean(s_pf)):.3f}")
                if s_freq is not None:
                    summary_lines.append(f"Frequency avg: {float(np.mean(s_freq)):.3f} Hz")
                if s_vthd is not None:
                    summary_lines.append(f"Voltage THD max: {float(np.max(s_vthd))*100.0:.2f}%")
                if s_ithd is not None:
                    summary_lines.append(f"Current THD avg: {float(np.mean(s_ithd))*100.0:.2f}%")
            except Exception as e:
                summary_lines.append(f"Power log parse failed: {e}")

        if power_wfm is not None:
            try:
                dfw = self._read_report_csv_frame(power_wfm.path)
                cols = list(dfw.columns)
                t_col = self._find_csv_column(cols, ["time", "timestamp", "t"]) or (cols[0] if cols else None)
                v_col = self._find_csv_column(cols, ["voltage", "v_rms", "vrms", "v"]) or (cols[1] if len(cols) > 1 else None)
                i_col = self._find_csv_column(cols, ["current", "i_rms", "irms", "i"]) or (cols[2] if len(cols) > 2 else None)
                s_t = self._series_numeric(dfw, t_col)
                s_v = self._series_numeric(dfw, v_col)
                s_i = self._series_numeric(dfw, i_col)
                if s_t is not None and s_v is not None and s_i is not None:
                    n = min(len(s_t), len(s_v), len(s_i))
                    if n > 1:
                        x = s_t.iloc[:n].to_numpy(dtype=float) * 1000.0
                        yv = s_v.iloc[:n].to_numpy(dtype=float)
                        yi = s_i.iloc[:n].to_numpy(dtype=float)
                        ax_bottom.plot(x, yi, linewidth=1.0, color="#2a9d8f", label="Current")
                        ax_bottom_twin.plot(x, yv, linewidth=1.0, color="#264653", label="Voltage")
                        ax_bottom.set_xlabel("Waveform time (ms)")
                        ax_bottom.set_ylabel("Current (A)", color="#2a9d8f")
                        ax_bottom_twin.set_ylabel("Voltage (V)", color="#264653")
                        ax_bottom.grid(True, alpha=0.3)
                        plotted = True
                else:
                    summary_lines.append("Waveform CSV missing usable Time/V/I columns.")
            except Exception as e:
                summary_lines.append(f"Waveform parse failed: {e}")

        if not plotted:
            ax_top.set_axis_off()
            ax_bottom.set_axis_off()
            ax_bottom_twin.set_axis_off()
            ax_top.text(0.5, 0.5, "No electrical preview data.", ha="center", va="center", transform=ax_top.transAxes)

        summary = "\n".join(summary_lines) if summary_lines else "No electrical metrics available."
        self.report_electrical_summary_var.set(summary)
        msg = "Electrical preview refreshed."
        self._preview_set_message("electrical", msg)
        self._preview_draw("electrical")
        return summary_lines[0] if summary_lines else msg

    # ---- Report Builder actions ----
    def on_add_report_sw3_files(self):
        """Prompt for SW3 scans and add them to the Report Builder list."""
        paths = filedialog.askopenfilenames(
            title="Add SW3/eegbin Files (Report Builder)",
            filetypes=[("SW3/eegbin", "*.sw3 *.eegbin *.bin *.*")],
        )
        if not paths:
            return
        ensure_eegbin_imported()
        n = 0
        for path in paths:
            sweep = self._load_sweep_from_path(path, warn=True, trace=True)
            if sweep is None:
                continue
            meta = self._meta_from_sweep(sweep)
            phases = self._summarize_scan_phases(sweep)
            meta["phase_count"] = len(phases)
            inferred_role = suggest_report_scan_role(phases)
            phase_tags = {}
            for ph in phases:
                idx = int(ph.get("index", 0))
                tag = suggest_report_phase_tag(ph)
                if tag and tag != "Unassigned":
                    phase_tags[idx] = tag
            scan_id = self._next_report_scan_id()
            self.report_scans[scan_id] = ReportScanRecord(
                scan_id=scan_id,
                path=os.path.abspath(path),
                label=os.path.basename(path),
                role=inferred_role,
                meta=meta,
                phases=phases,
                phase_tags=phase_tags,
            )
            self._report_sweep_cache[scan_id] = sweep
            n += 1
        if n:
            if not self.report_meta_vars["reporting_name"].get().strip() and self.report_scans:
                first = next(iter(self.report_scans.values()))
                self.report_meta_vars["reporting_name"].set(first.label)
                if not self.report_meta_vars["acq_name"].get().strip():
                    self.report_meta_vars["acq_name"].set(first.label)
            self._refresh_report_scans_tv()
            self._refresh_report_segment_selectors()
            self._set_status(f"Added {n} report scan(s).")

    def on_add_report_csv_files(self):
        """Prompt for CSV files and add them to the Report Builder list."""
        paths = filedialog.askopenfilenames(
            title="Add CSV Files (Report Builder)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not paths:
            return
        n = 0
        for path in paths:
            meta = self._load_report_csv_meta(path)
            suggested_role = self._suggest_report_csv_role(path, meta)
            csv_id = self._next_report_csv_id()
            self.report_csvs[csv_id] = ReportCsvRecord(
                csv_id=csv_id,
                path=os.path.abspath(path),
                label=os.path.basename(path),
                role=suggested_role,
                meta=meta,
            )
            n += 1
        if n:
            self._refresh_report_csvs_tv()
            self._set_status(f"Added {n} report CSV file(s).")

    def on_remove_report_scans(self):
        """Remove selected Report Builder scans."""
        sel = list(self.tv_report_scans.selection())
        if not sel:
            messagebox.showinfo("Remove", "Select one or more scans to remove.")
            return
        if not messagebox.askyesno("Remove", f"Remove {len(sel)} scan(s) from Report Builder?"):
            return
        for sid in sel:
            self.report_scans.pop(sid, None)
            self._report_sweep_cache.pop(sid, None)
            if self._report_selected_scan_id == sid:
                self._report_selected_scan_id = None
        self._refresh_report_scans_tv()
        self._refresh_report_segment_selectors()

    def on_remove_report_csvs(self):
        """Remove selected Report Builder CSVs."""
        sel = list(self.tv_report_csvs.selection())
        if not sel:
            messagebox.showinfo("Remove", "Select one or more CSVs to remove.")
            return
        if not messagebox.askyesno("Remove", f"Remove {len(sel)} CSV file(s) from Report Builder?"):
            return
        for cid in sel:
            self.report_csvs.pop(cid, None)
        self._refresh_report_csvs_tv()

    def on_rename_report_scan(self):
        """Rename the selected Report Builder scan label."""
        sel = self.tv_report_scans.selection()
        if not sel:
            return
        sid = sel[0]
        rec = self.report_scans.get(sid)
        if rec is None:
            return
        new = simpledialog.askstring("Rename Scan", "New label:", initialvalue=rec.label, parent=self)
        if new:
            rec.label = new.strip()
            self._refresh_report_scans_tv()
            self._refresh_report_segment_selectors()
            if self._report_selected_scan_id == sid:
                self.report_scan_info_var.set(f"{rec.label} ({sid})")

    def on_rename_report_csv(self):
        """Rename the selected Report Builder CSV label."""
        sel = self.tv_report_csvs.selection()
        if not sel:
            return
        cid = sel[0]
        rec = self.report_csvs.get(cid)
        if rec is None:
            return
        new = simpledialog.askstring("Rename CSV", "New label:", initialvalue=rec.label, parent=self)
        if new:
            rec.label = new.strip()
            self._refresh_report_csvs_tv()

    def on_report_set_scan_role(self):
        """Assign the selected scan role."""
        sid = self._current_report_scan_id()
        if not sid:
            messagebox.showinfo("Scan Role", "Select a scan first.")
            return
        rec = self.report_scans.get(sid)
        if rec is None:
            return
        rec.role = self.report_scan_role_var.get() or "Unassigned"
        self._refresh_report_scans_tv()

    def on_report_apply_role_to_phases(self):
        """Apply the current scan role to all phases of the selected scan."""
        sid = self._current_report_scan_id()
        if not sid:
            messagebox.showinfo("Apply Role", "Select a scan first.")
            return
        rec = self.report_scans.get(sid)
        if rec is None:
            return
        if not rec.phases:
            messagebox.showinfo("Apply Role", "Selected scan has no phases to tag.")
            return
        role = self.report_scan_role_var.get() or "Unassigned"
        role_to_phase = {
            "Warm-up Run": "Warm-up",
            "Startup Scan": "Warm-up",
            "Spectrum Scan": "Spectrum Point",
            "R2 Validation Scan": "R2 Pullback",
            "Burn-in Scan": "Burn-in",
        }
        phase_tag = role_to_phase.get(role)
        if not phase_tag:
            messagebox.showinfo(
                "Apply Role",
                "The selected role is scan-level only. Use phase tagging or Auto-tag for segment tags.",
            )
            return
        for ph in rec.phases:
            idx = int(ph.get("index", 0))
            rec.phase_tags[idx] = phase_tag
        self._refresh_report_phases_tv(rec)
        self._refresh_report_segment_selectors()

    def on_report_set_phase_tag(self):
        """Set tag for selected phases of the active scan."""
        sid = self._current_report_scan_id()
        if not sid:
            messagebox.showinfo("Phase Tag", "Select a scan first.")
            return
        rec = self.report_scans.get(sid)
        if rec is None:
            return
        sel = list(self.tv_report_phases.selection())
        if not sel:
            messagebox.showinfo("Phase Tag", "Select one or more phases to tag.")
            return
        tag = self.report_phase_tag_var.get() or "Unassigned"
        for iid in sel:
            try:
                idx = int(iid)
            except Exception:
                continue
            rec.phase_tags[idx] = tag
        self._refresh_report_phases_tv(rec)
        self._refresh_report_segment_selectors()

    def on_report_auto_tag_phases(self):
        """Auto-tag phases based on phase name/type heuristics."""
        sid = self._current_report_scan_id()
        if not sid:
            messagebox.showinfo("Auto-tag", "Select a scan first.")
            return
        rec = self.report_scans.get(sid)
        if rec is None:
            return
        if not rec.phases:
            messagebox.showinfo("Auto-tag", "Selected scan has no phases to tag.")
            return
        for ph in rec.phases:
            idx = int(ph.get("index", 0))
            suggested = self._suggest_report_phase_tag(ph)
            if suggested and suggested != "Unassigned":
                rec.phase_tags[idx] = suggested
        rec.role = suggest_report_scan_role(rec.phases)
        self.report_scan_role_var.set(rec.role or "Unassigned")
        self._refresh_report_scans_tv()
        self._refresh_report_phases_tv(rec)
        self._refresh_report_segment_selectors()

    def on_report_set_csv_role(self):
        """Assign the selected CSV role."""
        cid = self._current_report_csv_id()
        if not cid:
            messagebox.showinfo("CSV Role", "Select a CSV first.")
            return
        rec = self.report_csvs.get(cid)
        if rec is None:
            return
        rec.role = self.report_csv_role_var.get() or "Other"
        self._refresh_report_csvs_tv()

    def on_report_set_lamp_image(self):
        """Choose the lamp photo image used in report output."""
        path = self._choose_report_image_path("Select Lamp Photo")
        if path:
            self._set_report_image_path("lamp", path)

    def on_report_set_axes_image(self):
        """Choose the axes-labeled photo used in report output."""
        path = self._choose_report_image_path("Select Axes Photo")
        if path:
            self._set_report_image_path("axes", path)

    def _report_image_initial_dir(self) -> str:
        """Best-effort initial directory for image picker."""
        for candidate in (
            self.report_lamp_image_path_var.get().strip(),
            self.report_axes_image_path_var.get().strip(),
        ):
            if candidate:
                folder = os.path.dirname(candidate)
                if os.path.isdir(folder):
                    return folder
        sid = self._current_report_scan_id()
        rec = self.report_scans.get(sid) if sid else None
        if rec and rec.path:
            folder = os.path.dirname(rec.path)
            if os.path.isdir(folder):
                return folder
        return os.getcwd()

    def _choose_report_image_path(self, title: str) -> Optional[str]:
        """Open an image picker for report asset files."""
        path = filedialog.askopenfilename(
            title=title,
            initialdir=self._report_image_initial_dir(),
            filetypes=[
                ("PNG images", "*.png"),
                ("JPEG images", "*.jpg *.jpeg"),
                ("All images", "*.png *.jpg *.jpeg *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return None
        return os.path.abspath(path)

    def _set_report_image_path(self, kind: str, path: str):
        """Update stored report image path for lamp/axes assets."""
        normalized = os.path.abspath(path) if path else ""
        if kind == "lamp":
            self.report_lamp_image_path_var.set(normalized)
            label = "lamp photo"
        elif kind == "axes":
            self.report_axes_image_path_var.set(normalized)
            label = "axes photo"
        else:
            return
        if normalized:
            self._set_status(f"Set {label}: {os.path.basename(normalized)}")
        else:
            self._set_status(f"Cleared {label}.")

    def on_report_autoselect_segments_latest(self):
        """Auto-pick segment selections using latest-run heuristics."""
        self._refresh_report_segment_selectors(force_default=True, scan_id_hint=None)
        self._set_status("Auto-selected report segments from latest available runs.")

    def on_report_autoselect_segments_from_selected_scan(self):
        """Auto-pick segment selections preferring the currently selected scan."""
        sid = self._current_report_scan_id()
        if not sid:
            messagebox.showinfo("Segment Selection", "Select a scan first.")
            return
        self._refresh_report_segment_selectors(force_default=True, scan_id_hint=sid)
        self._set_status("Auto-selected report segments from selected scan where available.")

    def _on_report_segment_choice(self, seg_key: str):
        """Persist user selection for a report segment type."""
        var = self.report_segment_choice_vars.get(seg_key)
        if var is None:
            return
        disp = var.get().strip()
        seg_id = self._report_segment_display_to_id.get(seg_key, {}).get(disp)
        if seg_id:
            self.report_segment_selection[seg_key] = seg_id
        else:
            self.report_segment_selection.pop(seg_key, None)
        self._report_pattern_cache = {}

    def _phase_tag_for_record(self, rec: ReportScanRecord, idx: int, phase: Dict[str, object]) -> str:
        """Return explicit phase tag or inferred fallback."""
        tag = rec.phase_tags.get(idx)
        if isinstance(tag, str) and tag.strip() and tag != "Unassigned" and tag in REPORT_PHASE_TAGS:
            return tag
        return suggest_report_phase_tag(phase)

    def _format_report_segment_candidate(self, rec: ReportScanRecord, idx: int, phase: Dict[str, object]) -> str:
        """Build display text for a segment candidate."""
        phase_name = str(phase.get("name", f"Phase {idx + 1}"))
        start_ts = phase.get("start_ts")
        start_txt = human_datetime(start_ts) if isinstance(start_ts, (int, float)) else ""
        duration_h = phase.get("duration_h", 0.0)
        duration_txt = (
            f"{float(duration_h):.3f} h"
            if isinstance(duration_h, (int, float)) and float(duration_h) > 0
            else "n/a"
        )
        return f"{rec.label} :: P{idx + 1} {phase_name} :: {start_txt} :: {duration_txt}"

    def _collect_report_segment_candidates(self) -> Dict[str, List[Dict[str, object]]]:
        """Collect candidates for each canonical report segment type."""
        out: Dict[str, List[Dict[str, object]]] = {key: [] for key, _ in REPORT_SEGMENT_TYPES}
        scans_sorted = sorted(
            self.report_scans.items(),
            key=lambda item: (
                float(item[1].meta.get("first_ts")) if isinstance(item[1].meta.get("first_ts"), (int, float)) else -1.0,
                item[0],
            ),
        )
        for sid, rec in scans_sorted:
            for phase in rec.phases:
                idx = int(phase.get("index", 0))
                tag = self._phase_tag_for_record(rec, idx, phase)
                seg_key = report_segment_key_for_phase(phase, tag)
                if seg_key is None:
                    continue
                start_ts = phase.get("start_ts")
                duration_h = phase.get("duration_h", 0.0)
                out[seg_key].append(
                    {
                        "segment_id": f"{sid}:{idx}",
                        "display": self._format_report_segment_candidate(rec, idx, phase),
                        "scan_id": sid,
                        "phase_idx": idx,
                        "start_ts": float(start_ts) if isinstance(start_ts, (int, float)) else float("-inf"),
                        "duration_h": float(duration_h) if isinstance(duration_h, (int, float)) else 0.0,
                    }
                )
        for seg_key in out:
            out[seg_key].sort(key=lambda c: (c["start_ts"], c["phase_idx"]))
        return out

    def _pick_default_segment(
        self,
        seg_key: str,
        candidates: List[Dict[str, object]],
        scan_id_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Pick a default segment id for a segment type."""
        if not candidates:
            return None
        if scan_id_hint:
            hinted = [c for c in candidates if c.get("scan_id") == scan_id_hint]
            if hinted:
                if seg_key == "warmup":
                    return choose_latest_full_hour_warmup_segment_id(hinted)
                return max(hinted, key=lambda c: c.get("start_ts", -1.0)).get("segment_id")
        if seg_key == "warmup":
            return choose_latest_full_hour_warmup_segment_id(candidates)
        return max(candidates, key=lambda c: c.get("start_ts", -1.0)).get("segment_id")

    def _refresh_report_segment_selectors(
        self,
        force_default: bool = False,
        scan_id_hint: Optional[str] = None,
    ):
        """Refresh segment pickers and preserve explicit user choices when possible."""
        candidates_by_key = self._collect_report_segment_candidates()
        for seg_key, _seg_label in REPORT_SEGMENT_TYPES:
            combo = self._report_segment_combos.get(seg_key)
            if combo is None:
                continue
            candidates = candidates_by_key.get(seg_key, [])
            display_to_id: Dict[str, str] = {}
            values: List[str] = []
            for cand in candidates:
                disp = str(cand.get("display", ""))
                if disp in display_to_id:
                    disp = f"{disp} [{cand.get('segment_id')}]"
                display_to_id[disp] = str(cand.get("segment_id"))
                values.append(disp)
            self._report_segment_display_to_id[seg_key] = display_to_id
            combo["values"] = values

            candidate_ids = {str(c.get("segment_id")) for c in candidates}
            current_id = self.report_segment_selection.get(seg_key)
            target_id: Optional[str] = current_id if current_id in candidate_ids else None
            if force_default or target_id is None:
                target_id = self._pick_default_segment(seg_key, candidates, scan_id_hint=scan_id_hint)
            if target_id:
                self.report_segment_selection[seg_key] = target_id
            else:
                self.report_segment_selection.pop(seg_key, None)

            target_display = ""
            if target_id:
                for disp, seg_id in display_to_id.items():
                    if seg_id == target_id:
                        target_display = disp
                        break
            self.report_segment_choice_vars[seg_key].set(target_display)
        self._report_pattern_cache = {}

    def _summarize_scan_phases(self, sweep) -> List[Dict[str, object]]:
        """Return a list of phase summaries for a scan."""
        phases = []
        for idx, ph in enumerate(getattr(sweep, "phases", []) or []):
            rows = [r for r in getattr(ph, "members", []) if getattr(r, "timestamp", None) is not None]
            ts = [r.timestamp for r in rows]
            start = float(min(ts)) if ts else None
            end = float(max(ts)) if ts else None
            duration_h = ((end - start) / 3600.0) if (start is not None and end is not None and end > start) else 0.0
            phases.append({
                "index": idx,
                "name": getattr(ph, "name", "") or f"Phase {idx + 1}",
                "phase_type": getattr(getattr(ph, "phase_type", None), "name", str(getattr(ph, "phase_type", ""))),
                "axis": getattr(getattr(ph, "major_axis", None), "name", str(getattr(ph, "major_axis", ""))),
                "start_ts": start,
                "end_ts": end,
                "duration_h": float(duration_h),
                "count": len(getattr(ph, "members", []) or []),
            })
        return phases

    def _suggest_report_phase_tag(self, phase: Dict[str, object]) -> str:
        """Return a suggested tag for a phase based on name/type."""
        return suggest_report_phase_tag(phase)

    def _load_report_csv_meta(self, path: str) -> Dict[str, object]:
        """Load lightweight metadata for a Report Builder CSV file."""
        abs_path = os.path.abspath(path)
        meta: Dict[str, object] = {}
        try:
            meta["size_bytes"] = int(os.path.getsize(abs_path))
        except Exception:
            meta["size_bytes"] = None
        try:
            header = pd.read_csv(abs_path, nrows=0)
            cols = [str(c) for c in list(header.columns)]
            meta["columns"] = cols
            meta["column_count"] = len(cols)
        except Exception as e:
            meta["columns"] = []
            meta["column_count"] = 0
            meta["error"] = str(e)
        return meta

    def _suggest_report_csv_role(self, path: str, meta: Dict[str, object]) -> str:
        """Infer a report CSV role from filename/column hints."""
        name = os.path.basename(path).lower()
        cols = [str(c).lower() for c in meta.get("columns", [])]
        colset = set(cols)

        if "wfm" in name or "waveform" in name:
            return "Power Waveform"
        if "pwr_log" in name or "power_log" in name or "powerlog" in name:
            return "Power Log"

        pf_markers = ("pf", "power_factor", "thd", "vrms", "irms", "current", "voltage", "frequency")
        if any(any(marker in c for marker in pf_markers) for c in colset):
            return "Power Factor/Current"

        has_ts = any(c in colset for c in ("timestamp", "time", "epoch"))
        has_power = any(c in colset for c in ("w_active", "watts", "power_w", "w"))
        if has_ts and has_power:
            return "Power Log"

        return "Other"

    def _format_columns_summary(self, cols: List[str]) -> str:
        """Return a compact string summary for CSV columns."""
        if not cols:
            return ""
        short = ", ".join(cols[:3])
        extra = "" if len(cols) <= 3 else f", ... (+{len(cols) - 3})"
        return f"{len(cols)} cols: {short}{extra}"

    def _format_size(self, size_bytes: Optional[int]) -> str:
        """Pretty format a byte size."""
        if not size_bytes or size_bytes <= 0:
            return ""
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size_bytes} B"

    def _refresh_report_scans_tv(self):
        """Refresh the Report Builder scans treeview."""
        self.tv_report_scans.delete(*self.tv_report_scans.get_children())
        for sid, rec in self.report_scans.items():
            first = human_datetime(rec.meta.get("first_ts")) if rec.meta.get("first_ts") else ""
            last = human_datetime(rec.meta.get("last_ts")) if rec.meta.get("last_ts") else ""
            phase_count = rec.meta.get("phase_count", len(rec.phases))
            self.tv_report_scans.insert(
                "",
                "end",
                iid=sid,
                values=(rec.label, rec.role, phase_count, first, last, rec.path),
            )
        self._refresh_report_phase_scan_selector()
        active_sid = self._report_selected_scan_id if self._report_selected_scan_id in self.report_scans else None
        if active_sid is None and self.report_scans:
            active_sid = next(iter(self.report_scans.keys()))
        self._set_active_report_scan(active_sid, sync_tree=True, sync_combo=True)

    def _refresh_report_csvs_tv(self):
        """Refresh the Report Builder CSV treeview."""
        self.tv_report_csvs.delete(*self.tv_report_csvs.get_children())
        for cid, rec in self.report_csvs.items():
            cols = rec.meta.get("columns", []) if isinstance(rec.meta, dict) else []
            col_text = self._format_columns_summary(cols)
            size_text = self._format_size(rec.meta.get("size_bytes") if isinstance(rec.meta, dict) else None)
            self.tv_report_csvs.insert(
                "",
                "end",
                iid=cid,
                values=(rec.label, rec.role, col_text, size_text, rec.path),
            )

    def _refresh_report_phases_tv(self, rec: Optional[ReportScanRecord]):
        """Refresh the phases list for the selected report scan."""
        self.tv_report_phases.delete(*self.tv_report_phases.get_children())
        if rec is None:
            return
        for ph in rec.phases:
            idx = int(ph.get("index", 0))
            name = ph.get("name", f"Phase {idx + 1}")
            ptype = ph.get("phase_type", "")
            axis = ph.get("axis", "")
            start = human_datetime(ph.get("start_ts")) if ph.get("start_ts") else ""
            end = human_datetime(ph.get("end_ts")) if ph.get("end_ts") else ""
            duration_h = ph.get("duration_h", 0.0)
            duration_txt = f"{float(duration_h):.3f}" if isinstance(duration_h, (int, float)) else ""
            count = ph.get("count", "")
            raw_tag = rec.phase_tags.get(idx)
            tag = raw_tag if raw_tag in REPORT_PHASE_TAGS else suggest_report_phase_tag(ph)
            self.tv_report_phases.insert(
                "",
                "end",
                iid=str(idx),
                values=(idx + 1, name, ptype, axis, start, end, duration_txt, count, tag),
            )

    def _refresh_report_phase_scan_selector(self):
        """Refresh scan picker shown on the Phases tab."""
        if not hasattr(self, "cb_report_phase_scan"):
            return
        label_counts: Dict[str, int] = {}
        for rec in self.report_scans.values():
            label_counts[rec.label] = label_counts.get(rec.label, 0) + 1
        self._report_phase_scan_display_to_id.clear()
        self._report_phase_scan_id_to_display.clear()
        values: List[str] = []
        for sid, rec in self.report_scans.items():
            disp = rec.label if label_counts.get(rec.label, 0) == 1 else f"{rec.label} [{sid}]"
            self._report_phase_scan_display_to_id[disp] = sid
            self._report_phase_scan_id_to_display[sid] = disp
            values.append(disp)
        self.cb_report_phase_scan["values"] = values

    def _set_active_report_scan(self, sid: Optional[str], sync_tree: bool = True, sync_combo: bool = True):
        """Set active report scan and sync both Inputs and Phases controls."""
        sid = sid if (sid and sid in self.report_scans) else None
        self._report_selected_scan_id = sid

        if sid is None:
            self.report_scan_info_var.set("No scan selected.")
            self.report_scan_role_var.set("Unassigned")
            self._refresh_report_phases_tv(None)
            if sync_tree and hasattr(self, "tv_report_scans"):
                cur = list(self.tv_report_scans.selection())
                if cur:
                    self.tv_report_scans.selection_remove(*cur)
            if sync_combo:
                self.report_phase_scan_display_var.set("")
            return

        rec = self.report_scans.get(sid)
        if rec is None:
            self._set_active_report_scan(None, sync_tree=sync_tree, sync_combo=sync_combo)
            return

        self.report_scan_info_var.set(f"{rec.label} ({sid})")
        self.report_scan_role_var.set(rec.role or "Unassigned")
        self._refresh_report_phases_tv(rec)

        if sync_tree and hasattr(self, "tv_report_scans"):
            cur = self.tv_report_scans.selection()
            if tuple(cur) != (sid,):
                self.tv_report_scans.selection_set(sid)
                self.tv_report_scans.focus(sid)
        if sync_combo:
            disp = self._report_phase_scan_id_to_display.get(sid, "")
            self.report_phase_scan_display_var.set(disp)

    def _on_report_scan_selected(self):
        """Update phase tagging panel when a report scan is selected in Inputs tab."""
        sel = self.tv_report_scans.selection()
        if not sel:
            return
        sid = sel[0]
        self._set_active_report_scan(sid, sync_tree=False, sync_combo=True)

    def _on_report_phase_scan_combo_selected(self):
        """Update active scan when selection changes in Phases tab scan picker."""
        disp = self.report_phase_scan_display_var.get().strip()
        sid = self._report_phase_scan_display_to_id.get(disp)
        if sid:
            self._set_active_report_scan(sid, sync_tree=True, sync_combo=False)

    def _on_report_csv_selected(self):
        """Update CSV role selection when a report CSV is selected."""
        sel = self.tv_report_csvs.selection()
        if not sel:
            return
        cid = sel[0]
        rec = self.report_csvs.get(cid)
        if rec is None:
            return
        self.report_csv_role_var.set(rec.role or "Other")

    def _current_report_scan_id(self) -> Optional[str]:
        """Return the selected report scan id, if any."""
        sel = self.tv_report_scans.selection()
        if sel:
            return sel[0]
        return self._report_selected_scan_id

    def _current_report_csv_id(self) -> Optional[str]:
        """Return the selected report CSV id, if any."""
        sel = self.tv_report_csvs.selection()
        if sel:
            return sel[0]
        return None

    # ---- View helpers ----
    def _set_initial_sashes(self):
        """Favor Controls, but guarantee Files has enough height to show its button bar."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height() or self.winfo_height() or 800
            MIN_FILES = 180
            MIN_GROUPS = 140
            MIN_CONTROLS = 260
            # initial proportional split
            pos0 = int(total_h * 0.18)
            pos1 = int(total_h * 0.42)
            # enforce minimum heights
            pos0 = max(MIN_FILES, pos0)
            pos1 = max(pos0 + MIN_GROUPS, pos1)
            if (total_h - pos1) < MIN_CONTROLS:
                pos1 = max(pos0 + MIN_GROUPS, total_h - MIN_CONTROLS)
            pos1 = min(pos1, total_h - 40)
            self._paned.sashpos(0, pos0)
            self._paned.sashpos(1, pos1)
        except Exception:
            pass

    def on_toggle_controls(self):
        """Show or hide the Controls pane based on the checkbox."""
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
        """Shrink the controls section to a compact but usable height (~280 px)."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            # Keep controls usable while still compact.
            self._paned.sashpos(0, 150)
            self._paned.sashpos(1, max(300, total_h - 280))
        except Exception:
            pass

    def on_favor_controls(self):
        """Give more height to the Controls section (Files ~25%, Groups ~30%, Controls ~45%)."""
        try:
            self._set_initial_sashes()
        except Exception:
            pass

    def on_maximize_files(self):
        """Resize the panes to maximize the Files section."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, total_h - 40)
            self._paned.sashpos(1, total_h - 20)
        except Exception:
            pass

    def on_maximize_groups(self):
        """Resize the panes to maximize the Groups section."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, 120)
            self._paned.sashpos(1, total_h - 40)
        except Exception:
            pass

    def on_maximize_controls(self):
        """Resize the panes to maximize the Controls section."""
        try:
            self.update_idletasks()
            total_h = self._paned.winfo_height()
            self._paned.sashpos(0, 120)
            self._paned.sashpos(1, total_h - 10)
        except Exception:
            pass

    # ---- File ops ----
    def on_add_sw3_files(self):
        """Prompt for SW3/eegbin files and add them to the project."""
        paths = filedialog.askopenfilenames(title="Add SW3/eegbin Files",
                                            filetypes=[("SW3/eegbin", "*.sw3 *.eegbin *.bin *.*")])
        if not paths: return
        ensure_eegbin_imported()
        n=0
        for path in paths:
            sweep = self._load_sweep_from_path(path, warn=True, trace=True)
            if sweep is None:
                continue
            meta = self._meta_from_sweep(sweep)
            fid = self._next_file_id()
            self.files[fid] = FileRecord(fid, "sw3", path, os.path.basename(path), meta)
            n += 1
        if n:
            self._refresh_files_tv()
            self._refresh_display_mappings()
            self._invalidate_aligned_cache()
            self._set_status(f"Added {n} SW3 file(s).")

    def on_add_power_files(self):
        """Prompt for power CSV files and add them to the project."""
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
            self._invalidate_aligned_cache()
            self._set_status(f"Added {n} power file(s).")

    def _meta_from_sweep(self, sweep) -> Dict[str, object]:
        """Extract summary metadata (timestamps/count) from a sweep."""
        rows = [r for r in sweep.rows if getattr(r, "timestamp", None) is not None]
        ts = [r.timestamp for r in rows]
        first = float(min(ts)) if ts else None
        last = float(max(ts)) if ts else None
        return {"first_ts": first, "last_ts": last, "count": len(rows), "lamp_name": getattr(sweep, "lamp_name", "")}

    def _power_map_signature(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Return a stable signature for current power column candidate settings."""
        ts = tuple(str(x).strip().lower() for x in self.power_column_map.get("timestamp", []))
        watts = tuple(str(x).strip().lower() for x in self.power_column_map.get("watts", []))
        return ts, watts

    def _power_file_signature(self, path: str) -> Tuple[int, int]:
        """Return (mtime_ns, size_bytes) used to invalidate power CSV cache entries."""
        st = os.stat(path)
        return int(st.st_mtime_ns), int(st.st_size)

    def _load_power_csv_with_meta(self, path: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Load a power CSV file and return (dataframe, metadata)."""
        abs_path = os.path.abspath(path)
        map_sig = self._power_map_signature()
        file_sig = self._power_file_signature(abs_path)
        cache_key = (abs_path, file_sig, map_sig)
        cached = self._power_csv_cache.get(cache_key)
        if cached is not None:
            return cached[0], dict(cached[1])

        header = pd.read_csv(abs_path, nrows=0)
        ts_col = self._detect_col(header.columns, self.power_column_map["timestamp"])
        w_col  = self._detect_col(header.columns, self.power_column_map["watts"])
        if not ts_col or not w_col:
            raise ValueError(
                f"CSV must include timestamp + watts. Candidates tried: {self.power_column_map}. "
                f"Found: {list(header.columns)}"
            )

        df = pd.read_csv(
            abs_path,
            usecols=[ts_col, w_col],
            low_memory=False,
        ).rename(columns={ts_col: "Timestamp", w_col: "W_Active"})
        df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
        df["W_Active"]  = pd.to_numeric(df["W_Active"], errors="coerce")
        df = df.dropna(subset=["Timestamp","W_Active"])
        if not df.empty and np.any(np.diff(df["Timestamp"].to_numpy(dtype=float)) < 0):
            df = df.sort_values("Timestamp", kind="mergesort")
        df = df.reset_index(drop=True)
        meta = {"first_ts": float(df["Timestamp"].min()) if not df.empty else None,
                "last_ts":  float(df["Timestamp"].max()) if not df.empty else None,
                "count": int(len(df))}
        result = (df, meta)
        # Simple cap keeps memory bounded on repeated analysis runs.
        if len(self._power_csv_cache) >= 24:
            self._power_csv_cache.clear()
        self._power_csv_cache[cache_key] = result
        return result[0], dict(result[1])

    def _detect_col(self, cols, candidates: List[str]) -> Optional[str]:
        """Pick the first column matching a list of candidate names."""
        m = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in m:
                return m[cand.lower()]
        return None

    def _refresh_files_tv(self):
        """Refresh the Files treeview with current metadata."""
        self.tv_files.delete(*self.tv_files.get_children())
        for fid, rec in self.files.items():
            first = human_datetime(rec.meta.get("first_ts")) if rec.meta.get("first_ts") else ""
            last  = human_datetime(rec.meta.get("last_ts")) if rec.meta.get("last_ts") else ""
            count = rec.meta.get("count","")
            self.tv_files.insert("", "end", iid=fid, values=(rec.label, rec.kind, first, last, count, rec.path))

    def _refresh_groups_tv(self):
        """Refresh the Groups treeview and selection list."""
        self.tv_groups.delete(*self.tv_groups.get_children())
        for gid, g in self.groups.items():
            sw3c = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "sw3")
            powc = sum(1 for fid in g.file_ids if self.files.get(fid, FileRecord("", "", "", "")).kind == "power")
            self.tv_groups.insert("", "end", iid=gid, values=(g.name, self._format_group_trim(g), sw3c, powc))
        self._refresh_group_select_list()
        self._refresh_display_mappings()

    def _refresh_group_select_list(self):
        """Refresh the group listbox used by decay overlay selection."""
        self.lb_groups_select.delete(0, tk.END)
        for gid, g in self.groups.items():
            self.lb_groups_select.insert(tk.END, f"{gid}  {g.name}")

    def _refresh_display_mappings(self):
        """Refresh friendly display labels for files and groups."""
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
        """Resolve a displayed SW3 label back to its file id."""
        return self._sw3_display_to_id.get(disp)

    def _display_to_group_id(self, disp: str) -> Optional[str]:
        """Resolve a displayed Group label back to its group id."""
        return self._group_display_to_id.get(disp)

    def _load_sweep_from_path(self, path: str, warn: bool = True, trace: bool = False):
        """Load an SW3/eegbin file and return the sweep object or None on failure."""
        try:
            with open(path, "rb") as fd:
                buf = fd.read()
            ensure_eegbin_imported()
            try:
                return eegbin.load_eegbin3(buf, from_path=path)
            except Exception:
                return eegbin.load_eegbin2(buf, from_path=path)
        except Exception as e:
            if trace:
                traceback.print_exc()
            if warn:
                messagebox.showwarning("Load error", f"Failed to load {path}\n{e}")
            return None

    def _row_intensity_uW_cm2(self, row, normalize_to_1m: bool) -> float:
        """Compute intensity in µW/cm² from a row, optionally normalized to 1 m."""
        I_wm2 = float(getattr(row.capture, "integral_result", 0.0) or 0.0)
        if normalize_to_1m:
            dist_m = float(getattr(row.coords, "lin_mm", 0.0) or 0.0) / 1000.0
            if dist_m > 0:
                I_wm2 = I_wm2 * (dist_m ** 2)
        return I_wm2 * WM2_TO_UW_CM2

    def _extract_intensity_rows(
        self,
        sweep,
        only_zero: bool,
        normalize_to_1m: bool,
        include_row: bool = False,
    ):
        """Return sorted (timestamp, intensity[, row, sweep]) rows from a sweep."""
        out = []
        for r in getattr(sweep, "rows", []):
            if hasattr(r, "valid") and not r.valid:
                continue
            if only_zero and not (
                getattr(r.coords, "yaw_deg", None) == 0 and getattr(r.coords, "roll_deg", None) == 0
            ):
                continue
            ts = getattr(r, "timestamp", None)
            if ts is None:
                continue
            intensity = self._row_intensity_uW_cm2(r, normalize_to_1m)
            if include_row:
                out.append((float(ts), intensity, r, sweep))
            else:
                out.append((float(ts), intensity))
        out.sort(key=lambda x: x[0])
        return out

    # ---- File actions ----
    def on_rename_file(self):
        """Rename the selected file's display label."""
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
            self._invalidate_aligned_cache()

    def on_remove_files(self):
        """Remove selected files from the project and all group mappings."""
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
        self._invalidate_aligned_cache()

    def on_assign_files_to_group(self):
        """Assign selected files to a chosen group."""
        sel = list(self.tv_files.selection())
        if not sel:
            messagebox.showinfo("Assign", "Select files (SW3 and/or Power) to assign to a group."); return
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        for fid in sel: g.file_ids.add(fid)
        self._refresh_groups_tv()
        self._invalidate_aligned_cache()
        self._set_status(f"Assigned {len(sel)} file(s) to group '{g.name}'.")

    # ---- Groups ----
    def on_new_group(self):
        """Create a new group."""
        name = simpledialog.askstring("New Group", "Group name:", parent=self)
        if not name: return
        gid = self._next_group_id()
        self.groups[gid] = GroupRecord(gid, name.strip())
        self._refresh_groups_tv()
        self._set_status(f"Created group '{name}'.")

    def on_rename_group(self):
        """Rename the selected group."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        new = simpledialog.askstring("Rename Group", "New name:", initialvalue=g.name, parent=self)
        if new:
            g.name = new.strip()
            self._refresh_groups_tv()
            self._invalidate_aligned_cache()

    def on_delete_group(self):
        """Delete the selected group (does not delete files)."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        if not messagebox.askyesno("Delete Group", "Delete this group (files remain in project)?"):
            return
        del self.groups[gid]
        self._refresh_groups_tv()
        self._invalidate_aligned_cache()

    def on_edit_associations(self):
        """Open the associations dialog for the selected group."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        sw3s = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind=="sw3"]
        pows = [(fid, self.files[fid]) for fid in g.file_ids if self.files[fid].kind=="power"]
        if not sw3s or not pows:
            messagebox.showinfo("Associations", "Assign at least one SW3 and one Power file to the group first."); return
        dlg = AssociationDialog(self, g, sw3s, pows); self.wait_window(dlg)
        if dlg.updated:
            self._invalidate_aligned_cache()
            self._set_status("Updated associations.")

    def _format_group_trim(self, g) -> str:
        """Return a user-friendly trim string for a group."""
        s = fmt_hhmmss(max(0, int(getattr(g, "trim_start_s", 0))))
        e = fmt_hhmmss(max(0, int(getattr(g, "trim_end_s", 0))))
        return "—" if (s=="00:00:00" and e=="00:00:00") else f"{s} | {e}"

    def _group_time_window(
        self,
        g: GroupRecord,
        files_map: Optional[Dict[str, FileRecord]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Compute trimmed time window [tmin,tmax] from group's SW3 files and trims.
           Returns None if window degenerates or no SW3 meta is available.
        """
        file_lookup = self.files if files_map is None else files_map
        sw3_times = []
        for fid in g.file_ids:
            fr = file_lookup.get(fid)
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
        """Prompt for and apply group-level trim values."""
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
            self._invalidate_aligned_cache()
        ttk.Button(btns, text="OK", command=save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=4)

    def on_clear_group_trim(self):
        """Clear trim values for the selected group."""
        gid = self._ensure_group_selected_or_prompt()
        if not gid: return
        g = self.groups[gid]
        g.trim_start_s = 0; g.trim_end_s = 0
        self._refresh_groups_tv()
        self._invalidate_aligned_cache()

    def _ensure_group_selected_or_prompt(self) -> Optional[str]:
        """Return a selected group id or prompt the user to choose one."""
        sel = self.tv_groups.selection()
        if sel: return sel[0]
        if not self.groups:
            messagebox.showinfo("Groups", "Create a group first."); return None
        choices = list(self.groups.keys()); labels = [self.groups[c].name for c in choices]
        idx = simpledialog.askinteger("Select Group", "Enter group number:\n"+ "\n".join(f"{i+1}. {labels[i]}" for i in range(len(labels))),
                                      minvalue=1, maxvalue=len(labels), parent=self)
        return choices[idx-1] if idx else None

    def _on_group_selection_changed(self):
        """Handle group selection changes (currently no-op)."""
        pass  # selection is used via friendly comboboxes

    # ---- Settings ----
    def on_set_power_columns(self):
        """Edit the candidate column names used for power CSV detection."""
        top = tk.Toplevel(self); top.title("Power Columns Map"); top.resizable(True, False)
        ttk.Label(top, text="Candidate names for timestamp column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        e1 = ttk.Entry(top, width=60); e1.insert(0, ", ".join(self.power_column_map["timestamp"])); e1.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(top, text="Candidate names for watts column (comma‑separated):").pack(anchor="w", padx=6, pady=(6,0))
        e2 = ttk.Entry(top, width=60); e2.insert(0, ", ".join(self.power_column_map["watts"])); e2.pack(fill=tk.X, padx=6, pady=3)
        def save():
            self.power_column_map["timestamp"] = [s.strip() for s in e1.get().split(",") if s.strip()]
            self.power_column_map["watts"] = [s.strip() for s in e2.get().split(",") if s.strip()]
            self._power_csv_cache.clear()
            self._invalidate_aligned_cache()
            top.destroy()
        b = ttk.Frame(top); b.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(b, text="OK", command=save).pack(side=tk.RIGHT, padx=3)
        ttk.Button(b, text="Cancel", command=top.destroy).pack(side=tk.RIGHT, padx=3)

    def on_add_module_path(self):
        """Add a folder to the module search path for eegbin/util."""
        folder = filedialog.askdirectory(title="Add folder to Python module search path (eegbin/util)")
        if not folder: return
        _add_module_path(folder); self._set_status(f"Added module path: {folder}")

    def on_list_module_paths(self):
        """Show the currently registered extra module paths."""
        if not _EXTRA_MODULE_PATHS:
            messagebox.showinfo("Module Paths", "No extra module paths added."); return
        msg = "Extra module search paths:\n\n" + "\n".join("• "+p for p in _EXTRA_MODULE_PATHS)
        messagebox.showinfo("Module Paths", msg)

    # ---- Reload ----
    def _start_background_reload(self, targets: List[str]) -> bool:
        """Start asynchronous metadata reload for selected/all files."""
        if self._reload_thread is not None and self._reload_thread.is_alive():
            return False

        reload_items: List[Tuple[str, str, str, str]] = []
        has_sw3 = False
        for fid in targets:
            rec = self.files.get(fid)
            if not rec:
                continue
            reload_items.append((fid, rec.kind, rec.path, rec.label))
            has_sw3 = has_sw3 or (rec.kind == "sw3")
        if not reload_items:
            return False

        if has_sw3:
            # Ensure eegbin imports happen on the UI thread to avoid
            # background-thread dialogs if the module path is missing.
            try:
                ensure_eegbin_imported()
            except Exception:
                return False

        self._reload_job_id += 1
        job_id = self._reload_job_id
        total = len(reload_items)
        self._set_status(f"Reload started ({total} file(s))...")
        self._set_busy(True, "Reloading file metadata...")

        def _worker():
            updates: Dict[str, Dict[str, object]] = {}
            errors: List[Tuple[str, str]] = []
            ok = 0
            try:
                for idx, (fid, kind, path, label) in enumerate(reload_items, start=1):
                    shown = os.path.basename(path) or label or fid
                    self._reload_queue.put(("progress", job_id, idx, total, shown))
                    try:
                        if kind == "sw3":
                            sweep = self._load_sweep_from_path(path, warn=False, trace=False)
                            if sweep is None:
                                raise RuntimeError("Failed to load SW3 file.")
                            meta = self._meta_from_sweep(sweep)
                        else:
                            _, meta = self._load_power_csv_with_meta(path)
                        updates[fid] = meta
                        ok += 1
                    except Exception as e:
                        errors.append((label or fid, str(e)))
            except Exception:
                errors.append(("reload worker", traceback.format_exc()))
            finally:
                err = max(0, total - ok)
                self._reload_queue.put(("done", job_id, updates, ok, err, errors))

        self._reload_thread = threading.Thread(
            target=_worker,
            name=f"reload-meta-{job_id}",
            daemon=True,
        )
        self._reload_thread.start()
        self.after(120, self._poll_background_reload)
        return True

    def _poll_background_reload(self):
        """Consume background reload progress and apply final metadata updates."""
        saw_done = False
        while True:
            try:
                event = self._reload_queue.get_nowait()
            except Empty:
                break
            etype = event[0]
            if etype == "progress":
                _, job_id, idx, total, shown = event
                if job_id != self._reload_job_id:
                    continue
                msg = f"Reloading {idx}/{total}: {shown}"
                self._set_status(msg)
                self._set_busy(True, msg)
                continue
            if etype == "done":
                _, job_id, updates, ok, err, errors = event
                if job_id != self._reload_job_id:
                    continue
                saw_done = True
                for fid, meta in updates.items():
                    rec = self.files.get(fid)
                    if rec is not None:
                        rec.meta = meta
                self._refresh_files_tv()
                self._refresh_display_mappings()
                self._set_status(f"Reloaded {ok}/{ok + err} file(s){'; errors: ' + str(err) if err else ''}.")
                if err:
                    details = []
                    for lab, text in errors[:5]:
                        details.append(f"• {lab}: {text}")
                    more = "" if len(errors) <= 5 else f"\n• ... and {len(errors) - 5} more"
                    messagebox.showwarning(
                        "Reload",
                        "Reload completed with errors:\n\n" + "\n".join(details) + more,
                    )
                if ok:
                    self._invalidate_aligned_cache()
                    self._trigger_reprocess_after_reload()

        if self._reload_thread is not None and self._reload_thread.is_alive():
            self.after(120, self._poll_background_reload)
            return
        if self._reload_thread is not None and not saw_done:
            # Catch any final queued completion event after thread exit.
            self.after(60, self._poll_background_reload)
            return
        self._reload_thread = None
        self._set_busy(self._is_any_background_busy())

    def on_reload_all_files(self, only_selected: bool=False):
        """Reload file metadata from disk (all or selected) in background."""
        if self._reload_thread is not None and self._reload_thread.is_alive():
            messagebox.showinfo("Reload", "Reload is already running in the background.")
            return
        targets = list(self.tv_files.selection()) if only_selected else list(self.files.keys())
        if only_selected and not targets:
            messagebox.showinfo("Reload", "No files selected.")
            return
        if not targets:
            self._set_status("No files available to reload.")
            return
        started = self._start_background_reload(targets)
        if not started:
            messagebox.showinfo("Reload", "Could not start reload.")

    # ---- Session I/O ----
    def on_save_session(self):
        """Save the full session (files, groups, controls) to JSON."""
        path = filedialog.asksaveasfilename(title="Save Session", defaultextension=".json",
                                            filetypes=[("Session JSON","*.json")])
        if not path: return
        data = {
            "app_version": APP_VERSION,
            "session_schema_version": SESSION_SCHEMA_VERSION,
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
                "time_weighted_ema": self.time_weighted_ema.get(),
                "auto_power_ema": self.auto_power_ema.get(),
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
            },
            "report": {
                "scans": [asdict(sr) for sr in self.report_scans.values()],
                "csvs": [asdict(cr) for cr in self.report_csvs.values()],
                "segment_selection": dict(self.report_segment_selection),
                "lamp_image_path": self.report_lamp_image_path_var.get(),
                "axes_image_path": self.report_axes_image_path_var.get(),
                "exposure_mode": self.report_exposure_mode_var.get(),
                "optical_power_mode": self.report_optical_power_mode_var.get(),
                "optical_power_min_nm": self.report_optical_power_min_nm_var.get(),
                "optical_power_max_nm": self.report_optical_power_max_nm_var.get(),
                "pattern_plane_idx": int(self.report_pattern_plane_idx_var.get()),
                "metadata": self._collect_report_metadata(),
                "comments": self._collect_report_comments(),
            }
        }
        with open(path, "w") as fd: json.dump(data, fd, indent=2)
        self._set_status(f"Saved session to {path}")

    def on_load_session(self):
        """Load a session JSON file and restore UI state."""
        path = filedialog.askopenfilename(title="Load Session",
                                          filetypes=[("Session JSON","*.json"),("All files","*.*")])
        if not path: return
        with open(path, "r") as fd: data = json.load(fd)
        _EXTRA_MODULE_PATHS.clear()
        for p in data.get("module_paths", []): _add_module_path(p)
        self.power_column_map = data.get("power_column_map", self.power_column_map)
        self._power_csv_cache.clear()
        self.files = {fr["file_id"]: FileRecord(**fr) for fr in data.get("files", [])}
        self.groups = {}
        for g in data.get("groups", []):
            gr = GroupRecord(group_id=g["group_id"], name=g["name"],
                             trim_start_s=g.get("trim_start_s", 0), trim_end_s=g.get("trim_end_s", 0))
            gr.file_ids = set(g.get("file_ids", []))
            gr.associations = {k: set(v) for k,v in g.get("associations", {}).items()}
            self.groups[gr.group_id] = gr
        report = data.get("report", {})
        self.report_scans = {}
        for sr in report.get("scans", []):
            rec = ReportScanRecord(**sr)
            if isinstance(rec.phase_tags, dict):
                rec.phase_tags = {int(k): v for k, v in rec.phase_tags.items()}
            self.report_scans[rec.scan_id] = rec
        self.report_csvs = {}
        for cr in report.get("csvs", []):
            rec = ReportCsvRecord(**cr)
            self.report_csvs[rec.csv_id] = rec
        allowed_segment_keys = {k for k, _ in REPORT_SEGMENT_TYPES}
        self.report_segment_selection = {
            str(k): str(v)
            for k, v in report.get("segment_selection", {}).items()
            if str(k) in allowed_segment_keys
        }
        self.report_lamp_image_path_var.set(str(report.get("lamp_image_path", "") or ""))
        self.report_axes_image_path_var.set(str(report.get("axes_image_path", "") or ""))
        saved_exposure_mode = str(report.get("exposure_mode", "") or "").strip()
        self.report_exposure_mode_var.set(saved_exposure_mode if saved_exposure_mode in REPORT_EXPOSURE_MODES else "IEC")
        saved_power_mode = str(report.get("optical_power_mode", "") or "").strip()
        self.report_optical_power_mode_var.set(
            saved_power_mode if saved_power_mode in REPORT_OPTICAL_POWER_MODES else "Band"
        )
        self.report_optical_power_min_nm_var.set(str(report.get("optical_power_min_nm", "200") or "200"))
        self.report_optical_power_max_nm_var.set(str(report.get("optical_power_max_nm", "230") or "230"))
        self.report_pattern_plane_idx_var.set(int(report.get("pattern_plane_idx", 0) or 0))
        self._sync_report_optical_power_controls()
        self._sync_report_pattern_plane_controls([], int(self.report_pattern_plane_idx_var.get()))
        self._apply_report_metadata(report.get("metadata", {}))
        self._apply_report_comments(report.get("comments", {}))
        self._report_sweep_cache.clear()
        self._report_spectral_cache = {}
        self._report_pattern_cache = {}
        self.report_preview_status_var.set("Preview not generated.")
        self.report_electrical_summary_var.set("No electrical summary yet.")
        self._report_selected_scan_id = None
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
            ("time_weighted_ema", self.time_weighted_ema),
            ("auto_power_ema", self.auto_power_ema),
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
        self._refresh_report_scans_tv()
        self._refresh_report_csvs_tv()
        self._refresh_report_segment_selectors()
        self.on_toggle_controls()
        self._invalidate_aligned_cache()
        self._last_analyzed_gid = self._current_ovp_group_id()
        self._autostart_processing_after_session_load()

    def _invalidate_aligned_cache(self):
        """Drop cached aligned data after source files/settings changes."""
        self._aligned_cache = None
        self._aligned_cache_gid = None
        self._aligned_cache_signature = None
        self._ovp_series_cache = None
        self._sw3_plot_cache.clear()

    def _snapshot_analysis_params(self) -> Dict[str, object]:
        """Capture analysis settings so background workers avoid Tk variable access."""
        return {
            "intensity_ema_span": int(self.intensity_ema_span.get()),
            "power_ema_span": int(self.power_ema_span.get()),
            "align_tolerance_s": int(self.align_tolerance_s.get()),
            "time_weighted_ema": bool(self.time_weighted_ema.get()),
            "auto_power_ema": bool(self.auto_power_ema.get()),
            "only_yaw_roll_zero": bool(self.only_yaw_roll_zero.get()),
            "normalize_to_1m": bool(self.normalize_to_1m.get()),
            "resample_seconds": int(self.resample_seconds.get()),
        }

    def _analysis_params_signature(self, params: Dict[str, object]) -> Tuple:
        """Create a comparable immutable signature for analysis settings."""
        return (
            int(params["intensity_ema_span"]),
            int(params["power_ema_span"]),
            int(params["align_tolerance_s"]),
            bool(params["time_weighted_ema"]),
            bool(params["auto_power_ema"]),
            bool(params["only_yaw_roll_zero"]),
            bool(params["normalize_to_1m"]),
            int(params["resample_seconds"]),
        )

    def _current_ovp_group_id(self) -> Optional[str]:
        """Resolve the active OVP group id from combobox or current tree selection."""
        gdisp = self.ovp_group_display.get()
        gid = self._display_to_group_id(gdisp)
        if gid and gid in self.groups:
            return gid
        sel = self.tv_groups.selection()
        if sel and sel[0] in self.groups:
            return sel[0]
        return None

    def _start_background_alignment(self, gid: str) -> bool:
        """Start asynchronous aligned-data build for a group. Returns True if started."""
        if not gid or gid not in self.groups:
            return False
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            self._pending_reprocess_gid = gid
            if self._analysis_running_gid == gid:
                self._set_status(f"Background processing already running for '{self.groups[gid].name}'.")
            else:
                running_name = "unknown"
                if self._analysis_running_gid and self._analysis_running_gid in self.groups:
                    running_name = self.groups[self._analysis_running_gid].name
                self._set_status(
                    f"Background processing busy on '{running_name}'. Queued '{self.groups[gid].name}' next."
                )
            return False

        params = self._snapshot_analysis_params()
        params_sig = self._analysis_params_signature(params)
        self._analysis_job_id += 1
        job_id = self._analysis_job_id
        group_name = self.groups[gid].name
        self._analysis_running_gid = gid
        self._analysis_running_signature = params_sig
        self._set_status(f"Background processing started for '{group_name}'...")

        def _worker():
            try:
                aligned, plot_cache = self._build_aligned_for_group(
                    gid,
                    analysis_params=params,
                    ui_progress=False,
                    return_plot_cache=True,
                )
                self._analysis_queue.put((job_id, gid, aligned, None, params_sig, plot_cache))
            except Exception:
                self._analysis_queue.put((job_id, gid, None, traceback.format_exc(), params_sig, None))

        self._analysis_thread = threading.Thread(
            target=_worker,
            name=f"aligned-build-{gid}-{job_id}",
            daemon=True,
        )
        self._analysis_thread.start()
        self._set_busy(True, f"Preparing OVP cache ({group_name})...")
        self.after(120, self._poll_background_alignment)
        return True

    def _poll_background_alignment(self):
        """Harvest completed background alignment jobs and update cache/status."""
        saw_result = False
        while True:
            try:
                job_id, gid, aligned, err, params_sig, plot_cache = self._analysis_queue.get_nowait()
            except Empty:
                break
            saw_result = True
            if job_id != self._analysis_job_id:
                continue
            self._analysis_running_gid = None
            self._analysis_running_signature = None
            if err is not None:
                self._invalidate_aligned_cache()
                print(err, file=sys.stderr)
                self._set_status("Background processing failed. Check terminal for traceback.")
                messagebox.showwarning("Analyze", "Background processing failed. See terminal for details.")
                continue
            group_name = self.groups[gid].name if gid in self.groups else gid
            if plot_cache is not None:
                self._ovp_series_cache = plot_cache
            if aligned is None or aligned.empty:
                self._aligned_cache = None
                self._aligned_cache_gid = None
                self._aligned_cache_signature = None
                if plot_cache and (
                    plot_cache.get("sw3_series_by_id") or plot_cache.get("power_series_by_id")
                ):
                    self._set_status(
                        f"Background processing finished for '{group_name}' (partial streams found, but no aligned overlap)."
                    )
                else:
                    self._set_status(f"Background processing finished for '{group_name}' with no aligned data.")
                continue
            self._aligned_cache = aligned
            self._aligned_cache_gid = gid
            self._aligned_cache_signature = params_sig
            self._last_analyzed_gid = gid
            self._set_status(f"Aligned data ready for '{group_name}'. Click Analyze Group to plot.")

        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            running_name = ""
            if self._analysis_running_gid and self._analysis_running_gid in self.groups:
                running_name = self.groups[self._analysis_running_gid].name
            if running_name:
                self._set_busy(True, f"Preparing OVP cache ({running_name})...")
            else:
                self._set_busy(True, "Preparing OVP cache...")
            self.after(120, self._poll_background_alignment)
            return

        if self._pending_reprocess_gid:
            gid = self._pending_reprocess_gid
            self._pending_reprocess_gid = None
            self._start_background_alignment(gid)
            return

        if saw_result and self._analysis_running_gid is None and not self._pending_reprocess_gid:
            self._analysis_thread = None
        if self._analysis_thread is not None and not self._analysis_thread.is_alive() and not self._pending_reprocess_gid:
            self._analysis_thread = None
            self._analysis_running_gid = None
            self._analysis_running_signature = None
        self._set_busy(self._is_any_background_busy())

    def _ensure_group_aligned_ready(self, gid: str, action_label: str) -> Optional[pd.DataFrame]:
        """Return aligned cache for gid, or start background processing and warn."""
        if not gid or gid not in self.groups:
            messagebox.showinfo(action_label, "Select a Group.")
            return None
        current_sig = self._analysis_params_signature(self._snapshot_analysis_params())
        if (
            self._aligned_cache_gid == gid
            and self._aligned_cache is not None
            and not self._aligned_cache.empty
            and self._aligned_cache_signature == current_sig
        ):
            return self._aligned_cache
        if self._aligned_cache_gid == gid and self._aligned_cache_signature != current_sig:
            self._invalidate_aligned_cache()
        ovp_cache = self._ovp_series_cache or {}
        if (
            ovp_cache.get("gid") == gid
            and ovp_cache.get("signature") == current_sig
            and (
                bool(ovp_cache.get("sw3_series_by_id"))
                or bool(ovp_cache.get("power_series_by_id"))
            )
            and (self._analysis_thread is None or not self._analysis_thread.is_alive())
        ):
            messagebox.showinfo(
                action_label,
                "No aligned overlap is available with current settings. Adjust trims or alignment tolerance.",
            )
            return None
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            if self._analysis_running_gid == gid and self._analysis_running_signature == current_sig:
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is still processing in background. Try again when ready.",
                )
            else:
                self._pending_reprocess_gid = gid
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is queued after current background processing.",
                )
            return None
        started = self._start_background_alignment(gid)
        if started:
            messagebox.showinfo(
                action_label,
                f"Preparing '{self.groups[gid].name}' in background. Try again when status says data is ready.",
            )
        else:
            messagebox.showinfo(action_label, "Could not start background processing for this group.")
        return None

    def _ensure_group_ovp_ready(self, gid: str, action_label: str) -> Optional[Dict[str, object]]:
        """Return OVP series cache for gid, or start background processing and warn."""
        if not gid or gid not in self.groups:
            messagebox.showinfo(action_label, "Select a Group.")
            return None
        current_sig = self._analysis_params_signature(self._snapshot_analysis_params())
        cache = self._ovp_series_cache or {}
        if (
            cache.get("gid") == gid
            and cache.get("signature") == current_sig
            and (
                bool(cache.get("sw3_series_by_id"))
                or bool(cache.get("power_series_by_id"))
            )
        ):
            return cache
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            if self._analysis_running_gid == gid and self._analysis_running_signature == current_sig:
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is still processing in background. Try again when ready.",
                )
            else:
                self._pending_reprocess_gid = gid
                messagebox.showinfo(
                    action_label,
                    f"Data for '{self.groups[gid].name}' is queued after current background processing.",
                )
            return None
        started = self._start_background_alignment(gid)
        if started:
            messagebox.showinfo(
                action_label,
                f"Preparing '{self.groups[gid].name}' in background. Try again when status says data is ready.",
            )
        else:
            messagebox.showinfo(action_label, "Could not start background processing for this group.")
        return None

    def _trigger_reprocess_after_reload(self):
        """After reload, reprocess the most relevant group in background."""
        gid = self._pick_group_for_background_prep()
        if gid is not None:
            self._start_background_alignment(gid)
        self._start_background_sw3_preprocess_all()

    def _pick_group_for_background_prep(self) -> Optional[str]:
        """Choose the best group candidate for automatic background processing."""
        if self._last_analyzed_gid and self._last_analyzed_gid in self.groups:
            return self._last_analyzed_gid

        gid = self._current_ovp_group_id()
        if gid and gid in self.groups:
            return gid

        scored = []
        for gid_i, g in self.groups.items():
            if not g.associations:
                continue
            pair_count = sum(len(v) for v in g.associations.values())
            sw3_count = sum(
                1 for fid in g.file_ids
                if fid in self.files and self.files[fid].kind == "sw3"
            )
            pow_count = sum(
                1 for fid in g.file_ids
                if fid in self.files and self.files[fid].kind == "power"
            )
            scored.append((pair_count, min(sw3_count, pow_count), sw3_count + pow_count, gid_i))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][3]

    def _autostart_processing_after_session_load(self):
        """Kick off background processing immediately after session load."""
        self._start_background_sw3_preprocess_all()
        gid = self._pick_group_for_background_prep()
        if gid is None:
            self._set_status("Loaded session (no associated group available for auto-processing).")
            return
        started = self._start_background_alignment(gid)
        if not started:
            self._set_status(
                f"Loaded session. Auto-processing queued for '{self.groups[gid].name}'."
            )

    def _sw3_preprocess_signature(self, only_zero: bool, normalize_to_1m: bool) -> Tuple:
        """Return a stable signature for SW3 plot-cache preprocessing."""
        sw3_ids = tuple(sorted(
            fid for fid, rec in self.files.items()
            if rec.kind == "sw3"
        ))
        return (bool(only_zero), bool(normalize_to_1m), sw3_ids)

    def _start_background_sw3_preprocess_all(self) -> bool:
        """Build SW3 rows cache for IVT/GDO plots in background."""
        only_zero = bool(self.only_yaw_roll_zero.get())
        normalize_to_1m = bool(self.normalize_to_1m.get())
        sig = self._sw3_preprocess_signature(only_zero, normalize_to_1m)
        sw3_ids = list(sig[2])
        if not sw3_ids:
            return False
        if self._sw3_preprocess_thread is not None and self._sw3_preprocess_thread.is_alive():
            if self._sw3_preprocess_running_signature == sig:
                return False
            return False

        files_map = dict(self.files)
        self._sw3_preprocess_job_id += 1
        job_id = self._sw3_preprocess_job_id
        self._sw3_preprocess_running_signature = sig
        self._set_status(f"Background SW3 preprocessing started ({len(sw3_ids)} files)...")

        def _worker():
            cache_updates: Dict[Tuple[str, bool, bool], List[Tuple[float, float]]] = {}
            loaded = 0
            for fid in sw3_ids:
                rec = files_map.get(fid)
                if not rec or rec.kind != "sw3":
                    continue
                sweep = self._load_sweep_from_path(rec.path, warn=False, trace=False)
                if sweep is None:
                    continue
                rows = self._extract_intensity_rows(
                    sweep,
                    only_zero=only_zero,
                    normalize_to_1m=normalize_to_1m,
                    include_row=False,
                )
                cache_updates[(fid, only_zero, normalize_to_1m)] = rows
                loaded += 1
            self._sw3_preprocess_queue.put((job_id, sig, cache_updates, loaded, len(sw3_ids)))

        self._sw3_preprocess_thread = threading.Thread(
            target=_worker,
            name=f"sw3-prep-{job_id}",
            daemon=True,
        )
        self._sw3_preprocess_thread.start()
        self._set_busy(True, "Preparing SW3 cache...")
        self.after(120, self._poll_background_sw3_preprocess)
        return True

    def _poll_background_sw3_preprocess(self):
        """Harvest SW3 background preprocessing results."""
        saw_result = False
        while True:
            try:
                job_id, sig, cache_updates, loaded, total = self._sw3_preprocess_queue.get_nowait()
            except Empty:
                break
            saw_result = True
            if job_id != self._sw3_preprocess_job_id:
                continue
            self._sw3_plot_cache.update(cache_updates)
            self._sw3_preprocess_running_signature = None
            self._set_status(f"SW3 preprocessing ready ({loaded}/{total} files).")

        if self._sw3_preprocess_thread is not None and self._sw3_preprocess_thread.is_alive():
            self._set_busy(True, "Preparing SW3 cache...")
            self.after(120, self._poll_background_sw3_preprocess)
            return
        if saw_result:
            self._sw3_preprocess_thread = None
        self._set_busy(self._is_any_background_busy())

    def _get_sw3_rows_cached(
        self,
        fid: str,
        only_zero: bool,
        normalize_to_1m: bool,
        warn: bool = True,
    ) -> List[Tuple[float, float]]:
        """Get SW3 rows from cache or load on demand."""
        key = (fid, bool(only_zero), bool(normalize_to_1m))
        rows = self._sw3_plot_cache.get(key)
        if rows is not None:
            return rows
        rec = self.files.get(fid)
        if not rec or rec.kind != "sw3":
            return []
        sweep = self._load_sweep_from_path(rec.path, warn=warn, trace=False)
        if sweep is None:
            return []
        rows = self._extract_intensity_rows(
            sweep,
            only_zero=bool(only_zero),
            normalize_to_1m=bool(normalize_to_1m),
            include_row=False,
        )
        self._sw3_plot_cache[key] = rows
        return rows

    # ------------------------------------------------------------------
    # Analysis: Intensity vs Time
    # ------------------------------------------------------------------
    def on_plot_intensity_vs_time(self):
        """Plot intensity vs time for a file or group selection."""
        # Replace any existing IVT plot(s)
        self._close_plots('ivt', also=('spec',))

        mode = self.source_mode.get()
        ema_span = int(self.intensity_ema_span.get())
        trim_start = int(self.trim_start_s.get())
        trim_end   = int(self.trim_end_s.get())
        timeaware_ema = self.time_weighted_ema.get()
        only_zero  = self.only_yaw_roll_zero.get()
        norm_1m    = self.normalize_to_1m.get()
        show_pts   = self.ivt_show_points.get()
        show_line  = self.ivt_show_ema.get()
        p_alpha    = float(self.ivt_point_alpha.get())
        l_alpha    = float(self.ivt_line_alpha.get())

        def rows_from_sw3(fid: str):
            return self._get_sw3_rows_cached(
                fid=fid,
                only_zero=only_zero,
                normalize_to_1m=norm_1m,
                warn=True,
            )

        fig = plt.figure(); ax = fig.add_subplot(111)

        if mode == "file":
            disp = self.ivt_sw3_display.get()
            fid = self._display_to_sw3_id(disp)
            if not fid:
                messagebox.showinfo("Plot", "Select an SW3 file."); return
            rows = rows_from_sw3(fid)

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
                        break
            else:
                messagebox.showinfo("Plot", "No rows after filtering."); return

            if t.size==0:
                messagebox.showinfo("Plot", "No data to plot after trims."); return
            y_ema = ema_adaptive(t, y, ema_span, timeaware_ema)
            th = (t - t[0]) / 3600.0
            if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{self.files[fid].label} (raw)")
            if show_line:
                t_line, y_line = _insert_nan_gaps(t, y_ema, _gap_threshold_s(t, floor_s=60.0))
                ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, label=f"{self.files[fid].label} EMA")
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
                all_rows = []
                for fid in sw3_ids:
                    all_rows.extend(rows_from_sw3(fid))
                if win is not None and all_rows:
                    tmin, tmax = win
                    all_rows = self._filter_rows_by_window(all_rows, tmin, tmax)
                if not all_rows: messagebox.showinfo("Plot", "No rows after filtering."); return
                all_rows.sort(key=lambda x: x[0])
                t = np.array([r[0] for r in all_rows]); y = np.array([r[1] for r in all_rows])
                y_ema = ema_adaptive(t, y, ema_span, timeaware_ema); th = (t - t[0]) / 3600.0
                if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{g.name} (raw)")
                if show_line:
                    t_line, y_line = _insert_nan_gaps(t, y_ema, _gap_threshold_s(t, floor_s=60.0))
                    ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, label=f"{g.name} EMA")
            else:
                for i, fid in enumerate(sw3_ids):
                    rows = rows_from_sw3(fid)
                    if win is not None and rows:
                        tmin, tmax = win
                        rows = self._filter_rows_by_window(rows, tmin, tmax)
                    if not rows: continue
                    t = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
                    y_ema = ema_adaptive(t, y, ema_span, timeaware_ema); th = (t - t[0]) / 3600.0
                    c = colors[i]
                    if show_pts: ax.scatter(th, y, s=8, alpha=p_alpha, label=f"{self.files[fid].label} (raw)", color=c)
                    if show_line:
                        t_line, y_line = _insert_nan_gaps(t, y_ema, _gap_threshold_s(t, floor_s=60.0))
                        ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, label=f"{self.files[fid].label} EMA", color=c)

        ax.set_title("Measured Light Intensity vs Time")
        ax.set_xlabel("Time since start (hours)")
        ax.set_ylabel("Normalized Intensity at 1 m (µW/cm²)")
        ax.grid(True, linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        outside = _apply_smart_legend(ax, handles, labels)
        if outside:
            fig.tight_layout(rect=(0, 0, 0.78, 1))
        else:
            fig.tight_layout()
        self._register_fig('ivt', fig)
        self._attach_wavelength_inspector_to_ivt(fig)
        plt.show(block=False)

    # ------------------------------------------------------------------
    # Analysis: Build aligned frame for a group (applies group trims)
    # ------------------------------------------------------------------
    def _build_aligned_for_group(
        self,
        gid: str,
        analysis_params: Optional[Dict[str, object]] = None,
        ui_progress: bool = True,
        return_plot_cache: bool = False,
    ):
        """Return a merged intensity/power dataframe for a group or None."""
        g = self.groups.get(gid)
        if not g or not g.associations:
            return (None, None) if return_plot_cache else None
        files_map = dict(self.files)

        p = dict(analysis_params or self._snapshot_analysis_params())
        ema_int = int(p["intensity_ema_span"])
        ema_pow_manual = int(p["power_ema_span"])
        tol_s = int(p["align_tolerance_s"])
        timeaware_ema = bool(p["time_weighted_ema"])
        auto_power_ema = bool(p["auto_power_ema"])
        only_zero = bool(p["only_yaw_roll_zero"])
        norm_1m = bool(p["normalize_to_1m"])
        base_resample_s = max(1, int(p["resample_seconds"]))

        def report(status: str):
            if not ui_progress:
                return
            self._set_status(status)
            self.update_idletasks()

        frames = []
        win = self._group_time_window(g, files_map=files_map)

        # ------------------------------------------------------------------
        # Stage 1: preprocess SW3 once per file
        # ------------------------------------------------------------------
        sw3_ids = list(g.associations.keys())
        sw3_df_by_id: Dict[str, pd.DataFrame] = {}
        sw3_range_by_id: Dict[str, Tuple[float, float]] = {}
        pair_list: List[Tuple[str, str]] = []
        power_to_sw3: Dict[str, List[str]] = {}

        for i, sw3_id in enumerate(sw3_ids, start=1):
            report(f"Preparing SW3 {i}/{len(sw3_ids)}...")

            sw3_rec = files_map.get(sw3_id)
            if not sw3_rec or sw3_rec.kind != "sw3":
                continue

            sw_rows = self._get_sw3_rows_cached(
                fid=sw3_id,
                only_zero=only_zero,
                normalize_to_1m=norm_1m,
                warn=ui_progress,
            )
            if win is not None and sw_rows:
                tmin_win, tmax_win = win
                sw_rows = self._filter_rows_by_window(sw_rows, tmin_win, tmax_win)
            if not sw_rows:
                continue

            t_sw = np.array([r[0] for r in sw_rows], dtype=float)
            y_sw = np.array([r[1] for r in sw_rows], dtype=float)
            sw3_df_by_id[sw3_id] = pd.DataFrame(
                {"timestamp": t_sw, "intensity_ema": ema_adaptive(t_sw, y_sw, ema_int, timeaware_ema)}
            )
            sw3_range_by_id[sw3_id] = (float(t_sw.min()), float(t_sw.max()))

            for pid in g.associations.get(sw3_id, set()):
                pow_rec = files_map.get(pid)
                if not pow_rec or pow_rec.kind != "power":
                    continue
                pair_list.append((sw3_id, pid))
                power_to_sw3.setdefault(pid, []).append(sw3_id)

        if not pair_list:
            return (None, None) if return_plot_cache else None

        # Auto-tune power EMA from optics EMA and observed group cadence.
        sw3_step_candidates = []
        for sid, df_sw in sw3_df_by_id.items():
            ts = df_sw["timestamp"].to_numpy(dtype=float)
            sw3_step_candidates.append(_median_positive_step_s(ts))
        sw3_step_candidates = [x for x in sw3_step_candidates if np.isfinite(x) and x > 0]
        sw3_median_step_s = float(np.median(sw3_step_candidates)) if sw3_step_candidates else 1.0

        power_meta_step_candidates = []
        for pid in power_to_sw3.keys():
            rec = files_map.get(pid)
            if not rec:
                continue
            step_s = _estimate_step_s_from_meta(rec.meta)
            if step_s is not None and np.isfinite(step_s) and step_s > 0:
                power_meta_step_candidates.append(float(step_s))
        power_step_s = float(np.median(power_meta_step_candidates)) if power_meta_step_candidates else None

        if auto_power_ema:
            ema_pow = recommend_power_ema_span(
                intensity_ema_span=ema_int,
                sw3_median_step_s=sw3_median_step_s,
                resample_seconds=base_resample_s,
                power_step_s=power_step_s,
            )
            if ui_progress and ema_pow != self.power_ema_span.get():
                self.power_ema_span.set(ema_pow)
            report(f"Auto power EMA: {ema_pow} (optics span {ema_int}, sw3 step {sw3_median_step_s:.2f}s)")
        else:
            ema_pow = max(1, ema_pow_manual)

        # ------------------------------------------------------------------
        # Stage 2: preprocess power once per file across union window
        # ------------------------------------------------------------------
        power_series_by_id: Dict[str, pd.DataFrame] = {}
        power_ids = list(power_to_sw3.keys())
        for i, pid in enumerate(power_ids, start=1):
            report(f"Preparing power {i}/{len(power_ids)}...")

            pow_rec = files_map.get(pid)
            if not pow_rec or pow_rec.kind != "power":
                continue

            try:
                power_df, _ = self._load_power_csv_with_meta(pow_rec.path)
            except Exception as e:
                if ui_progress:
                    messagebox.showwarning("Load error", f"Failed to load CSV {pow_rec.path}\n{e}")
                continue

            linked_sw3 = [sid for sid in power_to_sw3.get(pid, []) if sid in sw3_range_by_id]
            if not linked_sw3:
                continue
            power_tmin = min(sw3_range_by_id[sid][0] for sid in linked_sw3) - tol_s
            power_tmax = max(sw3_range_by_id[sid][1] for sid in linked_sw3) + tol_s
            power_duration_s = max(0.0, float(power_tmax - power_tmin))
            effective_resample_s = choose_effective_resample_seconds(
                duration_s=power_duration_s,
                requested_seconds=base_resample_s,
            )
            # Keep smoothing timescale stable when auto-upscaling resample.
            ema_pow_effective = max(1, int(round(float(ema_pow) * base_resample_s / effective_resample_s)))
            if effective_resample_s > base_resample_s:
                report(
                    f"Preparing power {i}/{len(power_ids)} (auto resample {effective_resample_s}s for {power_duration_s/3600.0:.1f}h)..."
                )

            power_series = _prepare_power_series(
                power_df,
                power_tmin,
                power_tmax,
                ema_pow_effective,
                effective_resample_s,
                timeaware_ema,
            )
            if power_series is None or power_series.empty:
                continue
            power_series_by_id[pid] = power_series

        plot_cache = {
            "gid": gid,
            "signature": self._analysis_params_signature(p),
            "sw3_series_by_id": sw3_df_by_id,
            "power_series_by_id": power_series_by_id,
            "sw3_label_by_id": {sid: files_map[sid].label for sid in sw3_df_by_id.keys() if sid in files_map},
            "power_label_by_id": {pid: files_map[pid].label for pid in power_series_by_id.keys() if pid in files_map},
        }

        # ------------------------------------------------------------------
        # Stage 3: pairwise alignment from cached/prepared data
        # ------------------------------------------------------------------
        for i, (sw3_id, pid) in enumerate(pair_list, start=1):
            report(f"Aligning pair {i}/{len(pair_list)}...")

            df_sw = sw3_df_by_id.get(sw3_id)
            pdf_full = power_series_by_id.get(pid)
            sw3_rec = files_map.get(sw3_id)
            pow_rec = files_map.get(pid)
            if df_sw is None or pdf_full is None or sw3_rec is None or pow_rec is None:
                continue

            sw_tmin, sw_tmax = sw3_range_by_id[sw3_id]
            pdf = _slice_time_window(pdf_full, sw_tmin - tol_s, sw_tmax + tol_s, ts_col="timestamp")
            if pdf is None or pdf.empty:
                continue

            df_join = merge_asof_seconds(df_sw, pdf, tol_s).dropna(subset=["power_ema"])
            if df_join.empty:
                continue
            df_join["group_id"] = gid
            df_join["sw3_id"] = sw3_id
            df_join["power_id"] = pid
            df_join["sw3_label"] = sw3_rec.label
            df_join["power_label"] = pow_rec.label
            frames.append(df_join)

        if not frames:
            return (None, plot_cache) if return_plot_cache else None
        aligned = pd.concat(frames, ignore_index=True).sort_values("timestamp_s")
        if return_plot_cache:
            return aligned, plot_cache
        return aligned

    # ------------------------------------------------------------------
    # Analysis: OVP (time series) and correlation/scatter
    # ------------------------------------------------------------------
    def on_analyze_group(self):
        """Plot optics vs power using prepared series (or queue background prep)."""
        gid = self._current_ovp_group_id()
        ovp_cache = self._ensure_group_ovp_ready(gid, "Analyze")
        if ovp_cache is None:
            return
        aligned = None
        current_sig = self._analysis_params_signature(self._snapshot_analysis_params())
        if (
            self._aligned_cache is not None
            and not self._aligned_cache.empty
            and self._aligned_cache_gid == gid
            and self._aligned_cache_signature == current_sig
        ):
            aligned = self._aligned_cache
        self._plot_ovp_from_series(gid, ovp_cache, aligned)

    def _plot_ovp_from_series(
        self,
        gid: str,
        ovp_cache: Dict[str, object],
        aligned: Optional[pd.DataFrame] = None,
    ):
        """Render OVP plot from prepared SW3/power series cache."""
        # Replace any existing OVP plot(s)
        self._close_plots('ovp')

        show_pts = self.ovp_show_points.get()
        show_int = self.ovp_show_int_ema.get()
        show_pow = self.ovp_show_pow_ema.get()
        p_alpha  = float(self.ovp_point_alpha.get())
        l_alpha  = float(self.ovp_line_alpha.get())

        sw3_series_by_id = dict(ovp_cache.get("sw3_series_by_id", {}))
        power_series_by_id = dict(ovp_cache.get("power_series_by_id", {}))
        sw3_label_by_id = dict(ovp_cache.get("sw3_label_by_id", {}))
        power_label_by_id = dict(ovp_cache.get("power_label_by_id", {}))

        if not sw3_series_by_id and not power_series_by_id:
            messagebox.showinfo("Analyze", "No optical/power series available to plot.")
            return

        sw3_ids = sorted(sw3_series_by_id.keys())
        power_ids = sorted(power_series_by_id.keys())
        sw3_colors = get_solarized_colors(len(sw3_ids))
        power_colors = get_solarized_colors(len(power_ids))
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        t0_candidates = []
        for sid in sw3_ids:
            df_sw = sw3_series_by_id[sid]
            if df_sw is not None and not df_sw.empty:
                t0_candidates.append(float(df_sw["timestamp"].min()))
        for pid in power_ids:
            df_pow = power_series_by_id[pid]
            if df_pow is not None and not df_pow.empty:
                t0_candidates.append(float(df_pow["timestamp"].min()))
        if aligned is not None and not aligned.empty:
            t0_candidates.append(float(aligned["timestamp_s"].min()))
        if not t0_candidates:
            messagebox.showinfo("Analyze", "No data points available to plot.")
            return
        t0_global = min(t0_candidates)

        for i, sid in enumerate(sw3_ids):
            df_sw = sw3_series_by_id[sid]
            if df_sw is None or df_sw.empty:
                continue
            t_sw = df_sw["timestamp"].to_numpy(dtype=float)
            y_sw = df_sw["intensity_ema"].to_numpy(dtype=float)
            c_int = lighten(sw3_colors[i], 0.42)
            sw_label = sw3_label_by_id.get(sid, sid)
            gap_floor = max(float(self.align_tolerance_s.get()) * 3.0, 30.0)
            gap_s = _gap_threshold_s(t_sw, floor_s=gap_floor)
            if show_int:
                t_line, y_line = _insert_nan_gaps(t_sw, y_sw, gap_s)
                ax1.plot(
                    (t_line - t0_global) / 3600.0,
                    y_line,
                    label=f"{sw_label} (optical irradiance)",
                    alpha=l_alpha,
                    color=c_int,
                    linewidth=2,
                )
            if show_pts:
                ax1.scatter((t_sw - t0_global) / 3600.0, y_sw, s=8, alpha=p_alpha, color=c_int)

        for i, pid in enumerate(power_ids):
            df_pow = power_series_by_id[pid]
            if df_pow is None or df_pow.empty:
                continue
            t_pow = df_pow["timestamp"].to_numpy(dtype=float)
            y_pow = df_pow["power_ema"].to_numpy(dtype=float)
            c_pow = darken(power_colors[i], 0.35)
            pw_label = power_label_by_id.get(pid, pid)
            gap_floor = max(float(self.resample_seconds.get()) * 3.0, 30.0)
            gap_s = _gap_threshold_s(t_pow, floor_s=gap_floor)
            if show_pow:
                t_line, y_line = _insert_nan_gaps(t_pow, y_pow, gap_s)
                ax2.plot(
                    (t_line - t0_global) / 3600.0,
                    y_line,
                    label=f"{pw_label} (power)",
                    alpha=l_alpha,
                    color=c_pow,
                    linewidth=2,
                )

        ax1.set_xlabel("Time since earliest series start (hours)")
        ax1.set_ylabel("Intensity (µW/cm²)")
        ax2.set_ylabel("Power (W)", labelpad=10)
        ax2.tick_params(axis="y", pad=4)
        ax1.grid(True, linestyle="--", alpha=0.5)

        opt_handles, opt_labels = ax1.get_legend_handles_labels()
        pow_handles, pow_labels = ax2.get_legend_handles_labels()
        total_labels = len(opt_labels) + len(pow_labels)
        outside = total_labels > 6 or max([len(x) for x in (opt_labels + pow_labels)] + [0]) > 36
        if outside:
            if opt_labels:
                leg_opt = fig.legend(
                    opt_handles,
                    opt_labels,
                    title="Optical",
                    loc="upper left",
                    bbox_to_anchor=(0.695, 0.93),
                    bbox_transform=fig.transFigure,
                    borderaxespad=0.0,
                    fontsize=8,
                    frameon=True,
                )
            if pow_labels:
                fig.legend(
                    pow_handles,
                    pow_labels,
                    title="Power",
                    loc="lower left",
                    bbox_to_anchor=(0.695, 0.10),
                    bbox_transform=fig.transFigure,
                    borderaxespad=0.0,
                    fontsize=8,
                    frameon=True,
                )
        else:
            seen = {}
            for h, lab in zip(opt_handles + pow_handles, opt_labels + pow_labels):
                if lab not in seen:
                    seen[lab] = h
            _apply_smart_legend(ax1, list(seen.values()), list(seen.keys()))

        ax1.set_title(f"Optical irradiance and power vs time — {self.groups[gid].name}")
        if outside:
            # Reserve a dedicated right-side column for legends so they do not
            # overlap the right y-axis ticks/label.
            fig.subplots_adjust(left=0.08, right=0.60, top=0.92, bottom=0.12)
        else:
            fig.tight_layout()
        self._register_fig('ovp', fig)
        plt.show(block=False)
        self._set_status(f"Analyzed group '{self.groups[gid].name}' — time series plotted.")

    def on_corr_and_scatter(self):
        """Plot cross-correlation and scatter/regression for aligned data."""
        # Replace any existing correlation/scatter plots
        self._close_plots('corr')

        gid = self._current_ovp_group_id()
        aligned = self._ensure_group_aligned_ready(gid, "Correlation")
        if aligned is None:
            return

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
        handles, labels = axs.get_legend_handles_labels()
        outside = _apply_smart_legend(axs, handles, labels)
        if outside:
            fig_sc.tight_layout(rect=(0, 0, 0.78, 1))
        else:
            fig_sc.tight_layout()
        self._register_fig('corr', fig_sc); plt.show(block=False)

    def on_export_aligned_csv(self):
        """Export the last aligned dataframe to CSV."""
        gid = self._current_ovp_group_id()
        aligned = self._ensure_group_aligned_ready(gid, "Export")
        if aligned is None:
            return
        path = filedialog.asksaveasfilename(title="Export Aligned CSV", defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")])
        if not path: return
        df = aligned.rename(columns={
            "timestamp_s":"Timestamp", "intensity_ema":"Intensity_EMA_uW_cm2", "power_ema":"Power_EMA_W"
        }).copy()
        keep = ["Timestamp","Intensity_EMA_uW_cm2","Power_EMA_W","group_id","sw3_id","power_id","sw3_label","power_label"]
        df[keep].to_csv(path, index=False)
        self._set_status(f"Exported aligned CSV to {path}")

    # ------------------------------------------------------------------
    # Analysis: Group Decay Overlay (applies trim)
    # ------------------------------------------------------------------
    def on_plot_group_decay(self, selected_only: bool):
        """Plot decay overlay for selected or all groups."""
        # Replace any existing GDO plot(s)
        self._close_plots('gdo')

        ema_span = int(self.intensity_ema_span.get())
        timeaware_ema = self.time_weighted_ema.get()
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
                rows_all.extend(
                    self._get_sw3_rows_cached(
                        fid=sid,
                        only_zero=only_zero,
                        normalize_to_1m=norm_1m,
                        warn=True,
                    )
                )
            if win is not None and rows_all:
                tmin, tmax = win
                rows_all = self._filter_rows_by_window(rows_all, tmin, tmax)
            if not rows_all: continue
            rows_all.sort(key=lambda x:x[0])
            t = np.array([r[0] for r in rows_all]); y = np.array([r[1] for r in rows_all])
            y_ema = ema_adaptive(t, y, ema_span, timeaware_ema)
            y_pct = (y_ema/y_ema.max())*100.0 if y_ema.max()>0 else np.zeros_like(y_ema)
            th = (t - t[0]) / 3600.0
            c = base_colors[i]
            if show_pts: ax.scatter(th, y_pct, s=8, alpha=p_alpha, color=c, label=f"{g.name} (pts)")
            if show_line:
                t_line, y_line = _insert_nan_gaps(t, y_pct, _gap_threshold_s(t, floor_s=60.0))
                ax.plot((t_line - t[0]) / 3600.0, y_line, linewidth=2, alpha=l_alpha, color=c, label=g.name)

        ax.set_title("Decay Overlay by Group (EMA, normalized to each group’s peak)")
        ax.set_xlabel("Hours since group start"); ax.set_ylabel("Intensity (% of peak)")
        ax.grid(True, linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        outside = _apply_smart_legend(ax, handles, labels)
        if outside:
            fig.tight_layout(rect=(0, 0, 0.78, 1))
        else:
            fig.tight_layout()
        self._register_fig('gdo', fig)
        plt.show(block=False)

    # ---- Plot utils ----
    # ------------------------------------------------------------------
    # Interactive: Wavelength spectrum inspector for IVT plot
    # ------------------------------------------------------------------
    def _build_ivt_series_data_for_current_selection(self):
        """
        Build IVT series (timestamps in hours as plotted) **plus** row/sweep refs
        from the currently selected File or Group, honoring trims, zero-only, and
        1 m normalization toggle.
        Returns: list of dicts: {'label','t_hours','rows','sweeps'}
        """
        mode = self.source_mode.get()
        trim_start = int(self.trim_start_s.get())
        trim_end   = int(self.trim_end_s.get())
        only_zero  = self.only_yaw_roll_zero.get()
        norm_1m    = self.normalize_to_1m.get()

        ensure_eegbin_imported()

        def _extract_rows(fid: str):
            rec = self.files.get(fid)
            if not rec or rec.kind != "sw3":
                return []
            sweep = self._load_sweep_from_path(rec.path, warn=False, trace=False)
            if sweep is None:
                return []
            return self._extract_intensity_rows(sweep, only_zero, norm_1m, include_row=True)

        series = []

        if mode == "file":
            disp = self.ivt_sw3_display.get()
            fid = self._display_to_sw3_id(disp)
            if not fid:
                return []
            rows = _extract_rows(fid)
            if not rows:
                return []

            import numpy as np
            t = np.array([r[0] for r in rows], dtype=float)
            if t.size == 0:
                return []
            t0 = t[0]
            # Per-action trims
            mask = np.ones_like(t, dtype=bool)
            if trim_start > 0:
                mask &= (t - t0) >= trim_start
            if trim_end > 0:
                tend = t[-1]
                mask &= (tend - t) >= trim_end
            # Clip to group window if this file is in a trimmed group
            for g in self.groups.values():
                if fid in g.file_ids:
                    win = self._group_time_window(g)
                    if win is not None:
                        tmin, tmax = win
                        mask &= (t >= tmin) & (t <= tmax)
                    break
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                return []
            th = (t[idxs] - t[idxs][0]) / 3600.0
            series.append({
                "label": self.files[fid].label,
                "t_hours": th,
                "rows": [rows[i][2] for i in idxs],
                "sweeps": [rows[i][3] for i in idxs],
            })
        else:
            # Group mode
            gdisp = self.ivt_group_display.get()
            gid = self._display_to_group_id(gdisp)
            if not gid or gid not in self.groups:
                return []
            g = self.groups[gid]
            win = self._group_time_window(g)
            sw3_ids = [fid for fid in g.file_ids if self.files.get(fid) and self.files[fid].kind == "sw3"]
            if not sw3_ids:
                return []

            if self.combine_group_sw3.get():
                all_rows = []
                for fid in sw3_ids:
                    rows = _extract_rows(fid)
                    all_rows.extend(rows)
                if win is not None and all_rows:
                    tmin, tmax = win
                    all_rows = [rv for rv in all_rows if (rv[0] >= tmin and rv[0] <= tmax)]
                if not all_rows:
                    return []
                all_rows.sort(key=lambda x: x[0])
                import numpy as np
                t = np.array([rv[0] for rv in all_rows], dtype=float)
                th = (t - t[0]) / 3600.0
                series.append({
                    "label": g.name,
                    "t_hours": th,
                    "rows": [rv[2] for rv in all_rows],
                    "sweeps": [rv[3] for rv in all_rows],
                })
            else:
                for fid in sw3_ids:
                    rows = _extract_rows(fid)
                    if win is not None and rows:
                        tmin, tmax = win
                        rows = [rv for rv in rows if (rv[0] >= tmin and rv[0] <= tmax)]
                    if not rows:
                        continue
                    import numpy as np
                    t = np.array([rv[0] for rv in rows], dtype=float)
                    th = (t - t[0]) / 3600.0
                    series.append({
                        "label": self.files[fid].label,
                        "t_hours": th,
                        "rows": [rv[2] for rv in rows],
                        "sweeps": [rv[3] for rv in rows],
                    })

        return series

    def _attach_wavelength_inspector_to_ivt(self, fig: plt.Figure):
        """
        Attach pick and click handlers to IVT plot to open a Spectrum window.
        Respects toolbar zoom/pan, so it won't trigger while using those tools.
        """
        if not fig.axes:
            return
        ax = fig.axes[0]
        series = self._build_ivt_series_data_for_current_selection()
        if not series:
            return
        ax._ivt_series = series  # stash on axes

        # --- Toolbar guard: ignore while zoom/pan is active ---
        def _toolbar_mode_active():
            try:
                tb = getattr(fig.canvas.manager, "toolbar", None)
            except Exception:
                tb = None
            if tb is None:
                tb = getattr(fig.canvas, "toolbar", None)
            mode = ""
            try:
                mode = getattr(tb, "mode", "") or ""
            except Exception:
                mode = ""
            m = str(mode).lower()
            if m and (("zoom" in m) or ("pan" in m)):
                return True
            try:
                active = getattr(tb, "_active", None)
                if active:
                    return True
            except Exception:
                pass
            return False

        # --- Open-once guard: dedupe pick + click ---
        import time
        if not hasattr(fig, "_spec_click_guard"):
            fig._spec_click_guard = {"last_open_key": None, "t_last_open": 0.0}

        def _evt_key(mev):
            return (int(getattr(mev, "x", -1)),
                    int(getattr(mev, "y", -1)),
                    getattr(mev, "button", None))

        def _open_once(mouse_event, opener_callable, dt=0.25):
            if _toolbar_mode_active():
                return
            if getattr(mouse_event, "button", None) != 1:
                return  # left click only
            g = fig._spec_click_guard
            key = _evt_key(mouse_event)
            now = time.monotonic()
            if g["last_open_key"] == key and (now - g["t_last_open"] < dt):
                return
            g["last_open_key"] = key
            g["t_last_open"] = now
            opener_callable()

        # Tag artists with their base series by label (strip EMA/raw suffixes)
        def _base_label(lbl: str) -> str:
            if not isinstance(lbl, str): return ""
            s = lbl.replace(" EMA", "").replace(" (raw)", "")
            return s

        label_to_series = {s["label"]: s for s in series}
        for artist in list(ax.lines) + list(ax.collections):
            try:
                artist.set_picker(True)
            except Exception:
                pass
            sdict = label_to_series.get(_base_label(getattr(artist, "get_label", lambda: "")()))
            setattr(artist, "_ivt_series_meta", sdict)

        import numpy as np
        def nearest_index(sdict, xh):
            th = sdict["t_hours"]
            if getattr(th, "size", len(th)) == 0:
                return None
            return int(np.argmin(np.abs(th - float(xh))))

        def _nearest_with_spectrum(sdict, idx):
            rows = sdict["rows"]
            th = sdict["t_hours"]
            def has_spec(r):
                sr = getattr(r.capture, "spectral_result", None)
                return sr is not None and len(sr) > 0
            if idx is None:
                return None
            if has_spec(rows[idx]):
                return idx
            have = [i for i, r in enumerate(rows) if has_spec(r)]
            if not have:
                return None
            return min(have, key=lambda i: abs(th[i] - th[idx]))

        def show_for_index(sdict, idx, reason="click"):
            if sdict is None or idx is None:
                return
            idx2 = _nearest_with_spectrum(sdict, idx)
            if idx2 is None:
                return
            row = sdict["rows"][idx2]; sweep = sdict["sweeps"][idx2]
            self._show_wavelength_plot(row, sweep, sdict["label"])

        def on_pick(event):
            # Skip while toolbar is active or not a left click
            if _toolbar_mode_active():
                return
            mev = getattr(event, "mouseevent", None)
            if getattr(mev, "button", None) != 1:
                return

            def _open_from_pick():
                sdict = getattr(event.artist, "_ivt_series_meta", None)
                if sdict is None:
                    return
                if hasattr(event.artist, "get_offsets") and getattr(event, "ind", None):
                    show_for_index(sdict, int(event.ind[0]), reason="pick-point")
                else:
                    if mev and mev.xdata is not None:
                        show_for_index(sdict, nearest_index(sdict, mev.xdata), reason="pick-line")

            _open_once(mev, _open_from_pick)

        def on_click(event):
            if _toolbar_mode_active():
                return
            if event.button != 1:
                return
            if event.inaxes is not ax or event.xdata is None:
                return

            def _open_from_click():
                best = None
                best_dx = float("inf")
                for sdict in ax._ivt_series:
                    idx = nearest_index(sdict, event.xdata)
                    if idx is None:
                        continue
                    dx = abs(sdict["t_hours"][idx] - event.xdata)
                    if dx < best_dx:
                        best_dx = dx
                        best = (sdict, idx)
                if best is not None:
                    show_for_index(best[0], best[1], reason="click")

            _open_once(event, _open_from_click)

        if not hasattr(fig, "_spectrum_handlers_connected"):
            fig.canvas.mpl_connect("pick_event", on_pick)
            fig.canvas.mpl_connect("button_press_event", on_click)
            fig._spectrum_handlers_connected = True

    def _show_wavelength_plot(self, row, sweep, src_label: str):
        """
        New window with Wavelength (nm) vs Spectral Intensity,
        applying 1 m scaling if enabled.
        """
        import numpy as np
        wvls = np.asarray(getattr(sweep, "spectral_wavelengths", []), dtype=float)
        vals = np.asarray(list(getattr(row.capture, "spectral_result", []) or []), dtype=float)
        if wvls.size == 0 or vals.size == 0:
            return
        if self.normalize_to_1m.get():
            dist_m = float(getattr(row.coords, "lin_mm", 0.0) or 0.0) / 1000.0
            if dist_m > 0:
                vals = vals * (dist_m ** 2)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(wvls, vals, linewidth=1.6)
        t_s = getattr(row, "timestamp", None)
        yaw = getattr(row.coords, "yaw_deg", None)
        roll = getattr(row.coords, "roll_deg", None)
        title_bits = []
        if t_s is not None:
            title_bits.append(f"t={t_s:.0f}s")
        if yaw is not None and roll is not None:
            title_bits.append(f"yaw {yaw}, roll {roll}")
        if src_label:
            title_bits.append(src_label)
        ax.set_title("Spectrum — " + " | ".join(title_bits))
        units = getattr(sweep, "spectral_units", "")
        ylab = f"Spectral Intensity ({units})"
        if self.normalize_to_1m.get():
            ylab += ", scaled to 1 m"
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        self._register_fig('spec', fig)
        try:
            plt.show(block=False)
        except Exception:
            pass

    def _register_fig(self, cat: str, fig: plt.Figure):
        """Track a figure handle under a category key."""
        self._figs.setdefault(cat, []).append(fig)

    def _close_plots(self, cat: str, also: Tuple[str, ...]=()):
        """Close plot windows for one or more categories."""
        cats = (cat,) + tuple(also)
        for c in cats:
            figs = self._figs.get(c, [])
            for f in figs:
                try: plt.close(f)
                except Exception: pass
            self._figs[c] = []
        self._set_status(f"Closed {', '.join(cats)} plot windows.")

    def _select_all_groups(self):
        """Select all groups in the decay overlay listbox."""
        self.lb_groups_select.selection_set(0, tk.END)

    def on_save_last_figure(self):
        """Save the most recently created figure (any category)."""
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
        """Set the status bar text."""
        self.status_var.set(text)
        try:
            self.update_idletasks()
        except Exception:
            pass

    def _is_any_background_busy(self) -> bool:
        """Return True when any background preprocessing worker is active."""
        a_busy = self._analysis_thread is not None and self._analysis_thread.is_alive()
        s_busy = self._sw3_preprocess_thread is not None and self._sw3_preprocess_thread.is_alive()
        r_busy = self._reload_thread is not None and self._reload_thread.is_alive()
        return bool(a_busy or s_busy or r_busy)

    def _set_busy(self, busy: bool, text: str = "Processing..."):
        """Toggle a subtle busy indicator in the status bar."""
        busy = bool(busy)
        if busy:
            busy_text = f"Background: {text}"
            self._busy_var.set(busy_text)
            self._set_status(busy_text)
            if not self._busy_active:
                self._busy_pb.pack(side=tk.RIGHT, padx=(0, 6), pady=4)
                self._busy_pb.start(12)
                self._busy_active = True
            return
        self._busy_var.set("Background: idle")
        if self._busy_active:
            self._busy_pb.stop()
            self._busy_pb.pack_forget()
            self._busy_active = False

    def _next_file_id(self) -> str:
        """Return the next available file id (F###)."""
        i=1
        while f"F{i:03d}" in self.files: i+=1
        return f"F{i:03d}"

    def _next_group_id(self) -> str:
        """Return the next available group id (G###)."""
        i=1
        while f"G{i:03d}" in self.groups: i+=1
        return f"G{i:03d}"

    def _next_report_scan_id(self) -> str:
        """Return the next available report scan id (RS###)."""
        i = 1
        while f"RS{i:03d}" in self.report_scans:
            i += 1
        return f"RS{i:03d}"

    def _next_report_csv_id(self) -> str:
        """Return the next available report CSV id (RC###)."""
        i = 1
        while f"RC{i:03d}" in self.report_csvs:
            i += 1
        return f"RC{i:03d}"

# ------------------------------------------------------------------
# Association dialog (friendly labels; shows full mapping)
# ------------------------------------------------------------------

class AssociationDialog(tk.Toplevel):
    """Dialog for mapping SW3 files to Power files within a group."""
    def __init__(self, master: App, group: GroupRecord, sw3_list, power_list):
        """Build the associations editor UI."""
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
        """Associate the selected SW3 file with selected power files."""
        idx_sw = self.lb_sw3.curselection(); idx_pw = self.lb_pow.curselection()
        if not idx_sw or not idx_pw: return
        sw_id = self.sw3_ids[idx_sw[0]]; pow_ids = [self.pow_ids[i] for i in idx_pw]
        s = self.group.associations.get(sw_id, set()); s |= set(pow_ids)
        self.group.associations[sw_id] = s; self.updated = True; self._refresh_mapping()

    def _remove_mapping(self):
        """Remove associations for the selected SW3 file."""
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
        """Refresh the visible mapping table."""
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
    """Launch the GUI application."""
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
