#!/usr/bin/env python3
"""Validate Aerolamp reporting fixture inputs/outputs and phase classification."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eegbin
from gui import suggest_report_phase_tag, suggest_report_scan_role


EXPECTED_STATIC_FILES = [
    "apollo_dcjackpower_100percent.py",
    "apollo_dcjackpower_100percent_1749243198.sw3",
    "Apollo-DCPOWER-Seasoned-1m1749948341.sw3",
    "apollodev-dev_wfm_0.csv",
    "apollodev-pwr_log.csv",
    "aerolamp_dev.png",
    "aerolamp_dev_axes.png",
]

EXPECTED_OUTPUT_FILES = [
    "aerolamp devkit--27c4ca60-0da7-4669-8cc7-36746a06fb62--R00.html",
    "aerolamp devkit--27c4ca60-0da7-4669-8cc7-36746a06fb62--R00.ies",
    "aerolamp devkit--27c4ca60-0da7-4669-8cc7-36746a06fb62--R00-spectrum.csv",
]

EXPECTED_OUTPUT_MARKERS = {
    "aerolamp devkit--27c4ca60-0da7-4669-8cc7-36746a06fb62--R00.html": "Aerolamp DevKit",
    "aerolamp devkit--27c4ca60-0da7-4669-8cc7-36746a06fb62--R00.ies": "IESNA",
    "aerolamp devkit--27c4ca60-0da7-4669-8cc7-36746a06fb62--R00-spectrum.csv": "wavelength",
}

EXPECTED_SW3 = {
    "apollo_dcjackpower_100percent_1749243198.sw3": {
        "scan_role": "Complete Dataset",
        "tag_counts": {
            "Warm-up": 80,
            "Spectrum Point": 1,
            "R2 Pullback": 1,
            "Loose Spectrometer Scan": 1,
            "Tight Spectrometer Scan": 1,
        },
    },
    "Apollo-DCPOWER-Seasoned-1m1749948341.sw3": {
        "scan_role": "Main Scan",
        "tag_counts": {
            "Warm-up": 1,
            "Spectrum Point": 1,
            "R2 Pullback": 1,
            "Loose Spectrometer Scan": 1,
            "Tight Spectrometer Scan": 1,
        },
    },
}


def scan_phase_summary(sw3_path: Path):
    with sw3_path.open("rb") as fd:
        scan = eegbin.load_eegbin3(fd.read(), from_path=str(sw3_path))

    phases = []
    for idx, ph in enumerate(scan.phases):
        members = [r for r in (ph.members or []) if getattr(r, "timestamp", None) is not None]
        ts = [r.timestamp for r in members]
        start = float(min(ts)) if ts else None
        end = float(max(ts)) if ts else None
        phases.append(
            {
                "index": idx,
                "name": getattr(ph, "name", "") or f"Phase {idx + 1}",
                "phase_type": getattr(getattr(ph, "phase_type", None), "name", ""),
                "axis": getattr(getattr(ph, "major_axis", None), "name", ""),
                "start_ts": start,
                "end_ts": end,
            }
        )
    tags = [suggest_report_phase_tag(ph) for ph in phases]
    return suggest_report_scan_role(phases), Counter(tags), phases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "OSLUV Data" / "Aerolamp"),
        help="Path to Aerolamp fixture directory",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    outputs_dir = dataset_dir / "Outputs"

    failures = []

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 2

    print(f"Validating dataset: {dataset_dir}")

    for rel in EXPECTED_STATIC_FILES:
        path = dataset_dir / rel
        if not path.exists():
            failures.append(f"Missing file: {path}")

    for rel in EXPECTED_OUTPUT_FILES:
        path = outputs_dir / rel
        if not path.exists():
            failures.append(f"Missing output: {path}")
            continue
        if path.stat().st_size <= 0:
            failures.append(f"Empty output: {path}")
            continue
        marker = EXPECTED_OUTPUT_MARKERS.get(rel)
        if marker:
            text = path.read_text(errors="ignore")
            if marker.lower() not in text.lower():
                failures.append(f"Output marker {marker!r} not found in {path}")

    for name, expected in EXPECTED_SW3.items():
        sw3_path = dataset_dir / name
        if not sw3_path.exists():
            failures.append(f"Missing SW3 fixture: {sw3_path}")
            continue

        role, tag_counts, phases = scan_phase_summary(sw3_path)
        print(f"\n{name}")
        print(f"  inferred role: {role}")
        print(f"  tag counts   : {dict(tag_counts)}")
        print(f"  phase count  : {len(phases)}")

        if role != expected["scan_role"]:
            failures.append(f"{name}: expected role {expected['scan_role']!r}, got {role!r}")
        if dict(tag_counts) != expected["tag_counts"]:
            failures.append(
                f"{name}: expected tag counts {expected['tag_counts']!r}, got {dict(tag_counts)!r}"
            )

    if failures:
        print("\nValidation FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nValidation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
