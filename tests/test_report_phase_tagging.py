import unittest

from gui import (
    choose_latest_full_hour_warmup_segment_id,
    report_segment_key_for_phase,
    suggest_report_phase_tag,
    suggest_report_scan_role,
)


class TestReportPhaseTagging(unittest.TestCase):
    def test_phase_name_and_type_mapping(self):
        cases = [
            (
                {"name": "Warm-up/dT h99", "phase_type": "WARMUP_WAIT", "axis": "NA"},
                "Warm-up",
            ),
            (
                {"name": "Spectrum Point", "phase_type": "SPECTRUM_POINT", "axis": "NA"},
                "Spectrum Point",
            ),
            (
                {"name": "r2 pullback (both)", "phase_type": "LINEAR_PULLBACK", "axis": "LINEAR"},
                "R2 Pullback",
            ),
            (
                {"name": "Loose Spectrometer Scan", "phase_type": "SPECTRAL_WEB", "axis": "ROLL"},
                "Loose Spectrometer Scan",
            ),
            (
                {"name": "Tight Spectrometer Scan", "phase_type": "SPECTRAL_WEB", "axis": "ROLL"},
                "Tight Spectrometer Scan",
            ),
        ]
        for phase, expected in cases:
            with self.subTest(phase=phase):
                self.assertEqual(suggest_report_phase_tag(phase), expected)

    def test_phase_tag_ignore_and_burnin(self):
        self.assertEqual(
            suggest_report_phase_tag({"name": "discard this setup phase", "phase_type": "", "axis": ""}),
            "Ignore",
        )
        self.assertEqual(
            suggest_report_phase_tag({"name": "1000hr burnin segment", "phase_type": "", "axis": ""}),
            "Burn-in",
        )

    def test_role_inference_main_scan(self):
        phases = [
            {"name": "Warm-up/dT h99", "phase_type": "WARMUP_WAIT", "start_ts": 0.0, "end_ts": 3600.0},
            {"name": "Spectrum Point", "phase_type": "SPECTRUM_POINT", "start_ts": 3600.0, "end_ts": 3650.0},
            {"name": "r2 pullback (both)", "phase_type": "LINEAR_PULLBACK", "start_ts": 3650.0, "end_ts": 7200.0},
            {"name": "Loose Spectrometer Scan", "phase_type": "SPECTRAL_WEB", "start_ts": 7200.0, "end_ts": 7500.0},
            {"name": "Tight Spectrometer Scan", "phase_type": "SPECTRAL_WEB", "start_ts": 7500.0, "end_ts": 8600.0},
        ]
        self.assertEqual(suggest_report_scan_role(phases), "Main Scan")

    def test_role_inference_complete_dataset(self):
        phases = []
        for i in range(80):
            start = i * 3600.0
            end = (i + 1) * 3600.0
            phases.append(
                {
                    "name": f"Warm-up/dT h{i + 20}",
                    "phase_type": "WARMUP_WAIT",
                    "start_ts": start,
                    "end_ts": end,
                }
            )
        phases.extend(
            [
                {"name": "Spectrum Point", "phase_type": "SPECTRUM_POINT", "start_ts": 80 * 3600.0, "end_ts": 80 * 3600.0 + 60.0},
                {"name": "r2 pullback (both)", "phase_type": "LINEAR_PULLBACK", "start_ts": 80 * 3600.0 + 60.0, "end_ts": 81 * 3600.0},
                {"name": "Loose Spectrometer Scan", "phase_type": "SPECTRAL_WEB", "start_ts": 81 * 3600.0, "end_ts": 82 * 3600.0},
                {"name": "Tight Spectrometer Scan", "phase_type": "SPECTRAL_WEB", "start_ts": 82 * 3600.0, "end_ts": 83 * 3600.0},
            ]
        )
        self.assertEqual(suggest_report_scan_role(phases), "Complete Dataset")

    def test_role_inference_burnin_scan(self):
        phases = [
            {"name": "Burn-in Day 5", "phase_type": "WARMUP_WAIT", "start_ts": 0.0, "end_ts": 86400.0},
            {"name": "Spectrum Point", "phase_type": "SPECTRUM_POINT", "start_ts": 86500.0, "end_ts": 86520.0},
        ]
        self.assertEqual(suggest_report_scan_role(phases), "Burn-in Scan")

    def test_segment_key_mapping(self):
        self.assertEqual(
            report_segment_key_for_phase(
                {"name": "Loose Spectrometer Scan", "phase_type": "SPECTRAL_WEB"},
                tag="Loose Spectrometer Scan",
            ),
            "loose_web",
        )
        self.assertEqual(
            report_segment_key_for_phase(
                {"name": "Tight Spectrometer Scan", "phase_type": "SPECTRAL_WEB"},
                tag="Spectral Web",
            ),
            "tight_web",
        )
        self.assertIsNone(
            report_segment_key_for_phase(
                {"name": "Notes", "phase_type": "MANUAL"},
                tag="Ignore",
            )
        )

    def test_warmup_selector_prefers_latest_full_hour(self):
        candidates = [
            {"segment_id": "A:10", "start_ts": 10.0, "phase_idx": 10, "duration_h": 1.8},
            {"segment_id": "A:11", "start_ts": 20.0, "phase_idx": 11, "duration_h": 0.99},
            {"segment_id": "A:12", "start_ts": 30.0, "phase_idx": 12, "duration_h": 1.01},
        ]
        self.assertEqual(choose_latest_full_hour_warmup_segment_id(candidates), "A:12")

    def test_warmup_selector_falls_back_to_latest_when_no_full_hour(self):
        candidates = [
            {"segment_id": "B:1", "start_ts": 100.0, "phase_idx": 1, "duration_h": 0.4},
            {"segment_id": "B:2", "start_ts": 120.0, "phase_idx": 2, "duration_h": 0.6},
        ]
        self.assertEqual(choose_latest_full_hour_warmup_segment_id(candidates), "B:2")

    def test_warmup_selector_handles_empty_list(self):
        self.assertIsNone(choose_latest_full_hour_warmup_segment_id([]))


if __name__ == "__main__":
    unittest.main()
