# Changelog

## v2.0.0 (2026-02-08)
- Added a top-level `Report Builder` tab with dedicated `Inputs` and `Phases & Segments` views.
- Added report SW3 phase auto-tagging and scan-role inference (`Complete Dataset`, `Main Scan`, `Burn-in Scan`, etc.).
- Added duplicate segment selection UI for warm-up, spectrum point, R2 pullback, loose web, and tight web.
- Added warm-up auto-selection logic that prefers the latest approximately 1-hour warm-up segment.
- Added report CSV role management (`Power Log`, `Power Waveform`, `Power Factor/Current`).
- Added report image asset inputs for lamp photo and axes photo.
- Added Aerolamp fixture validator script: `scripts/validate_aerolamp_fixture.py`.
- Added report-tagging regression tests in `tests/test_report_phase_tagging.py`.
- Added session metadata fields: `app_version` and `session_schema_version`.
