# Changelog

## v3.0.0 (2026-02-08)
- Expanded `Report Builder` into four dedicated tabs: `Inputs`, `Phases & Segments`, `Metadata`, and `Preview`.
- Added full analysis preview suite in `Preview`: pattern, warm-up, burn-in, spectral, exposure, roll, R2, and electrical views.
- Added robust duplicate-segment workflows, including explicit segment selectors and faster custom pattern-plane slider rendering via precomputed caches.
- Added spectral report enhancements:
  - wavelength-band optical power integration presets (`200-230`, `200-300`) and band/full-spectrum mode,
  - waveband power table with totals and percentages,
  - linear + log spectral overlays for raw and weighted curves,
  - popout windows for table/linear/log spectral views.
- Added exposure workflow improvements:
  - default weighting mode set to `IEC`,
  - fail-safe distance marker rendering and cleaner distance-curve presentation.
- Vendored IES/IEC weighting source tables into the repository under `reporting_data/` to remove runtime dependency on external OpenGoniometer code paths.
- Added regression coverage for spectral weighting interpolation and vendored weighting table presence:
  - `tests/test_weighting_interpolation.py`
  - `tests/test_vendored_weight_tables.py`
- Added end-to-end Aerolamp fixture and report correctness support updates.

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
