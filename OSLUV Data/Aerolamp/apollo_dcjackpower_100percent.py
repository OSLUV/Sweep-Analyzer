scan = load_scans_path('Apollo/Apollo-DCPOWER-Seasoned-1m1749948341.sw3')
scan2 = load_scans_path('Apollo/apollo_dcjackpower_100percent_1749243198.sw3')



report = Report(
	reporting_name="Aerolamp DevKit",
	reporting_subtitle="(Aerolamp DevKit V1.0 - DC Jack Power - 100 Percent Output",
	catalog_id="27c4ca60-0da7-4669-8cc7-36746a06fb62",
	revision_history = [
		Revision("R00", "June 11, 2025", "Initial Version")
	],
	scan = scan,
	spectrum_scan = scan,
	startup_scan = scan,
	burnin_scan = scan2,
	supply_style = "12V AC/DC ",
	wall_power_type = "120 VAC",
	wall_power_W = 13,
	consumer_cost = "$600.00 USD",

	acq_name = "Aerolamp V1.0 Dev Kit ",
	acq_manufacturer = "Apollo Airtech",
	acq_model_number = "DevKit V1.0 - Dimmable",
	acq_serial_number = None,
	acq_production_date = None,
	osluv_serial_number = None ,
	acq_date = "June, 2025",
	acq_price = "Complementary",
	acq_method = "Donated",

	acq_picture_path = "assets/aerolamp_dev.png",
	axes_ref_picture_path = "assets/aerolamp_dev_axes.png",

	power_note = None ,
	power_data = load_power_data_from_prefix('Apollo/apollodev'),
	psu_desc = 'MeanWell 12 VDC Power Supply',
	psu_manuf = "MeanWell",
	psu_model = "NGE30U12-P1J",
	psu_serial = "SC491L6482",
	# psu_osluv_serial = None,

	webtool_meta = {
		"preview_setup": {
			"x": 0.1,
			"y": 2,
			"z": 2.6,
			"tilt": -45,
			"orientation": 0,
			"rotation": 0
		}
	}
)
