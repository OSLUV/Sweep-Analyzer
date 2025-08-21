import collections, struct, sys, array, json, base64, math, gzip, functools
from dataclasses import dataclass
from enum import Enum
from util import *

class Flags(Enum):
	IGNORE = 1<<0
	NO_SPECTRAL = 1<<1
	DIODE_CORR_MODE_0_AZ = 1<<2
	DIODE_CORR_MODE_1_NONE = 1<<3
	DIODE_CORR_MODE_2_SPECIAL = 1<<4
	SWEEP_START_WARMUP = 1<<6
	SWEEP_START_SCAN = 1<<7
	SWEEP_START_CUSTOM = 1<<8
	DISCARDED_DURING_SWEEP = 1<<9
	HIGHRES_MODE = 1<<10

@dataclass
class SpectrumCapture:
	integral_result: object # W/m^2
	spectral_result: object # list[W/m^2/nm]
	az_correction: float
	spectral_time_us: float
	integral_time_us: float
	spectral_saturation: float
	integral_saturation: float
	settings: object

	def get_spectral_point(self, scan, nm): # returns W/m^2/nm
		low = [(i, x) for i, x in enumerate(scan.spectral_wavelengths) if x <= nm][-1]
		hi = [(i, x) for i, x in enumerate(scan.spectral_wavelengths) if x >= nm][0]
		dist = hi[1] - low[1]
		if dist == 0:
			return self.spectral_result[low[0]]
		pos = (nm - low[1]) / dist

		low_v = self.spectral_result[low[0]]
		hi_v = self.spectral_result[hi[0]]

		return (low_v * (1-pos)) + (hi_v * pos)

@dataclass
class Coordinates:
	yaw_deg: float
	roll_deg: float
	lin_mm: float

	def scale_to_reference(self, value, reference=1):
		return value * ((self.lin_mm / (reference*1000)) ** 2)

	@classmethod
	def normalize_coordinates(cls, y, r):
		# Normalize from any random scheme to IES type-C standard coordinates
		# that is to say, 0 <= r <= 360 and 0 <= y <= 360 with POSITIVE ONLY yaw values

		if y < 0:
			y = -y
			r = r + 180

		r = r % 360

		return y, r

	def get_normal_yr(self):
		return self.normalize_coordinates(self.yaw_deg, self.roll_deg)

@dataclass
class GoniometerRow:
	coords: Coordinates
	capture: SpectrumCapture
	timestamp: float
	flags: int
	notes: str
	diag_notes: object

	@property
	def valid(self):
		return (not (self.flags & Flags.IGNORE.value)) and (not (self.flags & Flags.DISCARDED_DURING_SWEEP.value))
		# and (all(x>=-0.01 for x in self.capture.spectral_result))

	def compute_reference_integral(self, scan, wvl_range, reference_dist, force_integral=False): # Returns W/m^2 at reference_dist
		if force_integral or (self.flags & Flags.NO_SPECTRAL.value) or (not self.capture.spectral_result):
			if not math.isfinite(self.capture.integral_result):
				return 0
			return self.coords.scale_to_reference(self.capture.integral_result, reference_dist)
		return self.coords.scale_to_reference(scan.integrate_spectral(self.capture.spectral_result, *wvl_range), reference_dist)

class MajorAxis(Enum):
	NA = 0
	LINEAR = 1
	ROLL = 2 # 'normal'
	YAW = 3

class PhaseType(Enum):
	LINEAR_PULLBACK = 1
	SPECTRAL_WEB = 2
	INTEGRAL_WEB = 3
	WARMUP_WAIT = 4
	SPECTRUM_POINT = 5

	@property
	def is_web(self):
		return self == self.SPECTRAL_WEB or self == self.INTEGRAL_WEB

@dataclass
class GoniometerPhase:
	major_axis: MajorAxis
	phase_type: PhaseType
	name: str
	members: list[GoniometerRow]

@dataclass
class LampScan:
	lamp_name: str
	notes: str
	lamp_desc: object
	lamp_desc_filename: str

	spectral_wavelengths: list[float]

	spectral_units: str
	integral_units: str
	rows: list[GoniometerRow]
	phases: list[GoniometerPhase]
	
	_from_path: str = None
	_spectral_integral_cache: dict = None
	_wvl_index_cache: dict = None
	_bin_width_for_wvl_cache: dict = None

	def __post_init__(self):
		self._spectral_integral_cache = {}
		self._wvl_index_cache = {}
		self._bin_width_for_wvl_cache = {}

	@classmethod
	def make_spectral_wavelengths_from_legacy(cls, wvl_start, wvl_stop, wvl_step):
		r = []
		n = wvl_start
		while n <= wvl_stop:
			r.append(n)
			n += wvl_step
		return r

	def index_wvl(self, w):
		if w in self._wvl_index_cache: return self._wvl_index_cache[w]
		v = self.spectral_wavelengths.index([x for x in self.spectral_wavelengths if x>=w or x==self.spectral_wavelengths[-1]][0])
		self._wvl_index_cache[w] = v
		return v

	def get_bin_width(self, bin_num):
		# M's method
		if bin_num == len(self.spectral_wavelengths)-1:
			return 0
		else:
			return self.spectral_wavelengths[bin_num+1]-self.spectral_wavelengths[bin_num]


		if bin_num == 0:
			return self.spectral_wavelengths[1] - self.spectral_wavelengths[0]
		elif bin_num == len(self.spectral_wavelengths)-1:
			return self.spectral_wavelengths[bin_num] - self.spectral_wavelengths[bin_num-1]
		else:
			return ((self.spectral_wavelengths[bin_num] - self.spectral_wavelengths[bin_num-1])/2) + \
				   ((self.spectral_wavelengths[bin_num+1] - self.spectral_wavelengths[bin_num])/2)

	def get_bin_width_for_wvl(self, wvl):
		if wvl in self._bin_width_for_wvl_cache:
			return self._bin_width_for_wvl_cache[wvl]
		v = self.get_bin_width(self.spectral_wavelengths.index(wvl))
		self._bin_width_for_wvl_cache[wvl] = v
		return v

	def integrate_spectral(self, spectral, start=None, end=None, weighting=None): # returns W/m^2
		if type(spectral) is array.array:
			spectral_key = id(spectral)
		else:
			spectral_key = hash(tuple(spectral))
		cachekey = (spectral_key, start, end, weighting)
		cached = self._spectral_integral_cache.get(cachekey)
		if cached is not None:
			return cached

		if spectral is None: return 0

		if start is None: start = min(self.spectral_wavelengths)
		if end is None: end = max(self.spectral_wavelengths)

		istart = self.index_wvl(start)
		iend = self.index_wvl(end) or 1

		total = 0
		for wvl, value in zip(self.spectral_wavelengths[istart:iend], spectral[istart:iend]):
			v = value * self.get_bin_width_for_wvl(wvl)
			if weighting:
				v *= weighting(wvl)
			total += v

		self._spectral_integral_cache[cachekey] = total

		return total

	def get_best_value_in_band(self, row, start=None, end=None):
		if row.capture.spectral_result and len(row.capture.spectral_result)>3:
			return self.integrate_spectral(row.capture.spectral_result, start, end)
		elif row.capture.integral_result is not None:
			return row.capture.integral_result
		return 0

	def get_point_yr(self, yaw, roll, need = True):
		for row in self.rows:
			if not row.valid: continue

			if row.coords.yaw_deg == yaw and row.coords.roll_deg == roll:
				return row

		if need:
			raise IndexError("No Row")

	def get_point_yr_norm_all(self, yaw_, roll_, filter_valid=True):
		yaw, roll = Coordinates.normalize_coordinates(yaw_, roll_)
		res = []
		for row in self.rows:
			if filter_valid and not row.valid: continue

			ry, rr = row.coords.get_normal_yr()

			if ry == yaw and rr == roll:
				res.append(row)

			if row.coords.yaw_deg == yaw and row.coords.roll_deg == roll:
				res.append(row)

			if ry == 0 and yaw == 0:
				res.append(row) # assume rotational symmetry for 0

		return res

	def get_point_yr_norm(self, yaw_, roll_, if_missing = 'next_roll'):
		res = self.get_point_yr_norm_all(yaw_, roll_)

		if res:
			return res[0]

		if if_missing == 'next_roll':
			print(f"WARN: {self.lamp_name}: Missing point: ({yaw_}, {roll_}) -- taking next assuming rotational symmetry")
			return self.get_point_yr_norm(yaw_, roll_+22.5, 'die')
		else:
			raise IndexError(f"{self.lamp_name}: Failed to get_point_yr_norm({yaw_}, {roll_})")

	def get_point_ae(self, az, el, need = True):
		for row in self.rows:
			if not row.valid: continue

			if row.coords.roll_deg == 0 and row.coords.yaw_deg == az and el == 0:
				return row
			if row.coords.roll_deg == 90 and row.coords.yaw_deg == el and az == 0:
				return row
			if row.coords.yaw_deg == 0 and az == 0 and el == 0:
				return row

		if need:
			raise IndexError("No Row")

	def get_rolls(self):
		return set(x.coords.roll_deg for x in self.rows)

	def get_contiguous_arcs(self, roll, normalize=False, ignore_ignores=True):
		if normalize:
			roll = Coordinates.normalize_coordinates(0, roll)[1]
		arcs = []
		in_arc = False
		still_in_dT = None
		for row in self.rows:
			if row.flags & Flags.SWEEP_START_WARMUP.value:
				still_in_dT = True
			if row.flags & Flags.SWEEP_START_SCAN.value:
				still_in_dT = False

			if still_in_dT is not None and still_in_dT:
				continue

			if ignore_ignores and (row.flags & Flags.IGNORE.value):
				continue

			r = row.coords.roll_deg
			if normalize:
				r = Coordinates.normalize_coordinates(row.coords.yaw_deg, r)[1]
			
			if r == roll:
				if not in_arc:
					arcs.append([])
					in_arc = True

				arcs[-1].append(row)
			else:
				in_arc = False
		return arcs

	def get_webspec(self):
		yaws = list(set(row.coords.yaw_deg for row in self.rows))
		yaws.sort()
		rolls = list(set(row.coords.roll_deg for row in self.rows))
		rolls.sort()

		yawstep = yaws[1] - yaws[0]
		rollstep = rolls[1] - rolls[0]
		yaws = inclusive_range(0, 90, yawstep)
		rolls = inclusive_range(0, 360, rollstep)

		return (yaws, rolls)

def load_legacy_eegbin1(buffer, name, distance_mm, new_axes):
	header = '=dddidddd'
	header_sz = struct.calcsize(header)

	wvls = []

	def read(n):
		nonlocal buffer
		if len(buffer) < n:
			return b''
		r = buffer[:n]
		buffer = buffer[n:]
		return r

	expected_ws = None

	rows = []
	while 1:
		buf = read(header_sz)
		if len(buf) != header_sz:
			break
		az, el, li, length, wstart, wstop, wstep, integral_result = struct.unpack(header, buf)
		if expected_ws is None:
			expected_ws = (wstart, wstop, wstep)
		else:
			assert (wstart, wstop, wstep) == expected_ws, "Nonconstant spectral wavelength parameters"
		l = length*8
		buf = read(length*8)
		if len(buf) != l:
			break
		a = array.array('d', buf)

		if new_axes:
			yaw = az
			roll = el
		else:
			if az == 0:
				roll = 90
				yaw = el
			elif el == 0:
				roll = 0
				yaw = az
			else:
				assert False, "Non axial point"

		rows.append(GoniometerRow(
			Coordinates(yaw, roll, distance_mm),
			SpectrumCapture(integral_result, a, -1, -1, -1, -1, -1, None),
			0, 0, None, None))

	return LampScan(
		lamp_name = name,
		notes = "Imported from legacy eegbin1 file",
		lamp_desc = {},
		lamp_desc_filename = "?",
		spectral_wavelengths = LampScan.make_spectral_wavelengths_from_legacy(wstart, wstop, wstep),
		spectral_units = 'W/m2/nm',
		integral_units = 'W/m2',
		rows = rows,
		phases = []
	)


row_fmt_3 = '<dddddii'
row_fmt_4 = '<dddddidddi'
sep = b'\n\nB64 FOLLOWS\n\n'
#yaw, rol, lin, timestamp, integral, spec_settings key (unused), az correction, spec time, integ time, flags
def save_eegbin2(scan):
	buffer = b''

	metadata = {
		"version": 5,
		"scan": {
			k: v for k, v in scan.__dict__.items() if k not in ['rows', 'phases'] and not k.startswith("_")
		},
		"row_notes": {
			str(i): k.notes for i, k in enumerate(scan.rows) if k.notes
		}
	}
	buffer += json.dumps(metadata, indent=4).encode('utf-8')
	buffer += sep

	# print(f"# sw: {len(scan.spectral_wavelengths)}")

	rows_buf = b''
	for row in scan.rows:
		rows_buf += struct.pack(
			row_fmt_4, row.coords.yaw_deg, row.coords.roll_deg,
			row.coords.lin_mm, row.timestamp, row.capture.integral_result, 0,
			row.capture.az_correction, row.capture.spectral_time_us, row.capture.integral_time_us, row.flags)

		if not (row.flags & Flags.NO_SPECTRAL.value):
			rows_buf += bytes(array.array('d', row.capture.spectral_result))

	buffer += base64.standard_b64encode(rows_buf)

	return buffer

def load_eegbin2(buffer, from_path=None):
	metadata, rows_buf = buffer.split(sep, 1)
	metadata = json.loads(metadata)
	assert metadata['version'] in [3, 4, 5], "Unknown version"
	if metadata['version'] < 5 and 'spectral_wavelengths' not in metadata['scan']:
		metadata['scan']['spectral_wavelengths'] = LampScan.make_spectral_wavelengths_from_legacy(metadata['scan']['wvl_start'], metadata['scan']['wvl_stop'], metadata['scan']['wvl_step'])
		del metadata['scan']['wvl_start']
		del metadata['scan']['wvl_stop']
		del metadata['scan']['wvl_step']

	rows_buf = base64.standard_b64decode(rows_buf)

	scan = LampScan(**metadata['scan'], rows = [], phases=[], _from_path=from_path)

	# print(f"# sw: {len(scan.spectral_wavelengths)}")

	row_points_len = 8 * len(scan.spectral_wavelengths)
	
	rows_buf_ptr = 0
	def take(n):
		nonlocal rows_buf, rows_buf_ptr
		r = rows_buf[rows_buf_ptr:rows_buf_ptr+n]
		rows_buf_ptr += n
		return r

	i = 0
	while True:
		if rows_buf_ptr == len(rows_buf):
			break

		try:
			if metadata['version'] == 3:
				yaw_deg, roll_deg, lin_mm, timestamp, integral_result, _, flags = struct.unpack(row_fmt_3, take(struct.calcsize(row_fmt_3)))
				az_correction, spectral_time_us, integral_time_us = -1, -1, -1
			else:
				yaw_deg, roll_deg, lin_mm, timestamp, integral_result, _, az_correction, spectral_time_us, integral_time_us, flags = struct.unpack(row_fmt_4, take(struct.calcsize(row_fmt_4)))


			if not (flags & Flags.NO_SPECTRAL.value):
				points = array.array('d', take(row_points_len))
			else:
				points = None

			notes = metadata['row_notes'].get(str(i))
			scan.rows.append(GoniometerRow(Coordinates(yaw_deg, roll_deg, lin_mm), SpectrumCapture(integral_result, points, az_correction, spectral_time_us, integral_time_us, -1, -1, None), timestamp, flags, notes, None))
			i += 1
			if (i%1000) == 0:
				sys.stderr.write(f"Loaded {i} rows...\n")

			# print(f"{(yaw_deg, roll_deg, lin_mm)} -> {integral_result}")
		except struct.error:
			sys.stderr.write("E: Failed to unpack as expected\n")
			break

	return scan
	
eegbin3_magic = 'EEGBIN3\n'.encode('ascii')
eegbin3_sep = '\n\nGZIP FOLLOWS\n\n'.encode('ascii')
def save_eegbin3(scan, fd):
	def row_to_json(row):
		return {
			'yaw_deg': row.coords.yaw_deg,
			'roll_deg': row.coords.roll_deg,
			'lin_mm': row.coords.lin_mm,
			'timestamp': row.timestamp,
			'integral_result': row.capture.integral_result,
			'az_correction': row.capture.az_correction,
			'spectral_time_us': row.capture.spectral_time_us,
			'integral_time_us': row.capture.integral_time_us,
			'spectral_saturation': row.capture.spectral_saturation,
			'integral_saturation': row.capture.integral_saturation,
			'notes': row.notes,
			'diag_notes': row.diag_notes,
			'flags': [
				x.name for x in Flags if (row.flags & x.value)
			]
		}

	def phase_to_json(phase):
		return {
			'major_axis': phase.major_axis.name,
			'phase_type': phase.phase_type.name,
			'name': phase.name,
			'members': [
				scan.rows.index(r) for r in phase.members
			]
		}

	print("Building metadata...")

	buffer = eegbin3_magic
	fd.write(buffer)

	metadata = {
		"version": 6,
		"scan": {
			k: v for k, v in scan.__dict__.items() if k not in ['rows', 'phases'] and not k.startswith("_")
		},
		"rows": [
			row_to_json(r) for r in scan.rows
		],
		"phases": [
			phase_to_json(a) for a in scan.phases
		],
		"is_gzipped": False
	}

	print("json.dumps...")

	buffer = json.dumps(metadata, indent=None).encode('utf-8')

	fd.write(buffer)

	# buffer += eegbin3_sep

	fd.write(eegbin3_sep)

	print("Building array data...")

	rows_buf = b''
	for row in scan.rows:
		if not (row.flags & Flags.NO_SPECTRAL.value):
			# rows_buf += bytes(array.array('d', row.capture.spectral_result))
			fd.write(bytes(array.array('d', row.capture.spectral_result)))

	# print("Compressing...")
	# buffer += gzip.compress(rows_buf, 0)
	print("Done.")

	return buffer

def load_eegbin3(buffer, from_path=None):
	assert buffer.startswith(eegbin3_magic), "Bad magic"
	buffer = buffer[len(eegbin3_magic):]
	metadata, rows_buf = buffer.split(eegbin3_sep, 1)
	metadata = json.loads(metadata)
	assert metadata['version'] == 6, "Bad version"

	if metadata.get('is_gzipped', True):
		rows_buf = gzip.decompress(rows_buf)

	scan = LampScan(**metadata['scan'], rows = [], phases=[], _from_path=from_path)

	row_points_len = 8 * len(scan.spectral_wavelengths)
	
	rows_buf_ptr = 0
	def take(n):
		nonlocal rows_buf, rows_buf_ptr
		r = rows_buf[rows_buf_ptr:rows_buf_ptr+n]
		rows_buf_ptr += n
		return r

	for row in metadata['rows']:
		flags = sum(Flags[x].value for x in row['flags'])

		points = None
		if not (flags & Flags.NO_SPECTRAL.value):
			points = array.array('d', take(row_points_len))

		scan.rows.append(
			GoniometerRow(
				Coordinates(row['yaw_deg'], row['roll_deg'], row['lin_mm']),
				SpectrumCapture(
					row['integral_result'], points, row['az_correction'], row['spectral_time_us'], row['integral_time_us'],
					row.get('spectral_saturation', -1), row.get('integral_saturation', -1), None),
				row['timestamp'], flags, row['notes'], row.get('diag_notes', None)
			))

	for phase in metadata['phases']:
		scan.phases.append(
			GoniometerPhase(MajorAxis[phase['major_axis']], PhaseType[phase['phase_type']], phase['name'],
				[scan.rows[i] for i in phase['members']]))

	return scan
