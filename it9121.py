import socket, time, sys
from math import sqrt

class IT9121:
	def __init__(self, host):
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.connect((host, 30000))
		assert self.command_line_response("*IDN?").startswith('ITECH Ltd., IT9121,')

	def command_no_response(self, command):
		# print(f'[debug] send {repr(command)}')
		self.socket.sendall(command.encode('ascii') + b'\n')
		time.sleep(0.05)

	def command_line_response(self, command):
		self.command_no_response(command)

		r = b''
		while not r.endswith(b'\n'):
			r += self.socket.recv(1)

		r = r.decode('ascii')[:-1]

		# print(f'[debug] recv {repr(r)}')

		return r

	def command_int_response(self, command):
		return int(self.command_line_response(command).strip())

	def command_float_response(self, command):
		return float(self.command_line_response(command).strip())

	def command_harmonic_array_response(self, command):
		return [float(x) for x in self.command_line_response(command).strip().split(' ')]

	def configure_default(self):
		self.command_no_response('SYSTEM:REMOTE')
		self.command_no_response('RATE 0.1')
		self.command_no_response('INPUT:FILTER:FREQUENCY ON')
		self.command_no_response('INPUT:FILTER:LINE ON')

		# THD-F is referred to fundamental
		# THD-R is referred to RMS value

	def get_voltage_rms(self):
		return self.command_float_response('FETCH:VOLTAGE:RMS?')

	def get_current_rms(self):
		return self.command_float_response('FETCH:CURRENT:RMS?')

	def get_voltage_pp(self):
		return self.command_float_response('FETCH:VOLTAGE:PPEAK?')

	def get_current_pp(self):
		return self.command_float_response('FETCH:CURRENT:PPEAK?')

	def get_power_active(self):
		return self.command_float_response('FETCH:POWER:ACTIVE?')

	def get_power_reactive(self):
		return self.command_float_response('FETCH:POWER:REACTIVE?')

	def get_power_apparent(self):
		return self.command_float_response('FETCH:POWER:APPARENT?')

	def get_power_factor(self):
		return self.command_float_response('FETCH:POWER:PFACTOR?')

	def compute_thd_f(self, harmonics):
		harmonics = harmonics[1:]
		if harmonics[0] == 0:
			return 0
		return sqrt(sum(x**2 for x in harmonics[1:])) / harmonics[0]

	def get_voltage_thd_f(self):
		return self.compute_thd_f(self.command_harmonic_array_response("FETCH:HARM:VOLT:AMPL?"))

	def get_current_thd_f(self):
		return self.compute_thd_f(self.command_harmonic_array_response("FETCH:HARM:CURR:AMPL?"))

	def get_voltage_freq(self):
		return self.command_float_response('FETCH:FREQUENCY:VOLTAGE?')

	def get_current_freq(self):
		return self.command_float_response('FETCH:FREQUENCY:CURRENT?')

	# def get_active_power_thd_f(self):
	#     return self.compute_thd_f(self.command_harmonic_array_response("FETCH:HARM:POWER:ACTIVE:AMPL?"))

	# def get_reactive_power_thd_f(self):
	#     return self.compute_thd_f(self.command_harmonic_array_response("FETCH:HARM:POWER:REACTIVE?"))

	# def get_apparent_power_thd_f(self):
	#     return self.compute_thd_f(self.command_harmonic_array_response("FETCH:HARM:POWER:APPARENT?"))

	def get_single_waveform(self):
		time.sleep(0.1)
		i.command_no_response("WAVE:SINGLE")

		resp = ''
		while resp != "STOP":
			print(f"(Spin '{resp}')")
			time.sleep(0.1)
			resp = i.command_line_response("WAVE:TRIG?").upper()

		wave_v = [int(x) for x in i.command_line_response("WAVE:VOLT:DATA:NORM?").split(' ') if x]
		wave_i = [int(x) for x in i.command_line_response("WAVE:CURR:DATA:NORM?").split(' ') if x]

		v_pospk = i.command_float_response("FETCH:VOLTAGE:MAXPK?")
		v_negpk = i.command_float_response("FETCH:VOLTAGE:MINPK?")
		i_pospk = i.command_float_response("FETCH:CURRENT:MAXPK?")
		i_negpk = i.command_float_response("FETCH:CURRENT:MINPK?")

		# max val is 127
		# but the following approach does NOT work
		# wave_v_scale = (v_range*3)/1.
		# wave_i_scale = (i_range*3)/1.

		try:
			wave_v_scale_pos = v_pospk / max(wave_v)
			wave_v_scale_neg = v_negpk / min(wave_v)
			wave_v_scale = (wave_v_scale_pos + wave_v_scale_neg) / 2
		except ZeroDivisionError:
			wave_v_scale = 0

		try:
			wave_i_scale_pos = i_pospk / max(wave_i)
			wave_i_scale_neg = i_negpk / min(wave_i)
			wave_i_scale = (wave_i_scale_pos + wave_i_scale_neg) / 2
		except ZeroDivisionError:
			wave_i_scale = 0

		wave_v = [x*wave_v_scale for x in wave_v]
		wave_i = [x*wave_i_scale for x in wave_i]

		div_t = i.command_float_response("WAVE:TRIG:DIVT?")
		total_t = div_t * 12

		assert len(wave_v) == len(wave_i)
		t_step = total_t / len(wave_v)
		wave_t = [x * t_step for x in range(len(wave_v))]

		# print("V_pp", max(wave_v)-min(wave_v))
		# print("I_pp", max(wave_i)-min(wave_i))
		# print("I_pk+", max(wave_i))

		time.sleep(0.1)
		
		return (wave_t, wave_v, wave_i)

def test_plot(i):
	from matplotlib import pyplot as plt
	import matplotlib.animation as animation
	
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	twinx = True

	if twinx:
		ax2 = ax.twinx()
	else:
		ax2 = ax

	i.command_no_response("WAVE:TRIG:DELAY:TIME -0.00333333")
	i.command_no_response("WAVE:TRIG:DIVT 0.005")
	w_t, w_v, w_i = i.get_single_waveform()
	ax.plot(w_t, w_v, color='red')
	ax2.plot(w_t, w_i, color='blue')
	plt.show()

if __name__ == '__main__':
	i = IT9121("10.0.1.249")
	i.configure_default()

	work_prefix = sys.argv[1]

	log_fd = open(work_prefix+"_log.csv", 'a')

	log_fd.write("Timestamp,V_RMS,I_RMS,W_Apparent,W_Active,W_Reactive,V_THD-F,I_THD-F,PFactor,V_Freq\n")

	i.command_no_response('RATE 1')


	while 1:
		# i.command_no_response('WAVE:RUN')
		# i.command_line_response("MEASURE:VOLTAGE:AC?")
		time.sleep(0.1)
		for _ in range(10):
			time.sleep(1)
			row = ",".join(map(str, [
				time.time(),
				i.get_voltage_rms(), i.get_current_rms(),
				i.get_power_apparent(), i.get_power_active(), i.get_power_reactive(),
				i.get_voltage_thd_f(), i.get_current_thd_f(), i.get_power_factor(), i.get_voltage_freq()
			]))

			print(row)
			log_fd.write(row + '\n')
			log_fd.flush()

		# print("Perform waveform capture...")

		# w_t, w_v, w_i = i.get_single_waveform()

		# with open(work_prefix+'_wfm_'+str(time.time())+".csv", 'w') as fd:
		# 	fd.write("Time,V,I\n")
		# 	for t, v, i_ in zip(w_t, w_v, w_i):
		# 		fd.write(",".join(map(str, [
		# 			t, v, i_
		# 		])))
		# 		fd.write('\n')
		# 		fd.flush()
