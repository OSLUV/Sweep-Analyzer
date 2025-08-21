#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from imgui.integrations.pygame import PygameRenderer
import OpenGL.GL as gl
import imgui
import pygame
import sys
import os
import time
from array import array
import math
import traceback

import time
import matplotlib.pyplot as plt
import numpy as np

import integrate_top
import ies_to_sweep

import eegbin
from util import *

def open_cmdline_files(app):
	for item in sys.argv[1:]:
		if item.startswith('-'):
			print("Unknown parameter: " + item)
			continue

		if '.eegbin1' in item:
			old_axes = '/OLDAXES' in item
			item = item.replace('/OLDAXES', '')
			with open(item, 'rb') as fd:
				scan = eegbin.load_legacy_eegbin1(fd.read(), item, 1000, not old_axes)
		elif '.sweep' in item:
			with open(item, 'rb') as fd:
				scan = eegbin.load_eegbin2(fd.read())
		elif '.sw3' in item:
			with open(item, 'rb') as fd:
				scan = eegbin.load_eegbin3(fd.read())
		elif '.ies' in item:
			scan = ies_to_sweep.load_ies(item)
		else:
			raise ValueError("unknown file format/extension: " + item)

		app.open_files.append(SweepFileEditor(item, scan))
		# app.open_files[-1].draw_scan(app)

class SweepFileEditor:
	def __init__(self, fn, scan):
		self.fn = fn
		self.scan = scan
		self.selected_rows = []
		self.redraw = False
		self._top = None
		self._peak = None
		self.using_integral = False
		self.show_only_selected = False

	def compute_reference_integral(self, app, row, force_integral=False):
		if self.using_integral or (row.flags & eegbin.Flags.NO_SPECTRAL.value) or (not row.capture.spectral_result):
			if not math.isfinite(row.capture.integral_result):
				return 0
			return row.coords.scale_to_reference(row.capture.integral_result, app.settings.ref_distance)
		return row.coords.scale_to_reference(self.scan.integrate_spectral(row.capture.spectral_result, *app.settings.wvl_range), app.settings.ref_distance)

	def do_draw(self, app):
		if not imgui.begin(f"Sweep {self.scan.lamp_name} ({self.fn})###{id(self)}"):
			return
		imgui.text(f"Lamp Name:")
		imgui.same_line()
		_, self.scan.lamp_name = imgui.input_text('##ln', self.scan.lamp_name)
		imgui.text(f"Notes:")
		imgui.same_line()
		_, self.scan.notes = imgui.input_text('##no', self.scan.notes)

		changed, self.using_integral = imgui.checkbox("Force Integral?", self.using_integral)
		if changed:
			self._top = None
			self._peak = None
			self.redraw = True

		_, self.show_only_selected = imgui.checkbox("Show only selected?", self.show_only_selected)

		if imgui.tree_node("Parameters"):
			imgui.text(f"Wavelength Start: {min(self.scan.spectral_wavelengths)} nm")
			imgui.text(f"Wavelength Stop: {max(self.scan.spectral_wavelengths)} nm")
			imgui.text(f"Integral Units: {self.scan.integral_units}")
			imgui.text(f"Spectral Units: {self.scan.spectral_units}")
			imgui.text(f"Lamp Desc: <{repr(self.scan.lamp_desc)}> from '{self.scan.lamp_desc_filename}'")
			imgui.tree_pop()
		
		if imgui.tree_node("Stats", imgui.TREE_NODE_DEFAULT_OPEN):
			if self.scan.rows:
				if not self._peak:
					rows_and_refs = [(row, self.compute_reference_integral(app, row)) for row in self.scan.rows if row.valid]
					rows_and_refs.sort(key=lambda x:x[1])
					self._peak = rows_and_refs[-1]

				top_row, top_val = self._peak

				imgui.text(f"Peak: {top_val} {self.scan.integral_units} @ {app.settings.ref_distance}m")
				imgui.text(f"Peak Position: #{self.scan.rows.index(top_row)}, Yaw {top_row.coords.yaw_deg} / Roll {top_row.coords.roll_deg} / Lin {top_row.coords.lin_mm}")
				
				if not self._top:
					try:
						normed = integrate_top.normalize_sweep(self.scan)
						self._top = integrate_top.compute_top(normed, app.settings.wvl_range, force_integral=self.using_integral)*1000
					except Exception as e:
						traceback.print_exc()
						self._top = repr(e)
				imgui.text(f"TOP: {self._top} mW")
			else:
				imgui.text_disabled("(no rows)")

			imgui.tree_pop()

		imgui.text("Filename:")
		imgui.same_line()
		_, self.fn = imgui.input_text('##fn', self.fn)
		imgui.same_line()
		if imgui.button("Save") and (('.sweep' in self.fn) or ('.sw3' in self.fn)):
			self.save()

		if imgui.button("Close"):
			self.save()
			imgui.end()
			return True

		def mk_row_desc(row, i):
			flags_str = ""
				
			for k, v in eegbin.Flags.__members__.items():
				if row.flags & v.value:
					flags_str += f"[{k}] "

			if row.notes:
				flags_str += "[NOTE] "

			return f"{i:3}: Yaw {row.coords.yaw_deg:+3.0f} / Roll {row.coords.roll_deg:+3.0f} / Lin {row.coords.lin_mm:4} {flags_str}###{i}{row.coords}"

		if imgui.tree_node(f"{len(self.scan.phases)} Phases###Phases", imgui.TREE_NODE_DEFAULT_OPEN):
			for i, phase in enumerate(self.scan.phases):
				if imgui.tree_node(f"{i}: '{phase.name} ({phase.phase_type.name} {phase.major_axis.name}-major) -- {len(phase.members)} members"):
					imgui.text("No detail")
					imgui.tree_pop()

			imgui.tree_pop()

		if imgui.tree_node(f"Select from: {len(self.scan.rows)} Rows###Rows"):
			for i, row in enumerate(self.scan.rows):
				_, v = imgui.checkbox(mk_row_desc(row, i), row in self.selected_rows)
				if v and row not in self.selected_rows:
					self.selected_rows.append(row)
				elif not v and row in self.selected_rows:
					self.selected_rows.remove(row)

			imgui.tree_pop()
		
		if imgui.tree_node(f"Inspect from: {len(self.selected_rows)} Selected Rows###Selected", imgui.TREE_NODE_DEFAULT_OPEN):
			for i, row in enumerate(self.selected_rows):
				if imgui.button('x'):
					self.selected_rows.remove(row)

				imgui.same_line()

				if imgui.tree_node(mk_row_desc(row, i), imgui.TREE_NODE_FRAMED | imgui.TREE_NODE_DEFAULT_OPEN):
					imgui.text(f"Yaw: {row.coords.yaw_deg} deg / Roll: {row.coords.roll_deg} deg / Lin {row.coords.lin_mm} mm")
					imgui.text(f"Timestamp: {row.timestamp}")
					imgui.text(f"a(z) correction factor: {row.capture.az_correction}")
					imgui.text(f"Integration time: {row.capture.spectral_time_us} us Spectral / {row.capture.integral_time_us} us Integral")

					imgui.text(f"(RAW) Integral Sensor: {row.capture.integral_result} {self.scan.integral_units} / (RAW) Spectral Sensor (in-band): {self.scan.integrate_spectral(row.capture.spectral_result, *app.settings.wvl_range) if row.capture.spectral_result else 'x'}")
					imgui.text(f"Integral: {self.compute_reference_integral(app, row)} {self.scan.integral_units} @ {app.settings.ref_distance}m")

					if row.capture.spectral_result:
						max_v = max(row.capture.spectral_result)
						max_w = self.scan.spectral_wavelengths[row.capture.spectral_result.index(max_v)]
						imgui.text(f"(RAW) Spectral Peak: {max_v} {self.scan.spectral_units} @ {max_w} nm")

					imgui.text(f"Integral sat: {row.capture.integral_saturation:.1f}%, Spectral sat: {row.capture.spectral_saturation:.1f}%")

					imgui.text("Flags:")
					imgui.same_line()
					for k, v in eegbin.Flags.__members__.items():
						s = row.flags & v.value
						c, s2 = imgui.checkbox(k, s)
						if s != s2 and c:
							row.flags ^= v.value
							print(row.flags)
							self.redraw = True
						imgui.same_line()
					imgui.text("")

					imgui.text("Notes:")
					imgui.same_line()
					_, v = imgui.input_text('##x', row.notes or '')
					row.notes = v or None

					imgui.text(f"Diag notes: {str(row.diag_notes)}")

					if imgui.button("Delete"):
						self.scan.rows.remove(row)
						self.selected_rows.remove(row)
						self.redraw = True

					imgui.tree_pop()
			imgui.tree_pop()

		imgui.end()

	def save(self):
		sip_fn = self.fn+".save-in-progress"
		old_fn = self.fn+".old"

		with open(sip_fn, 'wb') as fd:
			# fd.write(eegbin.save_eegbin3(self.scan))
			eegbin.save_eegbin3(self.scan, fd)

		if os.stat(sip_fn).st_size == 0:
			print("FAILED TO SAVE")
			return

		try:
			os.rename(self.fn, old_fn)
		except FileNotFoundError:
			pass

		os.rename(sip_fn, self.fn)

		try:
			os.unlink(old_fn)
		except FileNotFoundError:
			pass
