import imgui

def do_editable_raw(preamble, value, units="", width=100):
	imgui.text(preamble)
	imgui.same_line()
	imgui.push_item_width(width)
	imgui.push_id(preamble)
	c, v = imgui.input_text(units, str(value))
	imgui.pop_id()
	imgui.pop_item_width()

	return (c, v)

def do_editable(preamble, value, units="", width=100, enable=True):
	if not enable:
		imgui.text_disabled(f"{preamble} {value} {units}")
		return value

	type_ = type(value)

	_, v = do_editable_raw(preamble, value, units, width)

	if v != value:
		try:
			v = type_(v)
		except:
			return value

		return v
	else:
		return value

def inclusive_range(start, stop, step):
	if step == 0: return []
	o = []
	while start <= stop:
		o.append(start)
		start += step
	return o

