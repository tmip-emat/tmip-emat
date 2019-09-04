
import webcolors
import re

DEFAULT_BASE_COLOR = 'rgb(31, 119, 180)'
DEFAULT_HIGHLIGHT_COLOR = 'rgb(255, 127, 14)'
DEFAULT_BOX_BG_COLOR = '#2ca02c'

color_names = {
	'#000000':'Black'	,
	'#FFFFFF':'White'	,
	'#FF0000':'Red'		,
	'#00FF00':'Lime'	,
	'#0000FF':'Blue'	,
	'#FFFF00':'Yellow'	,
	'#00FFFF':'Cyan' 	,
	'#FF00FF':'Magenta'	,
	'#C0C0C0':'Silver'	,
	'#808080':'Gray'	,
	'#800000':'Maroon'	,
	'#808000':'Olive'	,
	'#008000':'Green'	,
	'#800080':'Purple'	,
	'#008080':'Teal'	,
	'#000080':'Navy'	,
	'#A52A2A':'Brown'	,
	'#FFA500':'Orange'	,
	'#1F77B4':'Blue'    ,
	'#FF7F0E':'Orange'  ,
}

def interpret_color(colorstring):
	if isinstance(colorstring, str):
		rgb_parse = re.compile(
			"^(rgb)?\(?([01]?\d\d?|2[0-4]\d|25[0-5])"
			"(\W+)([01]?\d\d?|2[0-4]\d|25[0-5])"
			"\W+(([01]?\d\d?|2[0-4]\d|25[0-5])\)?)$"
		)
		try:
			cci = rgb_parse.findall(colorstring)[0]
			return webcolors.IntegerRGB(int(cci[1]), int(cci[3]), int(cci[5]))
		except:
			return webcolors.hex_to_rgb(colorstring)
	else:
		return colorstring

def closest_colour(requested_colour):
	requested_colour = interpret_color(requested_colour)
	min_colours = {}
	for key, name in color_names.items():
		r_c, g_c, b_c = webcolors.hex_to_rgb(key)
		rd = (r_c - requested_colour[0]) ** 2
		gd = (g_c - requested_colour[1]) ** 2
		bd = (b_c - requested_colour[2]) ** 2
		min_colours[(rd + gd + bd)] = name
	return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour, case=None):
	requested_colour = interpret_color(requested_colour)
	try:
		name = webcolors.rgb_to_name(requested_colour)
	except ValueError:
		name = closest_colour(requested_colour)
	if case is None:
		return name
	return case(name)
