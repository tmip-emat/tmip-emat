
from ..util import webcolors
import re
import numpy as np

class Color:

	def __new__(cls, *args, **kwargs):
		if len(kwargs) == 0 and len(args) == 1 and args[0] is None:
			return None
		return super().__new__(cls)

	def __init__(self,r,g=None,b=None,a=None,default=None):
		if r is None and default is not None:
			r = Color(default)
		if isinstance(r, Color) and g is None and b is None:
			r,g,b,a = r.r, r.g, r.b, r.a
		elif isinstance(r, str) and g is None and b is None:
			r,g,b = interpret_color(r)
		self.r = np.clip(int(r),0,255)
		self.g = np.clip(int(g),0,255)
		self.b = np.clip(int(b),0,255)
		if a is None:
			self.a = None
		else:
			self.a = np.clip(a,0,1.0)

	def __repr__(self):
		if self.a is None:
			return self.rgb()
		return self.rgba()

	def alpha(self, a):
		return type(self)(self.r,self.g,self.b,a)

	def rgb(self):
		return f"rgb({self.r},{self.g},{self.b})"

	def rgba(self, a=None):
		if a is not None:
			return f"rgba({self.r},{self.g},{self.b},{np.clip(a,0,1.0)})"
		if self.a is not None:
			return f"rgba({self.r},{self.g},{self.b},{self.a})"
		return f"rgba({self.r},{self.g},{self.b},1.0)"


DEFAULT_BASE_COLOR_RGB = Color(31, 119, 180)         # Blue
DEFAULT_HIGHLIGHT_COLOR_RGB = Color(255, 127, 14)    # Orange

DEFAULT_BASE_COLOR = 'rgb(31, 119, 180)'      # Blue
DEFAULT_HIGHLIGHT_COLOR = 'rgb(255, 127, 14)' # Orange
DEFAULT_LASSO_COLOR = 'rgb(255, 46, 241)'     # Hot Pink
DEFAULT_PRIMTARGET_COLOR = 'rgb(227, 20, 20)' # Red
DEFAULT_EXPRESSION_COLOR = 'rgb(227, 20, 20)' # Red
DEFAULT_BOX_BG_COLOR = '#2ca02c'    # Green
DEFAULT_BOX_BG_BORDER_COLOR = '#217821'  # Green, 25% darker
DEFAULT_BOX_LINE_COLOR = '#2ca02c'  # Green
DEFAULT_REF_LINE_COLOR = '#000000'  # Black
DEFAULT_PLOT_BACKGROUND_COLOR = '#E5ECF6'

DEFAULT_REF_LINE_STYLE = dict(
	line=dict(
		width=2,
		color=DEFAULT_REF_LINE_COLOR,
		dash="dot",
	),
	opacity=0.8,
)


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


def high_contrast_mode(toggle=True):
	global DEFAULT_BASE_COLOR_RGB
	global DEFAULT_HIGHLIGHT_COLOR_RGB
	global DEFAULT_BASE_COLOR
	global DEFAULT_HIGHLIGHT_COLOR
	global DEFAULT_LASSO_COLOR
	global DEFAULT_PLOT_BACKGROUND_COLOR

	if toggle:
		DEFAULT_BASE_COLOR_RGB = Color(9, 56, 89)  # Dark Blue
		DEFAULT_HIGHLIGHT_COLOR_RGB = Color(255, 161, 79)  # Orange
		DEFAULT_BASE_COLOR = 'rgb(9, 56, 89)'  # Dark Blue
		DEFAULT_HIGHLIGHT_COLOR = 'rgb(255, 161, 79)'  # Light Orange
		DEFAULT_LASSO_COLOR = 'rgb(255, 89, 244)'  # Light Pink
		DEFAULT_PLOT_BACKGROUND_COLOR = '#f5f6f7'
	else:
		DEFAULT_BASE_COLOR_RGB = Color(31, 119, 180)  # Blue
		DEFAULT_HIGHLIGHT_COLOR_RGB = Color(255, 127, 14)  # Orange
		DEFAULT_BASE_COLOR = 'rgb(31, 119, 180)'  # Blue
		DEFAULT_HIGHLIGHT_COLOR = 'rgb(255, 127, 14)'  # Orange
		DEFAULT_LASSO_COLOR = 'rgb(255, 46, 241)'  # Hot Pink
		DEFAULT_PLOT_BACKGROUND_COLOR = '#E5ECF6'
