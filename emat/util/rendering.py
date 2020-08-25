
import re
import ast

from ..util.loggers import get_module_logger
_logger = get_module_logger(__name__)

from .xmle import Show, Elem

_png = re.compile(r"png(\(([^\)]*)\))?")
_svg = re.compile(r"svg(\(([^\)]*)\))?")

def _parse_paren(s, format):
	for part in s.split(","):
		k, v = part.split("=", 1)
		k = k.strip()
		v = v.strip()
		try:
			k = ast.literal_eval(k)
		except ValueError:
			pass
		try:
			v = ast.literal_eval(v)
		except ValueError:
			pass
		format[k] = v


def render_plotly(figure, format):
	"""
	Convert a plotly figure to a static image.

	Args:
		figure (plotly.graph_objs.Figure):
			The source figure to convert
		format (str or dict):
			A string or dictionary that contains the
			output formatting instructions. Any format
			accepted by the plotly `to_image` method
			can be given as a dictionary, which is passed
			as keyword arguments to that function.
			Alternatively, give a string that contains
			a format type, optionally called with other
			keyword arguments. Currently only *svg* and
			*png* are implemented using the string
			approach. If no implemented format is
			available, the original plotly Figure is
			returned.

	Returns:
		xmle.Elem or plotly.graph_objs.Figure

	Examples:
		>>> from plotly.graph_objects import Figure
		>>> render_plotly(Figure(), {'format':'png','width':300,'height':500})
		...
		>>> render_plotly(Figure(), "svg")
		...
		>>> render_plotly(Figure(), "png(width=500,height=500)")
		...

	"""
	if isinstance(format, str):
		is_png = _png.search(format)
		if is_png:
			format = dict(format="png")
			if is_png.group(2):
				_parse_paren(is_png.group(2), format)

	if isinstance(format, str):
		is_svg = _svg.search(format)
		if is_svg:
			format = dict(format="svg")
			if is_svg.group(2):
				_parse_paren(is_svg.group(2), format)

	_logger.debug(f"render format is {format}")

	fallback = format.pop('fallback', False)

	if isinstance(format, dict):
		try:
			return Show(figure.to_image(**format))
		except:
			if fallback:
				return figure
			else:
				import traceback
				err_txt = traceback.format_exc()
				return Elem('pre', text=str(err_txt))

	else:
		return figure