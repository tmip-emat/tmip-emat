
from .xmle import Show


def render_plotly(figure, format):
	"""
	Convert a plotly figure to a static image.

	Args:
		figure (plotly.graph_objs.Figure):
			The source figure to convert
		format (str):
			A string that contains the output format.
			Currently .svg and .png are implemented.
			If no implemented format is available,
			the original plotly Figure is returned.
	Returns:
		xmle.Elem or Figure
	"""
	if ".svg" in format:
		return Show(figure.to_image(format="svg"))
	elif ".png" in format:
		return Show(figure.to_image(format="png"))
	else:
		return figure