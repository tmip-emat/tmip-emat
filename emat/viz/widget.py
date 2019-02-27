
from plotly.graph_objs import FigureWidget as _FigureWidget


class FigureWidget(_FigureWidget):
	"""FigureWidget with metadata."""

	def __init__(self, *args, metadata=None, **kwargs):
		super().__init__(*args, **kwargs)
		self._metadata = metadata if metadata is not None else {}

	@property
	def metadata(self):
		return self._metadata

	@metadata.setter
	def metadata(self, x):
		self._metadata = x
