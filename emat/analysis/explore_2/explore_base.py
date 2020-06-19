
import numpy
import pandas
import warnings
from emat.viz import colors
from emat.scope.box import GenericBox
from emat import styles

from plotly import graph_objs as go

from ipywidgets import Dropdown
import ipywidgets as widget

import logging
_logger = logging.getLogger('EMAT.widget')


class DataFrameExplorerBase():

	def __init__(
			self,
			data,
			selections=None,
			active_selection_name=None,
			reference_point=None,
	):
		assert isinstance(data, pandas.DataFrame)
		self.data = data
		if isinstance(selections, dict):
			make_selections = selections
			selections = None
		elif isinstance(selections, pandas.DataFrame) or selections is None:
			make_selections = None
		else:
			make_selections = {getattr(v, 'name', str(v)):v for v in selections}
			selections = None
		if selections is None:
			selections = pandas.DataFrame(
				data=True,
				index=self.data.index,
				columns=['None'],
			)
		assert isinstance(selections, pandas.DataFrame)
		self._selections = selections
		self._selection_defs = {}
		self._colors = {}
		self._update_state_ = set()
		if isinstance(reference_point, pandas.DataFrame) and len(reference_point)==1:
			reference_point = dict(reference_point.iloc[0])
		self._reference_point = reference_point
		if make_selections:
			for k,v in make_selections.items():
				self.new_selection(v, name=k, activate=False)
		if active_selection_name is None:
			active_selection_name = self.selection_names()[0]
		else:
			if active_selection_name not in self.selection_names():
				raise KeyError(f"active_selection_name '{active_selection_name}' not found")
		self._active_selection_name = active_selection_name

	def selection_names(self):
		return self._selections.columns

	def new_selection(self, values, name=None, color=None, activate=True):
		if name is None and hasattr(values, 'name'):
			name = values.name
		if name is None:
			name = self._active_selection_name
		assert isinstance(name, str)
		if isinstance(values, GenericBox):
			proposal = values.inside(self.data)
			self._selection_defs[name] = values
			values = proposal
		if isinstance(values, str):
			proposal = self.data.eval(values).fillna(0).astype(bool)
			self._selection_defs[name] = values
			values = proposal
		assert isinstance(values, pandas.Series)
		self._selections[name] = values
		if self._selections[name].dtype != numpy.bool:
			self._selections[name] = self._selections[name].fillna(0).astype(bool)
		if color is not None:
			self._colors[name] = color
		if activate:
			self._active_selection_name = name
			self._active_selection_changed()

	def active_selection(self):
		return self._selections[self._active_selection_name]

	def active_selection_name(self):
		return self._active_selection_name

	def active_selection_deftype(self):
		return self.selection_deftype(self.active_selection_name())

	def set_active_selection_name(self, activate_name):
		if activate_name not in self.selection_names():
			raise KeyError(f"activate_name '{activate_name}' not found")
		if self._active_selection_name != activate_name:
			self._active_selection_name = activate_name
			self._active_selection_changed()

	def active_selection_color(self):
		return self._colors.get(
			self._active_selection_name,
			colors.DEFAULT_HIGHLIGHT_COLOR,
		)

	def set_active_selection_color(self, color):
		# if self._active_selection_name in self._colors:
		# 	if color != self._colors[self._active_selection_name]:
		# 		print(f"change active selection [{self._active_selection_name}] color to {color}")
		self._colors[self._active_selection_name] = color

	def _active_selection_changed(self):
		pass

	def selection_deftype(self, name=None):
		"""
		Get the selection definition type for a selection.

		Args:
			name (str, optional):
				The name of the selection to check.  If not
				given, the active selection deftype is returned.

		Returns:
			{'explicit', 'box', 'unknown'}

		Raises:
			KeyError: If the name is not a known selection set.
		"""
		if name is None:
			name = self.active_selection_name()
		if name not in self.selection_names():
			raise KeyError(name)
		if name not in self._selection_defs:
			return 'explicit'
		if isinstance(self._selection_defs[name], GenericBox):
			return 'box'
		if isinstance(self._selection_defs[name], str):
			return 'expression'
		return 'unknown'

	def reference_point(self, column):
		if self._reference_point is None:
			return None
		return self._reference_point.get(column, None)

	def set_reference_point(self, column, value):
		if self._reference_point is None:
			self._reference_point = {}
		self._reference_point[column] = value

class DataFrameExplorer(DataFrameExplorerBase):

	def __init__(
			self,
			data,
			selections=None,
			active_selection_name=None,
			reference_point=None,
	):
		super().__init__(
			data,
			selections=selections,
			active_selection_name=active_selection_name,
			reference_point=reference_point,
		)
		self._active_selection_chooser = Dropdown(
			options=self.selection_names(),
			value=self.active_selection_name(),
		)
		self._active_selection_chooser.observe(self.set_active_selection_name, names='value')

	def set_active_selection_name(self, activate_name):
		if isinstance(activate_name, dict):
			if activate_name.get('type') == 'change':
				activate_name = activate_name.get('new')
			else: return
		if activate_name is None: return
		super().set_active_selection_name(activate_name)
		self._active_selection_chooser.options = self.selection_names()
		self._active_selection_chooser.value = activate_name
		self.set_active_selection_color(self.active_selection_color())

	def new_selection(self, values, name=None, color=None, activate=True):
		self._update_state_.add("DataFrameExplorer.new_selection")
		try:
			super().new_selection(values, name=name, color=color, activate=activate)
			self.refresh_selection_names()
		finally:
			self._update_state_.discard("DataFrameExplorer.new_selection")

	def refresh_selection_names(self):
		if "refresh_selection_names" in self._update_state_: return
		self._update_state_.add("refresh_selection_names")
		try:
			cache = self._active_selection_chooser.value
			self._active_selection_chooser.options = self.selection_names()
			try:
				self._active_selection_chooser.value = cache
			except:
				# The old value is gone, reset to the new selection value
				self.set_active_selection_name(self._active_selection_chooser.value)
		except AttributeError:
			pass
		finally:
			self._update_state_.discard("refresh_selection_names")