
from collections import namedtuple
from collections.abc import MutableMapping, Mapping
import itertools
from typing import Collection
import pandas
import numpy
import copy
from abc import ABC, abstractmethod

from ..util.distributions import truncated, get_distribution_bounds
from math import isclose

from .scope import Scope, ScopeError
from .parameter import IntegerParameter, CategoricalParameter, BooleanParameter
from .. import styles
from ..viz import colors

Bounds = namedtuple('Bounds', ['lowerbound', 'upperbound'])

Bounds.__doc__ = """
A lower and upper bound as a 2-tuple.

Args:
	lowerbound (numeric or None): 
		The lower bound to set, or None 
		if there is no lower bound.
	upperbound (numeric or None): 
		The upper bound to set, or None 
		if there is no upper bound.
"""

class GenericBox(MutableMapping, ABC):
	# Generic methods applicable to both Box and ChainedBox

	def inside(self, df):
		"""
		For each row of a DataFrame, identify if it is inside the box.

		Args:
			df (pandas.DataFrame): Must include a column matching every
				thresholded feature.

		Returns:
			pandas.Series
				With dtype bool.
		"""
		within = pandas.Series(True, index=df.index)
		for label, bounds in self.thresholds.items():
			if label not in df.columns: continue
			if isinstance(bounds, set):
				within &= numpy.in1d(df[label], list(bounds))
			else:
				if bounds.lowerbound is not None:
					within &= (df[label] >= bounds.lowerbound)
				if bounds.upperbound is not None:
					within &= (df[label] <= bounds.upperbound)
		return within

	def __init__(self):
		self._scope = None

	@property
	@abstractmethod
	def thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The restricted dimensions in this Box, with feature names as
			keys and :class:`Bounds` or a :class:`Set` of available discrete
			values as the dictionary values.
		"""
		raise NotImplementedError

	@property
	@abstractmethod
	def demanded_features(self):
		"""
		Set[str]: A set of features upon which thresholds are set.
		"""
		raise NotImplementedError

	@property
	@abstractmethod
	def relevant_features(self):
		"""
		 Set[str]:
			A :class:`Set` of features that are relevant for this Box.
			These are features, which are not themselves constrained,
			but should be considered in any analytical report developed
			based on this Box.
		"""
		raise NotImplementedError

	@property
	def relevant_and_demanded_features(self):
		"""
		Set[str]: The union of relevant and demanded features.
		"""
		return self.relevant_features | self.demanded_features

	@property
	def scope(self):
		"""Scope: A scope associated with this Box."""
		return self._scope

	@scope.setter
	def scope(self, x):
		if x is None or isinstance(x, Scope):
			self._scope = x
		else:
			raise TypeError('scope must be Scope or None')

	@scope.deleter
	def scope(self):
		self._scope = None

	@abstractmethod
	def set_bounds(self, key, lowerbound, upperbound=None):
		"""
		Set both lower and upper bounds.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			lowerbound (numeric, None, or Bounds):
				The lower bound, or a Bounds object that gives
				upper and lower bounds (in which case the `upperbound`
				argument is ignored).  Set explicitly to 'None' to
				leave unbounded from below.
			upperbound (numeric or None, default None):
				The upper bound. Set to 'None' to
				leave unbounded from above.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.

		"""
		raise NotImplementedError

	def set_lower_bound(self, key, value):
		"""
		Set a lower bound, retaining existing upper bound.

		Args:
			key (str):
				The feature name to which this lower bound
				will be attached.
			value (numeric or None):
				The lower bound. Set explicitly to 'None' to
				leave unbounded from below.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
		"""
		current = self.thresholds.get(key, Bounds(None,None))
		if isinstance(current, set):
			raise ValueError("cannot set lowerbound on a set")
		self.set_bounds(key, value, current.upperbound)

	def set_upper_bound(self, key, value):
		"""
		Set an upper bound, retaining existing lower bound.

		Args:
			key (str):
				The feature name to which this upper bound
				will be attached.
			value (numeric or None):
				The upper bound. Set explicitly to 'None' to
				leave unbounded from above.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
		"""
		current = self.thresholds.get(key, Bounds(None,None))
		if isinstance(current, set):
			raise ValueError("cannot set upperbound on a set")
		self.set_bounds(key, current.lowerbound, value)

	def add_to_allowed_set(self, key, value):
		"""
		Add a value to the allowed set

		Args:
			key (str):
				The feature name to which these allowed values
				will be attached.
			value (Any):
				A value to add to the allowed set.

		Raises:
			ValueError:
				If there is already a directional Bounds set for `key`.
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
		"""
		current = self.thresholds.get(key, set())
		if isinstance(current, Bounds):
			raise ValueError("cannot add to Bounds")
		current.add(value)
		self.replace_allowed_set(key, current)

	def remove_from_allowed_set(self, key, value):
		"""
		Remove a value from the allowed set

		Args:
			key (str):
				The feature name to which these allowed values
				will be attached.
			value (Any):
				A value to remove from the allowed set.

		Raises:
			ValueError:
				If the threshold set for `key` is a directional Bounds
				instead of a set.
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.
			KeyError:
				If the value to be removed was not already in the
				allowed set.
		"""
		current = None
		if key not in self.thresholds and self.scope is not None:
			v = self.scope.get_cat_values(key)
			if v is None:
				raise ValueError("cannot use allowed_set for float or int values, use Bounds instead")
			current = set(v)
		else:
			current = self.thresholds.get(key, set())
		if isinstance(current, Bounds):
			raise ValueError("cannot remove from Bounds")
		current.remove(value)
		self.replace_allowed_set(key, current)

	@abstractmethod
	def replace_allowed_set(self, key, values):
		"""
		Replace the allowed set.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			values (set):
				A set of values to use as the allowed set.
		"""
		raise NotImplementedError

	# def _compute_histogram(self, col, selection, bins=20):
	# 	if self._viz_data is None:
	# 		return
	# 	bar_heights, bar_x = numpy.histogram(self._viz_data[col], bins=bins)
	# 	bar_heights_select, bar_x = numpy.histogram(self._viz_data[col][selection], bins=bar_x)
	# 	return pandas.DataFrame({
	# 		'Total Freq': bar_heights,
	# 		'Inside Freq': bar_heights_select,
	# 		'Bins_Left': bar_x[:-1],
	# 		'Bins_Width': bar_x[1:] - bar_x[:-1],
	# 	})
	#
	# def _compute_frequencies(self, col, selection, labels):
	# 	if self._viz_data is None:
	# 		return
	# 	v = self._viz_data[col].astype(
	# 		pandas.CategoricalDtype(categories=labels, ordered=False)
	# 	).cat.codes
	# 	bar_heights, _ = numpy.histogram(v, bins=numpy.arange(0, len(labels) + 1))
	# 	bar_heights_select, _ = numpy.histogram(v[selection], bins=numpy.arange(0, len(labels) + 1))
	#
	# 	return pandas.DataFrame({
	# 		'Total Freq': bar_heights,
	# 		'Inside Freq': bar_heights_select,
	# 		'Label': labels,
	# 	})
	#
	# def _update_histogram_figure(self, col, *, selection=None):
	# 	if col in self._figures and self._viz_data is not None:
	# 		fig = self._figures[col]
	# 		bins = fig._bins
	# 		if selection is None:
	# 			selection = self.inside(self._viz_data)
	# 		h_data = self._compute_histogram(col, selection, bins=bins)
	# 		with fig.batch_update():
	# 			fig.data[0].y = h_data['Inside Freq']
	# 			fig.data[1].y = h_data['Total Freq'] - h_data['Inside Freq']
	#
	# def _update_frequencies_figure(self, col, *, selection=None):
	# 	if col in self._figures and self._viz_data is not None:
	# 		fig = self._figures[col]
	# 		labels = fig._labels
	# 		if selection is None:
	# 			selection = self.inside(self._viz_data)
	# 		h_data = self._compute_frequencies(col, selection, labels=labels)
	# 		with fig.batch_update():
	# 			fig.data[0].y = h_data['Inside Freq']
	# 			fig.data[1].y = h_data['Total Freq'] - h_data['Inside Freq']
	#
	# def _update_all_histogram_figures(self):
	# 	selection = self.inside(self._viz_data)
	# 	for col in self._figures:
	# 		if self._figures[col]._figure_kind == 'histogram':
	# 			self._update_histogram_figure(col, selection=selection)
	# 		elif self._figures[col]._figure_kind == 'frequency':
	# 			self._update_frequencies_figure(col, selection=selection)
	#
	# def _create_histogram_figure(self, col, bins=20):
	# 	if self._viz_data is None:
	# 		return
	# 	if col in self._figures:
	# 		self._update_histogram_figure(col)
	# 	else:
	# 		selection = self.inside(self._viz_data)
	# 		h_data = self._compute_histogram(col, selection, bins=bins)
	# 		fig = go.FigureWidget(
	# 			data=[
	# 				go.Bar(
	# 					x=h_data['Bins_Left'],
	# 					y=h_data['Inside Freq'],
	# 					width=h_data['Bins_Width'],
	# 					name='Inside',
	# 					marker_color=colors.DEFAULT_HIGHLIGHT_COLOR,
	# 				),
	# 				go.Bar(
	# 					x=h_data['Bins_Left'],
	# 					y=h_data['Total Freq'] - h_data['Inside Freq'],
	# 					width=h_data['Bins_Width'],
	# 					name='Outside',
	# 					marker_color=colors.DEFAULT_BASE_COLOR,
	# 				),
	# 			],
	# 			layout=dict(
	# 				barmode='stack',
	# 				showlegend=False,
	# 				margin=dict(l=10, r=10, t=10, b=10),
	# 				**styles.figure_dims,
	# 			),
	# 		)
	# 		fig._bins = bins
	# 		fig._figure_kind = 'histogram'
	# 		self._figures[col] = fig
	#
	# def _create_frequencies_figure(self, col, labels=None):
	# 	if self._viz_data is None:
	# 		return
	# 	if col in self._figures:
	# 		self._update_frequencies_figure(col)
	# 	else:
	# 		selection = self.inside(self._viz_data)
	# 		h_data = self._compute_frequencies(col, selection, labels=labels)
	# 		fig = go.FigureWidget(
	# 			data=[
	# 				go.Bar(
	# 					x=h_data['Label'],
	# 					y=h_data['Inside Freq'],
	# 					name='Inside',
	# 					marker_color=colors.DEFAULT_HIGHLIGHT_COLOR,
	# 				),
	# 				go.Bar(
	# 					x=h_data['Label'],
	# 					y=h_data['Total Freq'] - h_data['Inside Freq'],
	# 					name='Outside',
	# 					marker_color=colors.DEFAULT_BASE_COLOR,
	# 				),
	# 			],
	# 			layout=dict(
	# 				barmode='stack',
	# 				showlegend=False,
	# 				margin=dict(l=10, r=10, t=10, b=10),
	# 				width=250,
	# 				height=150,
	# 			),
	# 		)
	# 		fig._labels = labels
	# 		fig._figure_kind = 'frequency'
	# 		self._figures[col] = fig
	#
	#
	# def set_viz_data(self, df):
	# 	self._viz_data = df
	#
	# def get_histogram_figure(self, col, bins=20):
	# 	try:
	# 		this_type = self.scope.get_dtype(col)
	# 	except:
	# 		this_type = 'float'
	# 	if this_type in ('cat','bool'):
	# 		return self.get_frequency_figure(col)
	# 	self._create_histogram_figure(col, bins=bins)
	# 	return self._figures[col]
	#
	# def get_frequency_figure(self, col):
	# 	if self.scope.get_dtype(col) == 'cat':
	# 		labels = self.scope.get_cat_values(col)
	# 	else:
	# 		labels = [False, True]
	# 	self._create_frequencies_figure(col, labels=labels)
	# 	return self._figures[col]
	#
	# def _make_range_widget(
	# 		self,
	# 		i,
	# 		min_value=None,
	# 		max_value=None,
	# 		readout_format=None,
	# 		integer=False,
	# 		steps=20,
	# ):
	# 	"""Construct a RangeSlider to manipulate a Box threshold."""
	#
	# 	current_setting = self.get(i, (None, None))
	#
	# 	# Use current setting as min and max if still unknown
	# 	if current_setting[0] is not None and min_value is None:
	# 		min_value = current_setting[0]
	# 	if current_setting[1] is not None and max_value is None:
	# 		max_value = current_setting[1]
	#
	# 	if min_value is None:
	# 		raise ValueError("min_value cannot be None if there is no current setting")
	# 	if max_value is None:
	# 		raise ValueError("max_value cannot be None if there is no current setting")
	#
	# 	current_min = min_value if current_setting[0] is None else current_setting[0]
	# 	current_max = max_value if current_setting[1] is None else current_setting[1]
	#
	# 	slider_type = widget.IntRangeSlider if integer else widget.FloatRangeSlider
	#
	# 	controller = slider_type(
	# 		value=[current_min, current_max],
	# 		min=min_value,
	# 		max=max_value,
	# 		step=((max_value - min_value) / steps) if not integer else 1,
	# 		disabled=False,
	# 		continuous_update=False,
	# 		orientation='horizontal',
	# 		readout=True,
	# 		readout_format=readout_format,
	# 		description='',
	# 		style=styles.slider_style,
	# 		layout=styles.slider_layout,
	# 	)
	#
	# 	def on_value_change(change):
	# 		from ..util.loggers import get_logger
	# 		get_logger().critical("VALUE CHANGE")
	# 		new_setting = change['new']
	# 		if new_setting[0] <= min_value or isclose(new_setting[0], min_value):
	# 			new_setting = (None, new_setting[1])
	# 		if new_setting[1] >= max_value or isclose(new_setting[1], max_value):
	# 			new_setting = (new_setting[0], None)
	# 		self.set_bounds(i, *new_setting)
	# 		self._update_all_histogram_figures()
	#
	# 	controller.observe(on_value_change, names='value')
	#
	# 	return controller
	#
	# def _make_togglebutton_widget(
	# 		self,
	# 		i,
	# 		cats=None,
	# 		*,
	# 		df=None,
	# ):
	# 	"""Construct a MultiToggleButtons to manipulate a Box categorical set."""
	#
	# 	if cats is None and df is not None:
	# 		if isinstance(df[i].dtype, pandas.CategoricalDtype):
	# 			cats = df[i].cat.categories
	#
	# 	current_setting = self.get(i, set())
	#
	# 	from ..analysis.widgets import MultiToggleButtons_AllOrSome
	# 	controller = MultiToggleButtons_AllOrSome(
	# 		description='',
	# 		style=styles.slider_style,
	# 		options=list(cats),
	# 		disabled=False,
	# 		button_style='',  # 'success', 'info', 'warning', 'danger' or ''
	# 		layout=styles.slider_layout,
	# 	)
	# 	controller.values = current_setting
	#
	# 	def on_value_change(change):
	# 		new_setting = change['new']
	# 		self.replace_allowed_set(i, new_setting)
	# 		self._update_all_histogram_figures()
	#
	# 	controller.observe(on_value_change, names='value')
	#
	# 	return controller
	#
	#
	# def get_widget(
	# 		self,
	# 		i,
	# 		min_value=None,
	# 		max_value=None,
	# 		readout_format=None,
	# 		steps=20,
	# 		*,
	# 		df=None,
	# 		histogram=None,
	# 		tall=True,
	# ):
	# 	"""Get a control widget for a Box threshold."""
	#
	# 	if self.scope is None:
	# 		raise ValueError('cannot get_widget with no scope')
	#
	# 	if not hasattr(self, '_widgets'):
	# 		self._widgets = {}
	#
	# 	if i not in self._widgets:
	# 		# Extract min and max from scope if not given explicitly
	# 		if i not in self.scope.get_measure_names():
	# 			if min_value is None:
	# 				min_value = self.scope[i].min
	# 			if max_value is None:
	# 				max_value = self.scope[i].max
	#
	# 		# Extract min and max from `df` if still missing (i.e. for Measures)
	# 		if df is not None:
	# 			if min_value is None:
	# 				min_value = df[i].min()
	# 			if max_value is None:
	# 				max_value = df[i].max()
	#
	# 		# Extract min and max from `_viz_data` if still missing
	# 		if self._viz_data is not None:
	# 			if min_value is None:
	# 				min_value = self._viz_data[i].min()
	# 			if max_value is None:
	# 				max_value = self._viz_data[i].max()
	#
	# 		if isinstance(self.scope[i], BooleanParameter):
	# 			self._widgets[i] = self._make_togglebutton_widget(
	# 				i,
	# 				cats=[False, True],
	# 			)
	# 		elif isinstance(self.scope[i], CategoricalParameter):
	# 			cats = self.scope.get_cat_values(i)
	# 			self._widgets[i] = self._make_togglebutton_widget(
	# 				i,
	# 				cats=cats,
	# 			)
	# 		elif isinstance(self.scope[i], IntegerParameter):
	# 			readout_format = readout_format or 'd'
	# 			self._widgets[i] = self._make_range_widget(
	# 				i,
	# 				min_value=min_value,
	# 				max_value=max_value,
	# 				readout_format=readout_format,
	# 				integer=True,
	# 				steps=steps,
	# 			)
	# 		else:
	# 			readout_format = readout_format or '.3g'
	# 			self._widgets[i] = self._make_range_widget(
	# 				i,
	# 				min_value=min_value,
	# 				max_value=max_value,
	# 				readout_format=readout_format,
	# 				integer=False,
	# 				steps=steps,
	# 			)
	#
	# 	if tall:
	# 		if not isinstance(histogram, Mapping):
	# 			histogram = {}
	# 		return widget.VBox(
	# 			[
	# 				widget.Label(i),
	# 				self.get_histogram_figure(i, **histogram),
	# 				self._widgets[i],
	# 			],
	# 			layout=styles.widget_frame,
	# 		)
	#
	# 	if histogram is not None:
	# 		if not isinstance(histogram, Mapping):
	# 			histogram = {}
	# 		return widget.HBox(
	# 			[self._widgets[i], self.get_histogram_figure(i, **histogram)],
	# 			layout=dict(align_items = 'center'),
	# 		)
	# 	else:
	# 		return self._widgets[i]
	#
	# def visualization(self, include=None, data=None):
	#
	# 	if self.scope is None:
	# 		raise ValueError('cannot create visualization with no scope')
	#
	# 	if data is not None:
	# 		self.set_viz_data(data)
	# 		self._figures.clear()
	#
	# 	if include is None:
	# 		include = []
	#
	# 	viz_widgets = []
	# 	include = set(include)
	# 	include = include | self.relevant_and_demanded_features
	# 	for i in self.scope.get_parameter_names() + self.scope.get_measure_names():
	# 		if i in include:
	# 			viz_widgets.append(self.get_widget(i))
	#
	# 	return widget.Box(viz_widgets, layout=widget.Layout(flex_flow='row wrap'))


class Box(GenericBox):
	"""
	A Box defines a set of restricted dimensions for a Scope.

	Args:
		name (str): The name for this Box.
		parent (str, optional):
			The name of the parent for this Box.  When extracted
			as a :class:`ChainedBox` from a collection of :class:`Boxes`,
			the thresholds will also include any thresholds inherited
			from this box's ancestor(s).
		scope (Scope, optional):
			A scope to associate with this box.
		upper_bounds (Mapping[str, numeric], optional):
			If given, a mapping with keys giving feature names
			and values giving an upper bound for each feature.
		lower_bounds (Mapping[str, numeric], optional):
			If given, a mapping with keys giving feature names
			and values giving a lower bound for each feature.
		bounds (Mapping[str, Bounds], optional):
			If given, a mapping with keys giving feature names
			and values giving :class:`Bounds` for each feature.
		allowed (Mapping[str, Set], optional):
			If given, a mapping with keys giving feature names
			and values giving the available :class:`Set` for
			each feature.
		relevant (Iterable, optional):
			If given, a set of names of relevant features.

	Attributes:
		thresholds (Dict[str,Union[Bounds,Set]]):
			The restricted dimensions in this Box, with feature names as
			keys and :class:`Bounds` or a :class:`Set` of available discrete
			values as the dictionary values.
		relevant_features (Set[str]):
			A :class:`Set` of features that are relevant for this Box.
			These are features, which are not themselves constrained,
			but should be considered in any analytical report developed
			based on this Box.


	"""
	def __init__(
			self,
			name,
			parent=None,
			scope=None,
			upper_bounds=None,
			lower_bounds=None,
			bounds=None,
			allowed=None,
			relevant=None,
	):
		super().__init__()
		self._thresholds = {}

		if relevant is None:
			self.relevant_features = set()
		else:
			self.relevant_features = set(relevant)

		self.parent_box_name = parent
		self.scope = scope
		self.name = name

		if upper_bounds:
			for k,v in upper_bounds.items():
				self.set_upper_bound(k,v)

		if lower_bounds:
			for k,v in lower_bounds.items():
				self.set_lower_bound(k,v)

		if bounds:
			for k,v in bounds.items():
				self.set_bounds(k,v)

		if allowed:
			for k,v in allowed.items():
				self.replace_allowed_set(k,v)

	@property
	def scope(self):
		"""Scope: A scope associated with this Box."""
		return self._scope

	@scope.setter
	def scope(self, x):
		if x is None or isinstance(x, Scope):
			self._scope = x
		else:
			raise TypeError('scope must be Scope or None')

	@property
	def thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The restricted dimensions in this Box, with feature names as
			keys and :class:`Bounds` or a :class:`Set` of available discrete
			values as the dictionary values.
		"""
		return self._thresholds

	@thresholds.setter
	def thresholds(self, value):
		if not isinstance(value, MutableMapping):
			raise TypeError(f'thresholds must be MutableMapping not {type(value)}')
		self._thresholds = value

	@thresholds.deleter
	def thresholds(self):
		self._thresholds = {}

	@property
	def relevant_features(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			A :class:`Set` of features that are relevant for this Box.
			These are features, which are not themselves constrained,
			but should be considered in any analytical report developed
			based on this Box.
		"""
		return self._relevant_features

	@relevant_features.setter
	def relevant_features(self, value):
		if not isinstance(value, set):
			raise TypeError(f'thresholds must be MutableMapping not {type(value)}')
		self._relevant_features = value

	@relevant_features.deleter
	def relevant_features(self):
		self._relevant_features = set()

	@property
	def measure_thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The thresholds in this Box associated with performance measures.

			A Scope must be associated with this Box to access this property.
		"""
		if self.scope is None:
			raise ValueError("need scope")
		names = self.scope.get_measure_names()
		return {k:v for k,v in self._thresholds.items() if k in names}

	@property
	def uncertainty_thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The thresholds in this Box associated with exogenous uncertainties.

			A Scope must be associated with this Box to access this property.
		"""
		if self.scope is None:
			raise ValueError("need scope")
		names = self.scope.get_uncertainty_names()
		return {k:v for k,v in self._thresholds.items() if k in names}

	@property
	def lever_thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The thresholds in this Box associated with policy levers.

			A Scope must be associated with this Box to access this property.
		"""
		if self.scope is None:
			raise ValueError("need scope")
		names = self.scope.get_lever_names()
		return {k:v for k,v in self._thresholds.items() if k in names}

	def to_json(self):
		"""
		Dump the thresholds for this box to a json string.

		Only the thresholds are saved, not the scope.

		Returns:
			str
		"""
		temp = {'_name_':self.name}
		for k,v in self.thresholds.items():
			if isinstance(v, Bounds):
				temp[k] = {'lowerbound':v.lowerbound, 'upperbound':v.upperbound}
			else:
				temp[k] = list(v)
		from ..util.json_encoder import dumps
		return dumps(temp)

	@classmethod
	def from_json(cls, j):
		import json
		temp = json.loads(j)
		self = cls(temp.get('_name_', None))
		for k, v in temp.items():
			if k == '_name_':
				pass
			elif isinstance(v, dict):
				self.set_bounds(k, v.get('lowerbound', None), v.get('upperbound', None))
			else:
				self.replace_allowed_set(k, v)
		return self

	def __getitem__(self, key):
		return self._thresholds[key]

	def __setitem__(self, key, value):
		if not isinstance(value, (Bounds, set)):
			raise TypeError('thresholds must be Bounds or a set')
		if self.scope:
			if key in self.scope.get_all_names():
				self._thresholds[key] = value
			else:
				raise ScopeError("cannot set threshold on '{key}'")
		else:
			self._thresholds[key] = value

	def __delitem__(self, key):
		del self._thresholds[key]

	def clear(self):
		"""
		Clear thresholds and relevant_features.
		"""
		self._thresholds = {}
		self._relevant_features = set()

	@property
	def demanded_features(self):
		"""
		Set[str]: A set of features upon which thresholds are set.
		"""
		t = set(self._thresholds.keys())
		return t

	def set_bounds(self, key, lowerbound, upperbound=None):
		"""
		Set both lower and upper bounds.

		If values are both set to None and this key becomes
		unbounded but it was not previously unbounded, it is moved from
		thresholds to relevant_features.  Conversely, if no bounds were
		previously set but the key appears in relevant_features, it is
		removed from that set.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			lowerbound (numeric, None, or Bounds):
				The lower bound, or a Bounds object that gives
				upper and lower bounds (in which case the `upperbound`
				argument is ignored).  Set explicitly to 'None' to
				leave unbounded from below.
			upperbound (numeric or None, default None):
				The upper bound. Set to 'None' to
				leave unbounded from above.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.

		"""

		if isinstance(lowerbound, Bounds):
			b = lowerbound
			lowerbound, upperbound = b.lowerbound, b.upperbound

		if self.scope is not None:
			if key not in self.scope.get_all_names():
				raise ScopeError(f"cannot set bounds on '{key}'")

		if lowerbound is None and upperbound is None:
			if key in self._thresholds:
				del self._thresholds[key]
				self._relevant_features.add(key)
		else:
			self._thresholds[key] = Bounds(lowerbound, upperbound)
			if key in self._relevant_features:
				self._relevant_features.remove(key)

	def replace_allowed_set(self, key, values):
		"""
		Replace the allowed set.

		If the new allowed set is the same as the complete set of
		possible values defined in the scope, this key becomes
		unbounded and it is moved from thresholds to
		relevant_features.  Conversely, if no bounds were previously
		set but the key appears in relevant_features, it is
		removed from that set.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			values (set):
				A set of values to use as the allowed set.
				If an empty set or None is given, then any
				values are allowed.
		"""
		action = True
		if self.scope is not None:
			if key not in self.scope.get_all_names():
				raise ScopeError(f"cannot set allowed_set on '{key}'")
			cat_values = set(self.scope.get_cat_values(key))
			if values is None or len(values)==0:
				values = cat_values

			# handle True and False
			values = set(values)
			_t = True in cat_values
			_f = False in cat_values
			if _t or _f:
				_values = set()
				for v in values:
					if _t and str(v).lower() == 'true':
						_values.add(True)
					elif _f and str(v).lower() == 'false':
						_values.add(False)
					else:
						_values.add(v)
				values = _values

			if not cat_values.issuperset(values):
				raise ScopeError(f"allowed_set is not a subset of scope defined values for '{key}'")
			if len(cat_values) == len(values):
				action = False
		if action:
			self._thresholds[key] = set(values)
			if key in self._relevant_features:
				self._relevant_features.remove(key)
		else:
			if key in self._thresholds:
				del self._thresholds[key]
				self._relevant_features.add(key)

	def __iter__(self):
		return itertools.chain(
			iter(self._thresholds),
		)

	def __len__(self):
		return (
			len(self._thresholds)
		)

	def __repr__(self):
		if self.keys() or self.relevant_features or self.name=='0':
			demands = list(self.keys()) or [" "]
			relevent = list(self.relevant_features) or [" "]
			m = max(
				max(map(len, demands)) + 1,
				max(map(len, relevent)) + 1
			)
			members = []
			for k, v in self.items():
				if isinstance(v, Bounds):
					if v.lowerbound is None:
						if v.upperbound is None:
							v_ = ': unbounded'
						else:
							v_ = f' <= {v.upperbound}'
					else:
						if v.upperbound is None:
							v_ = f' >= {v.lowerbound}'
						else:
							v_ = f': {v.lowerbound} to {v.upperbound}'
				else:
					v_ = ': ' + repr(v)
				members.append("● "+k.rjust(m) + v_)
			for k in self.relevant_features:
				members.append("◌ "+k.rjust(m))

			head = f"{self.__class__.__name__}: {self.name}"
			if hasattr(self, 'coverage'):
				head += f'\n   coverage: {self.coverage:.5f}'
			if hasattr(self, 'density'):
				head += f'\n   density:  {self.density:.5f}'
			if hasattr(self, 'mass'):
				head += f'\n   mass:     {self.mass:.5f}'
			if members:
				return head+"\n   " + '\n   '.join(members)
			else:
				return head
		else:
			return "<empty "+ self.__class__.__name__ + ">"

	def __get_truncated_parameters(self, source):
		"""Get a list of truncate parameters.

		This method requires a scope to be set, and will adjust
		the uncertainty distributions in the scope to be
		appropriately truncated.

		Args:
			source (Collection): The list of parameters to possibly truncate.
		"""
		result = []
		for i in source:
			i = copy.deepcopy(i)
			if i.name in self._thresholds:
				bounds = self._thresholds[i.name]
				if isinstance(bounds, Bounds):
					lowerbound, upperbound = bounds
					if lowerbound is None:
						lowerbound = -numpy.inf
					if upperbound is None:
						upperbound = numpy.inf
					i.dist = truncated(i.dist, lowerbound, upperbound)
					i.lower_bound, i.upper_bound = get_distribution_bounds(i.dist)
				else:
					from .parameter import CategoricalParameter
					i = CategoricalParameter(i.name, bounds, singleton_ok=True)
			result.append(i)
		return result

	def get_uncertainties(self):
		"""Get a list of exogenous uncertainties.

		This method requires a scope to be set, and will adjust
		the uncertainty distributions in the scope to be
		appropriately truncated.
		"""
		return self.__get_truncated_parameters(self.scope.get_uncertainties())

	def get_levers(self):
		"""Get a list of policy levers.

		This method requires a scope to be set, and will adjust
		the policy lever distributions in the scope to be
		appropriately truncated.
		"""
		return self.__get_truncated_parameters(self.scope.get_levers())

	def get_constants(self):
		"""Get a list of model constants.

		This method requires a scope to be set, and will pass through
		constants in the scope unaltered."""
		return self.scope.get_constants()

	def get_parameters(self):
		"""Get a list of model parameters (uncertainties+levers+constants)."""
		return self.get_constants() + self.get_uncertainties() + self.get_levers()

	def get_measures(self):
		"""Get a list of performance measures."""
		return self.scope.get_measures()



class ChainedBox(GenericBox):
	"""
	A Box defines a set of restricted dimensions for a Scope.

	Args:
		boxes (Boxes):
			A collection of Boxes from which to assemble a chain.
		name (str):
			The name for this ChainedBox.  This must be the name of
			a Box in `boxes`, which serves as the seed for the chain.
			Ancestors are added recursively by finding the parent box
			of each box in the chain, until a box is found with no parent.

	"""

	def __init__(self, boxes, name):
		"""

		Parameters
		----------
		boxes : Mapping
			Dictionary of {str:Box} pairs
		name : str
			Name of this chained box
		"""
		GenericBox.__init__(self)
		c = boxes[name]
		self.chain = [c]
		self.names = [name]
		while c.parent_box_name is not None:
			self.names.insert(0, c.parent_box_name)
			c = boxes[c.parent_box_name]
			self.chain.insert(0, c)

	def __getitem__(self, key):
		return self.thresholds[key]

	def __setitem__(self, key, value):
		self.chain[-1][key] = value

	def __delitem__(self, key):
		del self.chain[-1][key]

	def clear(self):
		"""
		Clear thresholds and relevant_features on the last Box in the chain.
		"""
		self.chain[-1]._thresholds.clear()
		self.chain[-1]._relevant_features.clear()

	def __iter__(self):
		return itertools.chain(
			iter(self.thresholds),
		)

	def __len__(self):
		return len(self.thresholds)

	@property
	def name(self):
		"""str: The name of the last (defining) Box in this chain."""
		return self.names[-1]

	@property
	def thresholds(self):
		"""
		Dict[str,Union[Bounds,Set]]:
			The restricted dimensions in this ChainedBox, with feature names as
			keys and the Bounds or available set as the values.
		"""
		t = {}
		for single in self.chain:
			t.update(single.thresholds)
		return t

	@thresholds.setter
	def thresholds(self, value):
		self.chain[-1].thresholds = value

	@thresholds.deleter
	def thresholds(self):
		self.chain[-1].thresholds = {}

	def measure_thresholds(self):
		"""
		The thresholds in this Box or its ancestor(s) associated with performance measures.

		A Scope must be associated with each Box in the chain to access this property.

		Returns:
			Dict[str,Union[Bounds,Set]]
		"""
		t = {}
		for single in self.chain:
			t.update(single.measure_thresholds)
		return t

	def uncertainty_thresholds(self):
		"""
		The thresholds in this Box or its ancestor(s) associated with exogenous uncertainties.

		A Scope must be associated with each Box in the chain to access this property.

		Returns:
			Dict[str,Union[Bounds,Set]]
		"""
		t = {}
		for single in self.chain:
			t.update(single.uncertainty_thresholds)
		return t

	def lever_thresholds(self):
		"""
		The thresholds in this Box or its ancestor(s) associated with policy levers.

		A Scope must be associated with each Box in the chain to access this property.

		Returns:
			Dict[str,Union[Bounds,Set]]
		"""
		t = {}
		for single in self.chain:
			t.update(single.lever_thresholds)
		return t

	@property
	def relevant_features(self):
		"""
		Set[str]: A set of features that are relevant at any step of the chain.
		"""
		t = set()
		for single in self.chain:
			t |= single.relevant_features
		return t

	@property
	def demanded_features(self):
		"""
		Set[str]: A set of features upon which thresholds are set at any step of the chain.
		"""
		t = set()
		for single in self.chain:
			t |= set(single.thresholds.keys())
		return t

	def __repr__(self):
		if self.keys() or self.relevant_features or True:
			demands = list(self.keys()) or [" "]
			relevent = list(self.relevant_features) or [" "]
			m = max(
				max(map(len, demands)) + 1,
				max(map(len, relevent)) + 1
			)
			members = []
			for k, v in self.items():
				if isinstance(v, Bounds):
					if v.lowerbound is None:
						if v.upperbound is None:
							v_ = ': unbounded'
						else:
							v_ = f' <= {v.upperbound}'
					else:
						if v.upperbound is None:
							v_ = f' >= {v.lowerbound}'
						else:
							v_ = f': {v.lowerbound} to {v.upperbound}'
				else:
					v_ = ': ' + repr(v)
				members.append("● "+k.rjust(m) + v_)
			for k in self.relevant_features:
				members.append("◌ "+k.rjust(m))

			head = f"{self.__class__.__name__}: {self.name}"
			if hasattr(self, 'coverage'):
				head += f'\n   coverage: {self.coverage:.5f}'
			if hasattr(self, 'density'):
				head += f'\n   density:  {self.density:.5f}'
			if hasattr(self, 'mass'):
				head += f'\n   mass:     {self.mass:.5f}'
			return head+"\n   " + '\n   '.join(members)
		else:
			return "<empty "+ self.__class__.__name__ + ">"


	def chain_repr(self):
		return "\n".join(f"{repr(c)}" for n,c in zip(self.names,self.chain))

	@property
	def scope(self):
		"""Scope: A scope associated with this Box."""
		return self.chain[-1]._scope

	@scope.setter
	def scope(self, x):
		if x is None or isinstance(x, Scope):
			self.chain[-1]._scope = x
		else:
			raise TypeError('scope must be Scope or None')

	def replace_allowed_set(self, key, values):
		"""
		Replace the allowed set in the last Box on the chain.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			values (set):
				A set of values to use as the allowed set.
		"""
		self.chain[-1].replace_allowed_set(key, values)

	def set_bounds(self, key, lowerbound, upperbound=None):
		"""
		Set both lower and upper bounds in the last Box on the chain.

		Args:
			key (str):
				The feature name to which these bounds
				will be attached.
			lowerbound (numeric, None, or Bounds):
				The lower bound, or a Bounds object that gives
				upper and lower bounds (in which case the `upperbound`
				argument is ignored).  Set explicitly to 'None' to
				leave unbounded from below.
			upperbound (numeric or None, default None):
				The upper bound. Set to 'None' to
				leave unbounded from above.

		Raises:
			ScopeError:
				If a scope is attached to this box but the `key` cannot
				be found in the scope.

		"""
		self.chain[-1].set_bounds(key, lowerbound, upperbound)


def find_all_boxes_with_parent(universe:dict, parent=None):
	result = []
	for name, clusterdef in universe.items():
		if clusterdef.parent_box_name == parent:
			result.append(name)
	return result

def pseudoname_boxes(boxes, root=None):
	if root is None:
		try:
			fancy = [f"Scope: {boxes.scope.name}"]
		except AttributeError:
			fancy = [f"Boxes Universe"]
		plain = [None]
	else:
		fancy = []
		plain = []
	tops = sorted(find_all_boxes_with_parent(boxes, parent=root))
	for t in tops:
		fancy.append("▷ "+t if t[0] in "▶▷" else "▶ "+t)
		plain.append(t)
		f_, p_ = pseudoname_boxes(boxes, root=t)
		for f1, p1 in zip(f_, p_):
			fancy.append("▷ "+f1 if f1[0] in "▶▷" else "▶ "+f1)
			plain.append(p1)
	return fancy, plain

class Boxes(MutableMapping):

	def __init__(self, *args, scope=None, **kw):
		self._storage = dict()
		self._scope = scope
		if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
			args = args[0]
		for a in args:
			self.add(a)
		for k,v in kw.items():
			self[k] = v
		if scope is not None:
			for i in self._storage:
				self._storage[i].scope = scope
	@property
	def scope(self):
		return self._scope

	@scope.setter
	def scope(self, s):
		self._scope = s
		for i in self._storage:
			self._storage[i].scope = s

	def __getitem__(self, key):
		return self._storage[key]

	def __setitem__(self, key, value):
		if not isinstance(value, Box):
			raise TypeError(f"values must be Box not {type(value)}")
		if key != value.name:
			raise ValueError('key must match name, use Boxes.add(box)')
		self._storage[key] = value

	def add(self, value):
		if not isinstance(value, Box):
			raise TypeError(f"values must be Box not {type(value)}")
		self[value.name] = value

	def __delitem__(self, key):
		del self._storage[key]

	def __iter__(self):
		return iter(self._storage)

	def __len__(self):
		return len(self._storage)

	def plain_names(self):
		return pseudoname_boxes(self, root=None)[1]

	def fancy_names(self):
		return pseudoname_boxes(self, root=None)[0]

	def both_names(self):
		return pseudoname_boxes(self, root=None)

	def get_chain(self, name):
		return ChainedBox(self, name)

