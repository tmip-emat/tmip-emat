import numpy
import pandas
import warnings
import functools
from ...viz import colors
from ...scope.box import GenericBox
from traitlets import TraitError

from plotly import graph_objs as go

from ipywidgets import Dropdown
import ipywidgets as widget

import logging
_logger = logging.getLogger('EMAT.widget')

from .explore_base import DataFrameExplorer


def _deselect_all_points(trace):
	trace.selectedpoints = None


# def _debugprint(s):
# 	print(s.replace("rgb(255, 127, 14)", "<ORANGE>").replace("rgb(255, 46, 241)","<PINK>"))


from .components import *

class Visualizer(DataFrameExplorer):

	def __init__(
			self,
			data,
			selections=None,
			scope=None,
			active_selection_name=None,
			reference_point=None,
	):
		if selections is None:
			from ...scope.box import Box
			selections = {'Explore': Box(name='Explore', scope=scope)}
			if active_selection_name is None:
				active_selection_name = 'Explore'

		super().__init__(
			data,
			selections=selections,
			active_selection_name=active_selection_name,
			reference_point=reference_point,
		)
		self.scope = scope
		self._figures_hist = {}
		self._figures_freq = {}
		self._base_histogram = {}
		self._categorical_data = {}
		self._freeze = False
		self._two_way = {}
		self._splom = {}
		self._hmm = {}

		self._status_txt = widget.HTML(
			value="<i>Explore Status Not Set</i>",
		)
		self._status_pie = go.FigureWidget(
			go.Pie(
				values=[75, 250],
				labels=['Inside', 'Outside'],
				hoverinfo='label+value',
				textinfo='percent',
				textfont_size=10,
				marker=dict(
					colors=[
						self.active_selection_color(),
						colors.DEFAULT_BASE_COLOR,
					],
					line=dict(color='#FFF', width=0.25),
				)
			),
			layout=dict(
				width=100,
				height=100,
				showlegend=False,
				margin=dict(l=10, r=10, t=10, b=10),
			)
		)
		self._status = widget.HBox(
			[
				widget.VBox([self._active_selection_chooser, self._status_txt]),
				self._status_pie
			],
			layout=dict(
				justify_content = 'space-between',
				align_items = 'center',
			)
		)
		self._update_status()

	def get_histogram_figure(self, col, bins=20, marker_line_width=None):
		try:
			this_type = self.scope.get_dtype(col)
		except:
			this_type = 'float'
		if this_type in ('cat','bool'):
			return self.get_frequency_figure(col)
		if this_type in ('int',):
			param = self.scope[col]
			if param.max - param.min + 1 <= bins * 4:
				bins = param.max - param.min + 1
				if marker_line_width is None:
					marker_line_width = 0
		self._create_histogram_figure(col, bins=bins, marker_line_width=marker_line_width)
		return self._figures_hist[col]

	def get_frequency_figure(self, col):
		if self.scope.get_dtype(col) == 'cat':
			labels = self.scope.get_cat_values(col)
		else:
			labels = [False, True]
		self._create_frequencies_figure(col, labels=labels)
		return self._figures_freq[col]

	def _create_histogram_figure(self, col, bins=20, *, marker_line_width=None):
		if col in self._figures_hist:
			self._update_histogram_figure(col)
		else:
			selection = self.active_selection()
			if self.active_selection_deftype() == 'box':
				box = self._selection_defs[self.active_selection_name()]
			else:
				box = None
			fig = new_histogram_figure(
				selection, self.data[col], bins,
				marker_line_width=marker_line_width,
				on_deselect=lambda *a: self._on_deselect_from_histogram(*a,name=col),
				on_select=lambda *a: self._on_select_from_histogram(*a,name=col),
				box=box,
				title_text=self.scope.shortname(col),
				ref_point=self.reference_point(col),
			)
			self._figures_hist[col] = fig

	def _create_frequencies_figure(self, col, labels=None, *, marker_line_width=None):
		if col in self._figures_freq:
			self._update_frequencies_figure(col)
		else:
			selection = self.active_selection()
			if self.active_selection_deftype() == 'box':
				box = self._selection_defs[self.active_selection_name()]
			else:
				box = None
			fig = new_frequencies_figure(
				selection, self.data[col], labels,
				marker_line_width=marker_line_width,
				on_deselect=functools.partial(self._on_deselect_from_histogram, name=col),
				on_select=functools.partial(self._on_select_from_freq, name=col),
				#on_click=functools.partial(self._on_click_from_frequencies, name=col), # not always stable
				box=box,
				title_text=self.scope.shortname(col),
				ref_point=self.reference_point(col),
			)
			self._figures_freq[col] = fig

	def _update_histogram_figure(self, col):
		if col in self._figures_hist:
			fig = self._figures_hist[col]
			if self.active_selection_deftype() == 'box':
				box = self._selection_defs[self.active_selection_name()]
			else:
				box = None
			with fig.batch_update():
				update_histogram_figure(
					fig,
					self.active_selection(),
					self.data[col],
					box=box,
					ref_point=self.reference_point(col),
				)

	def _update_frequencies_figure(self, col):
		if col in self._figures_freq:
			fig = self._figures_freq[col]
			if self.active_selection_deftype() == 'box':
				box = self._selection_defs[self.active_selection_name()]
			else:
				box = None
			with fig.batch_update():
				update_frequencies_figure(
					fig,
					self.active_selection(),
					self.data[col],
					box=box,
					ref_point=self.reference_point(col),
				)

	def _compute_histogram(self, col, selection, bins=None):
		if col not in self._base_histogram:
			if bins is None:
				bins = 20
			bar_heights, bar_x = numpy.histogram(self.data[col], bins=bins)
			self._base_histogram[col] = bar_heights, bar_x
		else:
			bar_heights, bar_x = self._base_histogram[col]
		bins_left = bar_x[:-1]
		bins_width = bar_x[1:] - bar_x[:-1]
		bar_heights_select, bar_x = numpy.histogram(self.data[col][selection], bins=bar_x)
		return bar_heights, bar_heights_select, bins_left, bins_width

	def _compute_frequencies(self, col, selection, labels):
		if col in self._categorical_data:
			v = self._categorical_data[col]
		else:
			self._categorical_data[col] = v = self.data[col].astype(
				pandas.CategoricalDtype(categories=labels, ordered=False)
			).cat.codes
		if col not in self._base_histogram:
			bar_heights, bar_x = numpy.histogram(v, bins=numpy.arange(0, len(labels) + 1))
			self._base_histogram[col] = bar_heights, bar_x
		else:
			bar_heights, bar_x = self._base_histogram[col]
		bar_heights_select, _ = numpy.histogram(v[selection], bins=numpy.arange(0, len(labels) + 1))
		return bar_heights, bar_heights_select, labels

	def _on_select_from_histogram(self, *args, name=None):
		if self._freeze:
			return
		try:
			self._freeze = True
			select_min, select_max = args[2].xrange
			_logger.debug("name: %s  range: %f - %f", name, select_min, select_max)
			self._figures_hist[name].for_each_trace(_deselect_all_points)

			if self.active_selection_deftype() == 'box':
				box = self._selection_defs[self.active_selection_name()]
				box = interpret_histogram_selection(name, args[2].xrange, box, self.data, self.scope)
				self.new_selection(box, name=self.active_selection_name())
				self._active_selection_changed()
		except:
			_logger.exception("error in _on_select_from_histogram")
			raise
		finally:
			self._freeze = False

	def _on_deselect_from_histogram(self, *args, name=None):
		_logger.debug("deselect %s", name)
		if self.active_selection_deftype() == 'box':
			box = self._selection_defs[self.active_selection_name()]
			if name in box:
				del box[name]
				self.new_selection(box, name=self.active_selection_name())
				self._active_selection_changed()


	def _on_select_from_freq(self, *args, name=None):
		select_min, select_max = args[2].xrange
		select_min = int(numpy.ceil(select_min))
		select_max = int(numpy.ceil(select_max))

		fig = self.get_figure(name)

		toggles = fig.layout['meta']['x_tick_values'][select_min:select_max]
		fig.for_each_trace(_deselect_all_points)

		if self.active_selection_deftype() == 'box':
			box = self._selection_defs[self.active_selection_name()]
			box.scope = self.scope

			if name not in box:
				for x in toggles:
					box.add_to_allowed_set(name, x)
			else:
				for x in toggles:
					if name not in box or x in box[name]:
						box.remove_from_allowed_set(name, x)
						if len(box[name]) == 0:
							del box[name]
					else:
						box.add_to_allowed_set(name, x)
			if toggles:
				self.new_selection(box, name=self.active_selection_name())
				self._active_selection_changed()


	def _on_click_from_frequencies(self, *args, name=None):
		x = None
		if len(args) >= 2:
			xs = getattr(args[1],'xs',None)
			if xs:
				x = xs[0]
		if x is not None:
			if self.active_selection_deftype() == 'box':
				box = self._selection_defs[self.active_selection_name()]
				box.scope = self.scope
				if name not in box or x in box[name]:
					box.remove_from_allowed_set(name, x)
					if len(box[name]) == 0:
						del box[name]
				else:
					box.add_to_allowed_set(name, x)
				self.new_selection(box, name=self.active_selection_name())
				self._active_selection_changed()

	def _active_selection_changed(self):
		if hasattr(self, '_active_selection_changing_'):
			return # prevent recursive looping
		try:
			self._active_selection_changing_ = True
			with self._status_pie.batch_update():
				super()._active_selection_changed()
				self._update_status()
				for col in self._figures_hist:
					self._update_histogram_figure(col)
				for col in self._figures_freq:
					self._update_frequencies_figure(col)
				for key in self._two_way:
					self._two_way[key].refresh_selection_names()
					self._two_way[key]._on_change_selection_choose(payload={
						'new':self.active_selection_name(),
					})
				self._update_sploms()
				self._update_hmms()
		finally:
			del self._active_selection_changing_

	def status(self):
		return self._status

	def _update_status(self):
		text = '<span style="font-weight:bold;font-size:150%">{:,d} Cases Selected out of {:,d} Total Cases</span>'
		selection = self.active_selection()
		values = (int(numpy.sum(selection)), int(selection.size))
		self._status_txt.value = text.format(*values)
		self._status_pie.data[0].values = [values[0], values[1]-values[0]]



	def get_figure(self, col):
		if col in self._figures_hist:
			return self._figures_hist[col]
		if col in self._figures_freq:
			return self._figures_freq[col]
		return None

	def _clear_boxes_on_figure(self, col):
		fig = self.get_figure(col)
		if fig is None: return

		foreground_shapes = []
		refpoint = self.reference_point(col)
		if refpoint is not None:
			if refpoint in (True, False):
				refpoint = str(refpoint).lower()
			_y_max = sum(t.y for t in fig.select_traces()).max()
			y_range = (
				-_y_max * 0.02,
				_y_max * 1.04,
			)
			foreground_shapes.append(
				go.layout.Shape(
					type="line",
					xref="x1",
					yref="y1",
					x0=refpoint,
					y0=y_range[0],
					x1=refpoint,
					y1=y_range[1],
					**colors.DEFAULT_REF_LINE_STYLE,
				)
			)

		fig.layout.shapes= foreground_shapes
		fig.layout.title.font.color = 'black'
		fig.layout.title.text = col

	# def _draw_boxes_on_figure(self, col):
	#
	# 	if self.active_selection_deftype() != 'box':
	# 		self._clear_boxes_on_figure(col)
	# 		return
	#
	# 	fig = self.get_figure(col)
	# 	if fig is None: return
	# 	box = self._selection_defs[self.active_selection_name()]
	# 	if box is None:
	# 		self._clear_boxes_on_figure(col)
	# 		return
	#
	# 	from ...scope.box import Bounds
	#
	# 	if col in box.thresholds:
	# 		x_lo, x_hi = None, None
	# 		thresh = box.thresholds.get(col)
	# 		if isinstance(thresh, Bounds):
	# 			x_lo, x_hi = thresh
	# 		if isinstance(thresh, set):
	# 			x_lo, x_hi = [], []
	# 			for tickval, ticktext in enumerate(fig.data[0].x):
	# 				if ticktext in thresh:
	# 					x_lo.append(tickval-0.45)
	# 					x_hi.append(tickval+0.45)
	#
	# 		try:
	# 			x_range = (
	# 				fig.data[0].x[0] - (fig.data[0].width[0] / 2),
	# 				fig.data[0].x[-1] + (fig.data[0].width[-1] / 2),
	# 			)
	# 		except TypeError:
	# 			x_range = (
	# 				-0.5,
	# 				len(fig.data[0].x)+0.5
	# 			)
	# 		x_width = x_range[1] - x_range[0]
	# 		if x_lo is None:
	# 			x_lo = x_range[0]-x_width * 0.02
	# 		if x_hi is None:
	# 			x_hi = x_range[1]+x_width * 0.02
	# 		if not isinstance(x_lo, list):
	# 			x_lo = [x_lo]
	# 		if not isinstance(x_hi, list):
	# 			x_hi = [x_hi]
	#
	# 		y_lo, y_hi = None, None
	# 		_y_max = sum(t.y for t in fig.select_traces()).max()
	# 		y_range = (
	# 			-_y_max * 0.02,
	# 			_y_max * 1.04,
	# 		)
	# 		y_width = y_range[1] - y_range[0]
	# 		if y_lo is None:
	# 			y_lo = y_range[0]-y_width * 0
	# 		if y_hi is None:
	# 			y_hi = y_range[1]+y_width * 0
	# 		if not isinstance(y_lo, list):
	# 			y_lo = [y_lo]
	# 		if not isinstance(y_hi, list):
	# 			y_hi = [y_hi]
	#
	# 		x_pairs = list(zip(x_lo, x_hi))
	# 		y_pairs = list(zip(y_lo, y_hi))
	#
	# 		background_shapes = [
	# 			# Rectangle background color
	# 			go.layout.Shape(
	# 				type="rect",
	# 				xref="x1",
	# 				yref="y1",
	# 				x0=x_pair[0],
	# 				y0=y_pair[0],
	# 				x1=x_pair[1],
	# 				y1=y_pair[1],
	# 				line=dict(
	# 					width=0,
	# 				),
	# 				fillcolor=colors.DEFAULT_BOX_BG_COLOR,
	# 				opacity=0.2,
	# 				layer="below",
	# 			)
	# 			for x_pair in x_pairs
	# 			for y_pair in y_pairs
	# 		]
	#
	# 		foreground_shapes = [
	# 			# Rectangle reference to the axes
	# 			go.layout.Shape(
	# 				type="rect",
	# 				xref="x1",
	# 				yref="y1",
	# 				x0=x_pair[0],
	# 				y0=y_pair[0],
	# 				x1=x_pair[1],
	# 				y1=y_pair[1],
	# 				line=dict(
	# 					width=2,
	# 					color=colors.DEFAULT_BOX_LINE_COLOR,
	# 				),
	# 				fillcolor='rgba(0,0,0,0)',
	# 				opacity=1.0,
	# 			)
	# 			for x_pair in x_pairs
	# 			for y_pair in y_pairs
	# 		]
	#
	# 		refpoint = self.reference_point(col)
	# 		if refpoint is not None:
	# 			if refpoint in (True, False):
	# 				refpoint = str(refpoint).lower()
	# 			foreground_shapes.append(
	# 				go.layout.Shape(
	# 					type="line",
	# 					xref="x1",
	# 					yref="y1",
	# 					x0=refpoint,
	# 					y0=y_range[0],
	# 					x1=refpoint,
	# 					y1=y_range[1],
	# 					**colors.DEFAULT_REF_LINE_STYLE,
	# 				)
	# 			)
	#
	# 		fig.layout.shapes=background_shapes+foreground_shapes
	# 		fig.layout.title.font.color = colors.DEFAULT_BOX_LINE_COLOR
	# 		fig.layout.title.text = f'<b>{col}</b>'
	# 	else:
	# 		self._clear_boxes_on_figure(col)


	def _get_widgets(self, *include):

		if self.scope is None:
			raise ValueError('cannot create visualization with no scope')

		viz_widgets = []
		for i in include:
			if i not in self.scope:
				warnings.warn(f'{i} not in scope')
			elif i not in self.data.columns:
				warnings.warn(f'{i} not in data')
			else:
				fig = self.get_histogram_figure(i)
				if fig is not None:
					viz_widgets.append(fig)

		return widget.Box(viz_widgets, layout=widget.Layout(flex_flow='row wrap'))

	def uncertainty_selectors(self, style='hist'):
		return self._get_widgets(*self.scope.get_uncertainty_names())

	def lever_selectors(self, style='hist'):
		return self._get_widgets(*self.scope.get_lever_names())

	def measure_selectors(self, style='hist'):
		return self._get_widgets(*self.scope.get_measure_names())

	def complete(self, measure_style='hist'):
		return widget.VBox([
			self.status(),
			widget.HTML("<h3>Policy Levers</h3>"),
			self.lever_selectors(),
			widget.HTML("<h3>Exogenous Uncertainties</h3>"),
			self.uncertainty_selectors(),
			widget.HTML("<h3>Performance Measures</h3>"),
			#self._measure_notes(style=measure_style),
			self.measure_selectors(),
		])

	def set_active_selection_color(self, color):
		super().set_active_selection_color(color)
		for col, fig in self._figures_freq.items():
			fig.data[0].marker.color = color
		for col, fig in self._figures_hist.items():
			fig.data[0].marker.color = color
		c = self._status_pie.data[0].marker.colors
		self._status_pie.data[0].marker.colors = [color, c[1]]
		for k, twoway in self._two_way.items():
			#_debugprint(f"twoway[{self._active_selection_name}][{k}] to {color}")
			twoway.change_selection_color(color)

	def refresh_selection_names(self):
		super().refresh_selection_names()
		try:
			_two_way = self._two_way
		except AttributeError:
			pass
		else:
			for k, twoway in _two_way.items():
				twoway.refresh_selection_names()

	def two_way(
			self,
			key=None,
			reset=False,
			*,
			x=None,
			y=None,
			use_gl=True,
	):
		if key is None and (x is not None or y is not None):
			key = (x,y)

		if key in self._two_way and not reset:
			return self._two_way[key]

		from .twoway import TwoWayFigure
		self._two_way[key] = TwoWayFigure(self, use_gl=use_gl)
		self._two_way[key].selection_choose.value = self.active_selection_name()

		def _try_set_value(where, value, describe):
			if value is not None:
				try:
					where.value = value
				except TraitError:
					warnings.warn(f'"{value}" is not a valid value for {describe}')

		_try_set_value(self._two_way[key].x_axis_choose, x, 'the x axis dimension')
		_try_set_value(self._two_way[key].y_axis_choose, y, 'the y axis dimension')
		return self._two_way[key]

	def splom(
			self,
			key=None,
			reset=False,
			*,
			cols='M',
			rows='L',
			use_gl=True,
	):
		if not isinstance(rows, str):
			rows = tuple(rows)
		if not isinstance(cols, str):
			cols = tuple(cols)

		if key is None and (cols is not None or rows is not None):
			key = (cols,rows)

		if key in self._splom and not reset:
			return self._splom[key]

		box = None
		if self.active_selection_deftype() == 'box':
			name = self.active_selection_name()
			box = self._selection_defs[name]

		self._splom[key] = new_splom_figure(
			self.scope,
			self.data,
			rows=rows,
			cols=cols,
			use_gl=use_gl,
			mass=250,
			row_titles='side',
			size=150,
			selection=self.active_selection(),
			box=box,
			refpoint=self._reference_point,
			figure_class=go.FigureWidget,
			on_select=functools.partial(self._on_select_from_splom, name=key),
		)

		return self._splom[key]

	def _on_select_from_splom(self, row, col, trace, points, selection, name=None):
		# if len(points.point_inds)==0:
		# 	return
		# print("name=",name)
		# print(row, col, "->", selection)
		# print( "->", selection.xrange)
		# print( "->", selection.yrange)
		# print( "->", type(selection.yrange))
		# trace.selectedpoints = None
		pass

	def _update_sploms(self):
		box = None
		if self.active_selection_deftype() == 'box':
			name = self.active_selection_name()
			box = self._selection_defs[name]
		for fig in self._splom.values():
			with fig.batch_update():
				update_splom_figure(
					self.scope,
					self.data,
					fig,
					self.active_selection(),
					box,
					mass=None,
					selected_color=self.active_selection_color(),
				)

	def hmm(
			self,
			key=None,
			reset=False,
			*,
			cols='M',
			rows='L',
			emph_selected=True,
			show_points=-1,
			size=150,
	):
		if not isinstance(rows, str):
			rows = tuple(rows)
		if not isinstance(cols, str):
			cols = tuple(cols)

		if key is None and (cols is not None or rows is not None):
			key = (cols,rows)

		if key in self._hmm and not reset:
			return self._hmm[key]

		box = None
		if self.active_selection_deftype() == 'box':
			name = self.active_selection_name()
			box = self._selection_defs[name]

		self._hmm[key] = new_hmm_figure(
			self.scope,
			self.data,
			rows=rows,
			cols=cols,
			row_titles='side',
			size=size,
			selection=self.active_selection(),
			box=box,
			refpoint=self._reference_point,
			figure_class=go.FigureWidget,
			emph_selected=emph_selected,
			show_points=show_points,
		)

		return self._hmm[key]

	def _update_hmms(self):
		box = None
		if self.active_selection_deftype() == 'box':
			name = self.active_selection_name()
			box = self._selection_defs[name]
		for fig in self._hmm.values():
			with fig.batch_update():
				update_hmm_figure(
					self.scope,
					self.data,
					fig,
					self.active_selection(),
					box,
				)


	def __setitem__(self, key, value):
		if not isinstance(key, str):
			raise TypeError(f'selection names must be str not {type(key)}')
		color = None
		if value is None:
			from ...scope.box import Box
			value = Box(name=key, scope=self.scope)
		if isinstance(value, GenericBox):
			color = colors.DEFAULT_HIGHLIGHT_COLOR
		elif isinstance(value, str):
			color = colors.DEFAULT_EXPRESSION_COLOR
		elif isinstance(value, pandas.Series):
			color = colors.DEFAULT_LASSO_COLOR
		self.new_selection(value, name=key, color=color)

	def __getitem__(self, item):
		if item not in self.selection_names():
			return KeyError(item)
		return self._selection_defs.get(item, None)

	def prim(self, data='parameters', target=None, threshold=0.2, **kwargs):

		from .prim import Prim

		if target is None:
			of_interest = self.active_selection()
		elif isinstance(target, str):
			of_interest = self._selections[target]
		else:
			raise ValueError("must give a target")

		if data == 'parameters':
			data_ = self.data[self.scope.get_parameter_names()]
		elif data == 'levers':
			data_ = self.data[self.scope.get_lever_names()]
		elif data == 'uncertainties':
			data_ = self.data[self.scope.get_uncertainty_names()]
		elif data == 'measures':
			data_ = self.data[self.scope.get_measure_names()]
		elif data == 'all':
			data_ = self.data
		else:
			data_ = self.data[data]

		self._prim_target = of_interest

		if (of_interest).all():
			raise ValueError("all points are in the target, cannot run PRIM")
		if (~of_interest).all():
			raise ValueError("no points are in the target, cannot run PRIM")

		result = Prim(
			data_,
			of_interest,
			threshold=threshold,
			**kwargs,
		)

		result._explorer = self

		return result


	def clear_box(self, name=None):
		if name is None:
			name = self.active_selection_name()
		if self.selection_deftype(name) == 'box':
			box = self._selection_defs[self.active_selection_name()]
			box.clear()
			self[name] = box

	def new_box(self, name, **kwargs):
		from ...scope.box import Box
		scope = kwargs.pop('scope', self.scope)
		activate = kwargs.pop('activate', True)
		self.new_selection(
			Box(name, scope=scope, **kwargs),
			name=name,
			color=colors.DEFAULT_HIGHLIGHT_COLOR,
			activate=activate,
		)
