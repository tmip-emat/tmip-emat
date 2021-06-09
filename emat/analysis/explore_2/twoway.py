
import numpy
import pandas
import plotly.graph_objs as go
from plotly.callbacks import BoxSelector, LassoSelector
from ipywidgets import HBox, VBox, Dropdown, Label, Text, Output
from ...viz import colors
from ...util.naming import clean_name, multiindex_to_strings
from .explore_visualizer import Visualizer
from .menu import Menu

import logging
_logger = logging.getLogger('EMAT.widget')

# def _debugprint(s):
# 	print(s.replace("rgb(255, 127, 14)", "<ORANGE>").replace("rgb(255, 46, 241)","<PINK>"))

class BaseTwoWayFigure():

	def __init__(
			self,
			viz,
	):
		assert isinstance(viz, Visualizer)
		self._dfv = viz
		self._alt_selections = {}


	@property
	def scope(self):
		return self._dfv.scope


	@property
	def selection(self):
		return self._dfv.active_selection()


	def _get_sectional_names(self, columns):
		scope = self.scope
		uncs = scope.get_uncertainty_names()
		levs = scope.get_lever_names()
		cons = scope.get_constant_names()
		meas = scope.get_measure_names()

		c_uncs = []
		c_levs = []
		c_cons = []
		c_meas = []
		c_other = []

		for c in columns:
			if c in uncs:
				c_uncs.append(c)
			elif c in levs:
				c_levs.append(c)
			elif c in cons:
				c_cons.append(c)
			elif c in meas:
				c_meas.append(c)
			else:
				c_other.append(c)

		result = []
		if len(c_uncs):
			result.append("-- (X) Uncertainties --")
			result.extend(c_uncs)
		if len(c_levs):
			result.append("-- (L) Levers --")
			result.extend(c_levs)
		if len(c_cons):
			result.append("-- Constants --")
			result.extend(c_cons)
		if len(c_meas):
			result.append("-- (M) Measures --")
			result.extend(c_meas)
		if len(c_other):
			result.append("-- Other --")
			result.extend(c_other)

		return result


	def _get_shortname(self, name):
		if self.scope is None:
			return name
		return self.scope.shortname(name)

	def _manage_categorical(self, x, perturb=True):
		try:
			valid_scales = ['linear']
			x_categories = None
			x_ticktext = None
			x_tickvals = None
			if not isinstance(x, pandas.Series):
				x = pandas.Series(x)
			try:
				if numpy.issubdtype(x.dtype, numpy.bool_):
					x = x.astype('category')
			except TypeError:
				pass
			if isinstance(x.dtype, pandas.CategoricalDtype):
				x_categories = x.cat.categories
				codes = x.cat.codes
				x = codes.astype(float)
				s_ = x.size * 0.01
				s_ = s_ / (1 + s_)
				epsilon = 0.05 + 0.20 * s_
				if perturb:
					x = x + numpy.random.uniform(-epsilon, epsilon, size=x.shape)
				x_tickmode = 'array'
				x_ticktext = list(x_categories)
				x_tickvals = list(range(len(x_ticktext)))
			else:
				if x.min() >= 0:
					valid_scales.append('log')

			return x, x_ticktext, x_tickvals, valid_scales
		except:
			_logger.exception('ERROR IN _manage_categorical')
			raise

class TwoWayFigure(HBox, BaseTwoWayFigure):
	"""
	The two-way visualizer widget.

	This class encapsulates a TMIP-EMAT interactive widget
	that displays a configurable two dimensional scatter
	plot of Visualizer data.  Dropdown menus allow the user
	to change the data dimensions displayed on the x and
	y axes, as well as create and manipulate new selection
	sets directly from the interactive widget.

	Args:
		viz (emat.Visualizer):
			The visualizer object to link for this two-way.
		target_marker_opacity (numeric, default 1000):
			The number of scatter point markers to display
			fully opaque.  If the number of markers displayed
			exceeds this value, each marker is rendered
			partially transparent, such that the total marker
			weight (opacity * number of markers) is loosely
			approximate to this total value.
		minimum_marker_opacity (float, default 0.25):
			This is the minimum marker opacity used,
			notwithstanding any transparency level implied
			by `target_marker_opacity`.
		use_gl (bool, default True):
			Use Plotly's `Scattergl` instead of `Scatter`, which
			may provide some performance benefit for large data
			sets.
	"""

	def __init__(
			self,
			viz,
			target_marker_opacity=1000,
			minimum_marker_opacity=0.25,
			use_gl=True,
	):
		BaseTwoWayFigure.__init__(self, viz)

		axis_choices = self._get_sectional_names(self._dfv.data.columns)

		self.x_axis_choose = Dropdown(
			options=axis_choices,
			# description='X Axis',
			value=self._dfv.data.columns[0],
		)

		self.x_axis_scale = Dropdown(
			options=['linear' ,],
			value='linear',
		)

		self.y_axis_choose = Dropdown(
			options=axis_choices,
			# description='Y Axis',
			value=self._dfv.data.columns[-1],
		)

		self.y_axis_scale = Dropdown(
			options=['linear' ,],
			value='linear',
		)

		self.selection_choose = Dropdown(
			options=list(self._dfv.selection_names()) + ['Expr'] + list(self._alt_selections.keys()),
			value='None',
		)

		self.selection_edit_menu = Menu(
			"Edit Selection...",
			{
				"Use Manual Selection":self._use_manual_selection,
			}
		)

		self.selection_expr = Text(
			value='True',
			disabled=True,
		)

		self.selection_name = Text(
			value='',
			disabled=True,
		)

		self.axis_choose = VBox(
			[
				Label("X Axis"),
				self.x_axis_choose,
				self.x_axis_scale,
				Label("Y Axis"),
				self.y_axis_choose,
				self.y_axis_scale,
				Label("Selection"),
				self._dfv._active_selection_chooser,
				self.selection_edit_menu,
				self.selection_name,
			],
			layout=dict(
				overflow='hidden',
			)
		)

		self.minimum_marker_opacity = minimum_marker_opacity
		self.target_marker_opacity = target_marker_opacity
		marker_opacity = self._compute_marker_opacity()

		self._x_data_range = [0 ,1]
		self._y_data_range = [0 ,1]

		Scatter = go.Scattergl if use_gl else go.Scatter
		self.scattergraph = Scatter(
			x=None,
			y=None,
			mode = 'markers',
			marker=dict(
				opacity=marker_opacity[0],
				color=None,
				colorscale=[[0, colors.DEFAULT_BASE_COLOR], [1, colors.DEFAULT_HIGHLIGHT_COLOR]],
				cmin=0,
				cmax=1,
			),
			name='Cases',
		)

		self.x_hist = go.Histogram(
			x=None,
			# name='x density',
			marker=dict(
				color=colors.DEFAULT_BASE_COLOR,
				# opacity=0.7,
			),
			yaxis='y2',
			bingroup='xxx',
		)

		self.y_hist = go.Histogram(
			y=None,
			# name='y density',
			marker=dict(
				color=colors.DEFAULT_BASE_COLOR,
				# opacity=0.7,
			),
			xaxis='x2',
			bingroup='yyy',
		)

		self.x_hist_s = go.Histogram(
			x=None,
			marker=dict(
				color=colors.DEFAULT_HIGHLIGHT_COLOR,
				# opacity=0.7,
			),
			yaxis='y2',
			bingroup='xxx',
		)

		self.y_hist_s = go.Histogram(
			y=None,
			marker=dict(
				color=colors.DEFAULT_HIGHLIGHT_COLOR,
				# opacity=0.7,
			),
			xaxis='x2',
			bingroup='yyy',
		)

		self.graph = go.FigureWidget()
		self.scattergraph = self.graph.add_trace(self.scattergraph).data[-1]
		self.x_hist = self.graph.add_trace(self.x_hist).data[-1]
		self.y_hist = self.graph.add_trace(self.y_hist).data[-1]
		self.x_hist_s = self.graph.add_trace(self.x_hist_s).data[-1]
		self.y_hist_s = self.graph.add_trace(self.y_hist_s).data[-1]

		self.graph.layout =dict(
			xaxis=dict(
				domain=[0, 0.85],
				showgrid=True,
				# title=self._df.data.columns[0],
			),
			yaxis=dict(
				domain=[0, 0.85],
				showgrid=True,
				# title=self._df.data.columns[-1],
			),

			xaxis2=dict(
				domain=[0.85, 1],
				showgrid=True,
				zeroline=True,
				zerolinecolor='#FFF',
				zerolinewidth=4,
			),
			yaxis2=dict(
				domain=[0.85, 1],
				showgrid=True,
				zeroline=True,
				zerolinecolor='#FFF',
				zerolinewidth=4,
			),

			barmode="overlay",
			showlegend=False,
			margin=dict(l=10, r=10, t=10, b=10),
			dragmode="lasso",
			plot_bgcolor=colors.DEFAULT_PLOT_BACKGROUND_COLOR,
		)

		self.x_axis_choose.observe(self._observe_change_column_x, names='value')
		self.y_axis_choose.observe(self._observe_change_column_y, names='value')

		self.x_axis_scale.observe(self._observe_change_scale_x, names='value')
		self.y_axis_scale.observe(self._observe_change_scale_y, names='value')

		self.selection_choose.observe(self._on_change_selection_choose, names='value')
		self.selection_expr.observe(self._on_change_selection_expr, names='value')

		self.set_x(self._dfv.data.columns[0], drawbox=False)
		self.set_y(self._dfv.data.columns[-1])

		self.scattergraph.meta = multiindex_to_strings(self._dfv.data.index)
		if self._dfv.data.index.name:
			hover_name = self._dfv.data.index.name
		else:
			hover_name = "Experiment"
		self.scattergraph.hovertemplate = f"%{{xaxis.title.text}}: %{{x}}<br>%{{yaxis.title.text}}: %{{y}}<extra>{hover_name} %{{meta}}</extra>"

		self.draw_box()

		super().__init__(
			[
				self.graph,
				self.axis_choose,
			],
			layout=dict(
				align_items='center',
			)
		)

		self.scattergraph.on_selection(self._on_select_points)
		self.out_logger = Output()
		self._update_state_ = set()


	def _compute_marker_opacity(self):
		# if self.selection is None:
		# 	marker_opacity = 1.0
		# 	if len(self._df.data) > self.target_marker_opacity:
		# 		marker_opacity = self.target_marker_opacity / len(self._dfv.data)
		# 	if marker_opacity < self.minimum_marker_opacity:
		# 		marker_opacity = self.minimum_marker_opacity
		# 	return marker_opacity, 1.0
		# else:
		marker_opacity = [1.0, 1.0]
		n_selected = int(self._dfv.active_selection().sum())
		n_unselect = len(self._dfv.data) - n_selected
		if n_unselect > self.target_marker_opacity:
			marker_opacity[0] = self.target_marker_opacity / n_unselect
		if marker_opacity[0] < self.minimum_marker_opacity:
			marker_opacity[0] = self.minimum_marker_opacity

		if n_selected > self.target_marker_opacity:
			marker_opacity[1] = self.target_marker_opacity / n_selected
		if marker_opacity[1] < self.minimum_marker_opacity:
			marker_opacity[1] = self.minimum_marker_opacity
		return marker_opacity

	@property
	def _x_data_width(self):
		w = self._x_data_range[1] - self._x_data_range[0]
		if w > 0:
			return w
		return 1

	@property
	def _y_data_width(self):
		w = self._y_data_range[1] - self._y_data_range[0]
		if w > 0:
			return w
		return 1

	def _observe_change_column_x(self, payload):
		if payload['new'][:3] == '-- ' and payload['new'][-3:] == ' --':
			# Just a heading, not a real option
			payload['owner'].value = payload['old']
			return
		self.set_x(payload['new'])

	def _observe_change_column_y(self, payload):
		if payload['new'][:3] == '-- ' and payload['new'][-3:] == ' --':
			# Just a heading, not a real option
			payload['owner'].value = payload['old']
			return
		self.set_y(payload['new'])

	def _observe_change_scale_x(self, payload):
		self.set_x(self.x_axis_choose.value)

	def _observe_change_scale_y(self, payload):
		self.set_y(self.y_axis_choose.value)


	def set_x(self, col, drawbox=True):
		"""
		Set the new X axis data.

		Args:
			col (str or array-like):
				The name of the new `x` column in `df`, or a
				computed array or pandas.Series of values.
		"""
		try:
			with self.graph.batch_update():
				this_label = None
				if isinstance(col, str):
					x = self._dfv.data[col]
					this_label = self._get_shortname(col)
				else:
					x = col
					try:
						this_label = self._get_shortname(col.name)
					except:
						pass

				x, x_ticktext, x_tickvals, x_scales = self._manage_categorical(x)
				self.x_axis_scale.options = x_scales
				self._x = x
				self._x_ticktext = x_ticktext or []
				self._x_tickvals = x_tickvals or []

				is_linear = (self.x_axis_scale.value == 'linear')
				is_log = (self.x_axis_scale.value == 'log')

				if this_label is not None:
					self.graph.layout.xaxis.title = this_label
					self.x_hist.name = this_label +' Unselected Density'
					self.x_hist_s.name = this_label +' Selected Density'
				else:
					self.graph.layout.xaxis.title = 'x axis'
					self.x_hist.name = 'Unselected Density'
					self.x_hist_s.name = 'Selected Density'

				# if self.selection is None:
				# 	self.scattergraph.x = x
				# 	self.x_hist.x = x if is_linear else None
				# 	self.x_hist_s.x = None
				# else:
				self.scattergraph.x = x
				self.x_hist.x = x if is_linear else None
				self.x_hist_s.x = x[self.selection] if is_linear else None
				if x_ticktext is not None:
					self._x_data_range = [x.min(), x.max()]
					self.graph.layout.xaxis.range = (
						self._x_data_range[0] - 0.3,
						self._x_data_range[1] + 0.3,
					)
					self.graph.layout.xaxis.tickmode = 'array'
					self.graph.layout.xaxis.ticktext = x_ticktext
					self.graph.layout.xaxis.tickvals = x_tickvals
					self.x_hist.xbins = dict(
						start=-0.25, end=self._x_data_range[1] + 0.25, size=0.5,
					)
					self.x_hist.autobinx = False
				else:
					self._x_data_range = [x.min(), x.max()]
					self.graph.layout.xaxis.range = (
						self._x_data_range[0] - self._x_data_width * 0.07,
						self._x_data_range[1] + self._x_data_width * 0.07,
					)
					self.graph.layout.xaxis.type = self.x_axis_scale.value
					self.graph.layout.xaxis.tickmode = None
					self.graph.layout.xaxis.ticktext = None
					self.graph.layout.xaxis.tickvals = None
					self.x_hist.xbins = None
					self.x_hist.autobinx = True
				if drawbox:
					self.draw_box()
		except:
			_logger.exception('ERROR IN DataFrameViewer.set_x')
			raise

	def set_y(self, col):
		"""
		Set the new Y axis data.

		Args:
			col (str or array-like):
				The name of the new `y` column in `df`, or a
				computed array or pandas.Series of values.
		"""
		with self.graph.batch_update():
			this_label = None
			if isinstance(col, str):
				y = self._dfv.data[col]
				this_label = self._get_shortname(col)
			else:
				y = col
				try:
					this_label = self._get_shortname(col.name)
				except:
					pass

			y, y_ticktext, y_tickvals, y_scales = self._manage_categorical(y)
			self.y_axis_scale.options = y_scales
			self._y = y
			self._y_ticktext = y_ticktext or []
			self._y_tickvals = y_tickvals or []

			is_linear = (self.y_axis_scale.value == 'linear')
			is_log = (self.y_axis_scale.value == 'log')

			if this_label is not None:
				self.graph.layout.yaxis.title = this_label
				self.y_hist.name = this_label +' Unselected Density'
				self.y_hist_s.name = this_label +' Selected Density'
			else:
				self.graph.layout.yaxis.title = 'y axis'
				self.y_hist.name = 'Unselected Density'
				self.y_hist_s.name = 'Selected Density'

			if self.selection is None:
				self.scattergraph.y = y
				self.y_hist.y = y if is_linear else None
				self.y_hist_s.y = None
			else:
				self.scattergraph.y = y
				self.y_hist.y = y if is_linear else None
				self.y_hist_s.y = y[self.selection] if is_linear else None
			if y_ticktext is not None:
				self._y_data_range = [y.min(), y.max()]
				self.graph.layout.yaxis.range = (
					self._y_data_range[0] - 0.3,
					self._y_data_range[1] + 0.3,
				)
				self.graph.layout.yaxis.tickmode = 'array'
				self.graph.layout.yaxis.ticktext = y_ticktext
				self.graph.layout.yaxis.tickvals = y_tickvals
				self.y_hist.ybins = dict(
					start=-0.25, end=self._y_data_range[1] + 0.25, size=0.5,
				)
				self.y_hist.autobiny = False
			else:
				self._y_data_range = [y.min(), y.max()]
				self.graph.layout.yaxis.range = (
					self._y_data_range[0] - self._y_data_width * 0.07,
					self._y_data_range[1] + self._y_data_width * 0.07,
				)
				self.graph.layout.yaxis.type = self.y_axis_scale.value
				self.graph.layout.yaxis.tickmode = None
				self.graph.layout.yaxis.ticktext = None
				self.graph.layout.yaxis.tickvals = None
				self.y_hist.ybins = None
				self.y_hist.autobiny = True

			self.draw_box()

	def change_selection_color(self, new_color=None):
		if new_color is None:
			new_color = colors.DEFAULT_HIGHLIGHT_COLOR
		with self.graph.batch_update():
			self.scattergraph.marker.colorscale = [[0, colors.DEFAULT_BASE_COLOR], [1, new_color]]
			self.x_hist_s.marker.color = new_color
			self.y_hist_s.marker.color = new_color

	def change_selection(self, new_selection):
		if new_selection is None:
			self.selection = None
			# Update Selected Portion of Scatters
			x = self._x
			y = self._y
			with self.graph.batch_update():
				self.scattergraph.x = x
				self.scattergraph.y = y
				self.scattergraph.marker.color = numpy.zeros(x.shape, numpy.int8)
				marker_opacity = self._compute_marker_opacity()
				self.scattergraph.marker.opacity = marker_opacity[0]
				# Update Selected Portion of Histograms
				self.x_hist_s.x = None
				self.y_hist_s.y = None
				self.draw_box()
			return

		if new_selection.size != len(self._dfv.data):
			raise ValueError(f"new selection size ({new_selection.size}) "
							 f"does not match length of data ({len(self._dfv.data)})")
		#self.selection = new_selection
		with self.graph.batch_update():
			# Update Selected Portion of Scatters
			x = self._x
			y = self._y
			self.scattergraph.x = x  # [~self.selection]
			self.scattergraph.y = y  # [~self.selection]
			selection_as_int = self.selection.astype(int)
			self.scattergraph.marker.color = selection_as_int
			marker_opacity = self._compute_marker_opacity()
			self.scattergraph.marker.opacity = [marker_opacity[j] for j in selection_as_int]
			# Update Selected Portion of Histograms
			self.x_hist_s.x = x[self.selection]
			self.y_hist_s.y = y[self.selection]
			self.draw_box()

	def draw_box(self):
		from ...scope.box import Bounds
		x_label = self.x_axis_choose.value
		y_label = self.y_axis_choose.value
		background_shapes, foreground_shapes = [], []

		box = None

		if self._dfv.active_selection_deftype() == 'box':
			box = self._dfv._selection_defs[self._dfv.active_selection_name()]
		elif self._dfv.active_selection_deftype() in ('primbox', 'cartbox'):
			box = self._dfv._selection_defs[self._dfv.active_selection_name()].to_emat_box()

		if box is not None:
			if x_label in box.thresholds or y_label in box.thresholds:
				x_lo, x_hi = None, None
				thresh = box.thresholds.get(x_label)
				if isinstance(thresh, Bounds):
					x_lo, x_hi = thresh
				if isinstance(thresh, set):
					x_lo, x_hi = [], []
					for tickval, ticktext in zip(self._x_tickvals, self._x_ticktext):
						if ticktext in thresh:
							x_lo.append(tickval -0.3)
							x_hi.append(tickval +0.3)
				if x_lo is None:
					x_lo = self._x.min( ) -self._x_data_width * 0.02
				if x_hi is None:
					x_hi = self._x.max( ) +self._x_data_width * 0.02
				if not isinstance(x_lo, list):
					x_lo = [x_lo]
				if not isinstance(x_hi, list):
					x_hi = [x_hi]

				y_lo, y_hi = None, None
				thresh = box.thresholds.get(y_label)
				if isinstance(thresh, Bounds):
					y_lo, y_hi = thresh
				if isinstance(thresh, set):
					y_lo, y_hi = [], []
					for tickval, ticktext in zip(self._y_tickvals, self._y_ticktext):
						if ticktext in thresh:
							y_lo.append(tickval -0.3)
							y_hi.append(tickval +0.3)
				if y_lo is None:
					y_lo = self._y.min( ) -self._y_data_width * 0.02
				if y_hi is None:
					y_hi = self._y.max( ) +self._y_data_width * 0.02
				if not isinstance(y_lo, list):
					y_lo = [y_lo]
				if not isinstance(y_hi, list):
					y_hi = [y_hi]

				x_pairs = list(zip(x_lo, x_hi))
				y_pairs = list(zip(y_lo, y_hi))

				background_shapes += [
					# Rectangle background color
					go.layout.Shape(
						type="rect",
						xref="x1",
						yref="y1",
						x0=x_pair[0],
						y0=y_pair[0],
						x1=x_pair[1],
						y1=y_pair[1],
						line=dict(
							width=0,
						),
						fillcolor=colors.DEFAULT_BOX_BG_COLOR,
						opacity=0.2,
						layer="below",
					)
					for x_pair in x_pairs
					for y_pair in y_pairs
				]

				foreground_shapes += [
					# Rectangle reference to the axes
					go.layout.Shape(
						type="rect",
						xref="x1",
						yref="y1",
						x0=x_pair[0],
						y0=y_pair[0],
						x1=x_pair[1],
						y1=y_pair[1],
						line=dict(
							width=1,
							color=colors.DEFAULT_BOX_LINE_COLOR,
						),
						fillcolor='rgba(0,0,0,0)',
						opacity=1.0,
					)
					for x_pair in x_pairs
					for y_pair in y_pairs
				]

		x_refpoint = self._dfv.reference_point(x_label)
		y_refpoint = self._dfv.reference_point(y_label)

		if x_refpoint is not None:
			_y_lo = self._y.min( ) -self._y_data_width * 0.02
			_y_hi = self._y.max( ) +self._y_data_width * 0.02
			foreground_shapes.append(
				go.layout.Shape(
					type="line",
					xref="x1",
					yref="y1",
					y0=_y_lo,
					x0=x_refpoint,
					y1=_y_hi,
					x1=x_refpoint,
					**colors.DEFAULT_REF_LINE_STYLE,
				)
			)

		if y_refpoint is not None:
			_x_lo = self._x.min( ) -self._x_data_width * 0.02
			_x_hi = self._x.max( ) +self._x_data_width * 0.02
			foreground_shapes.append(
				go.layout.Shape(
					type="line",
					xref="x1",
					yref="y1",
					x0=_x_lo,
					y0=y_refpoint,
					x1=_x_hi,
					y1=y_refpoint,
					**colors.DEFAULT_REF_LINE_STYLE,
				)
			)

		self.graph.layout.shapes = background_shapes +foreground_shapes

	def _selection_eval(self, txt):
		df = self._dfv.data.rename(columns={i: clean_name(i) for i in self._dfv.data.columns})
		return df.eval(txt).astype(bool)

	def _use_manual_selection(self):
		if 'Use Manual Target' not in self._alt_selections:
			return
		lasso_target = self._alt_selections['Use Manual Target']
		del self._alt_selections['Use Manual Target']
		lasso_select_name = self.selection_name.value
		self._dfv.new_selection(lasso_target, lasso_select_name, color=colors.DEFAULT_LASSO_COLOR, activate=True)
		self.refresh_selection_names()
		self.selection_name.disabled = True
		self.selection_name.value = ""
		self._clear_selection()

	def _on_change_selection_choose(self, payload):
		if "on_change_selection_choose" in self._update_state_: return
		if "refresh_selection_names" in self._update_state_: return
		self._update_state_.add("on_change_selection_choose")
		try:
			with self.graph.batch_update():
				if payload['new'] != 'Expr':
					self.selection_expr.disabled = True
				if payload['new'] == 'Expr':
					self.selection_expr.disabled = False
					self._on_change_selection_expr({'new' :self.selection_expr.value})
				elif payload['new'] in (self._dfv.selection_names()):
					self._dfv.set_active_selection_name(payload['new'])
					self.change_selection(self._dfv.active_selection())
				elif payload['new'] == 'Use Manual Target':
					lasso_target = self._alt_selections['Use Manual Target']
					del self._alt_selections['Use Manual Target']
					lasso_select_name = self.selection_name.value
					self._dfv.new_selection(lasso_target, lasso_select_name, color=colors.DEFAULT_LASSO_COLOR)
					self.add_alt_selection(lasso_select_name, None)
					self.selection_choose.value = lasso_select_name
					self.selection_name.disabled = True
					self.selection_name.value = ""
					self._clear_selection()
					self._dfv.set_active_selection_name(lasso_select_name)
					self.change_selection(self._dfv.active_selection())
				else:
					self.change_selection(self._alt_selections[payload['new']])
		finally:
			self._update_state_.discard("on_change_selection_choose")

	def _clear_selection(self):
		self.scattergraph.selectedpoints = None
		self.x_hist.selectedpoints = None
		self.x_hist_s.selectedpoints = None
		self.y_hist.selectedpoints = None
		self.y_hist_s.selectedpoints = None

	def _on_change_selection_expr(self, payload):
		expression = payload['new']
		try:
			sel = self._selection_eval(expression)
		except Exception as err:
			# print("FAILED ON EVAL\n",expression, err)
			pass
		else:
			try:
				self.change_selection(sel)
			except Exception as err:
				# print("FAILED ON SETTING\n", expression, err)
				pass
			else:
				# print("PASSED", expression)
				pass

	def add_alt_selection(self, key, value):
		"""
		Add a new possible alternative selection vector.

		Parameters
		----------
		key : str
			An identifier for the new selection vector.
		value : array-like of bool
			A vector of boolean values giving the selection.
		"""
		if value is not None:
			if value.size != len(self._dfv.data):
				raise ValueError(f"value size ({value.size}) "
								 f"does not match length of data ({len(self._dfv.data)})")
			self._alt_selections[key] = value
		self.refresh_selection_names()

	def lasso_selected_indexes(self):
		"""
		Get the index values of points selected interactively in the figure.

		Returns
		-------
		pandas.Index
		"""

		return self._dfv.data.index[list(self.scattergraph.selectedpoints)]

	def lasso_selected_data(self):
		"""
		Get the full data of points selected interactively in the figure.

		Returns
		-------
		pandas.DataFrame
		"""
		return self._dfv.data.loc[self.lasso_selected_indexes()]

	def lasso_selected(self):
		"""
		Get a boolean array indicating points selected interactively in the figure.

		Returns
		-------
		pandas.Series
		"""
		s = pandas.Series(data=False, index=self._dfv.data.index)
		s.iloc[list(self.scattergraph.selectedpoints)] = True
		return s

	def _on_select_points(self, trace, points, selector):
		# callback function for selection
		s = pandas.Series(data=False, index=self._dfv.data.index)
		for i in points.point_inds:
			s.iloc[i] = True
		# with self.out_logger:
		# 	print(points)
		# 	print(selector)
		self.add_alt_selection("Use Manual Target", s)
		self.selection_name.disabled = False
		if isinstance(selector, LassoSelector):
			default_name = "Lasso Target"
		else:
			default_name = "Rectangular Target"
		if self.selection_name.value == "":
			self.selection_name.value = default_name
		tentative_name = self.selection_name.value
		n=2
		while tentative_name in self._dfv.selection_names():
			tentative_name = f"{default_name} {n}"
			n += 1
		self.selection_name.value = tentative_name

	def refresh_selection_names(self):
		if "refresh_selection_names" in self._update_state_: return
		self._update_state_.add("refresh_selection_names")
		try:
			cache = self.selection_choose.value
			self.selection_choose.options = list(self._dfv.selection_names()) + ['Expr'] + list(self._alt_selections.keys())
			self.selection_choose.value = self._dfv.active_selection_name()
			try:
				self.selection_choose.value = cache
			except:
				pass
		finally:
			self._update_state_.discard("refresh_selection_names")