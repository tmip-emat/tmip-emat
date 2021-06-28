
import pandas
import numpy
import asyncio
import scipy.stats
import ipywidgets as widget
from plotly import graph_objects as go
from ..scope.scope import Scope
from ..model import AbstractCoreModel
from ..viz import colors
from .. import styles

DEFAULT_BACKGROUND = 100

_plotly_blue = '#1f77b4'
_plotly_orange = '#ff7f0e'
_plotly_green = '#2ca02c'
_plotly_red = '#d62728'
_plotly_purple = '#9467bd'

_left_top_color = _plotly_green
_right_bottom_color = _plotly_purple

class Timer:
	def __init__(self, timeout, callback):
		self._timeout = timeout
		self._callback = callback
		self._task = asyncio.ensure_future(self._job())

	async def _job(self):
		await asyncio.sleep(self._timeout)
		self._callback()

	def cancel(self):
		self._task.cancel()

def debounce(wait):
	""" Decorator that will postpone a function's
		execution until after `wait` seconds
		have elapsed since the last time it was invoked. """
	def decorator(fn):
		timer = None
		def debounced(*args, **kwargs):
			nonlocal timer
			def call_it():
				fn(*args, **kwargs)
			if timer is not None:
				timer.cancel()
			timer = Timer(wait, call_it)
		return debounced
	return decorator


class AB_Contrast():
	"""
	Contrast the outputs from two sets of particular inputs.

	Args:
		model (AbstractCoreModel): The model to use for testing.
		a, b (Mapping):
			These two dicts describe the contrasting inputs that
			are to be tested.  Each should give key-value pairs that
			set designated inputs to the indicated value.  Both `a`
			and `b` should have the same set of keys, and typically
			at least one value should differ between them. It is
			reasonable for a subset of keys to have the same value,
			for example to contrast two complete lever policies that
			differ only on the value for one particular lever.
		background (pandas.DataFrame or int):
			A design of experiments that is used to define the range
			of values for the unset parameters. For example, this can
			give a distribution of values for the uncertainties.
		test_name (str):
			A name to use for this test, in setting the `design_name`
			attribute of the results, and for use in database storage
			of experimental results.
		scope (emat.Scope, optional):
			Override the model.scope with a replacement.

	Returns:
		pandas.DataFrame or 2-tuple of pandas.DataFrame
	"""

	def __init__(self, model, a, b, background, test_name=None, scope=None):

		self.model = model
		self.scope = scope or model.scope
		self.a = a.copy()
		self.b = b.copy()

		if test_name is None:
			if hasattr(background, 'design_name'):
				test_name = f'{background.design_name}_ab_test'
			else:
				import uuid
				test_name = f'ab_test-{uuid.uuid4()}'

		self.test_name = test_name

		if isinstance(background, int):
			from ..experiment import experimental_design
			self.background = background = experimental_design.design_experiments(
				self.scope,
				n_samples=background,
				design_name=test_name,
			)

		self.design_a = background.copy()
		self.design_a.design_name = f"{test_name}_a"
		for k, v in a.items():
			self.design_a[k] = v

		self.design_b = background.copy()
		self.design_b.design_name = f"{test_name}_b"
		for k, v in b.items():
			self.design_b[k] = v

		results_a = self.model.run_experiments(self.design_a, db=False)
		results_b = self.model.run_experiments(self.design_b, db=False)

		measure_names = self.scope.get_measure_names()
		not_measures = [i for i in results_a.columns if i not in measure_names]
		self.results_a = results_a.drop(columns=not_measures)
		self.results_b = results_b.drop(columns=not_measures)

	def get_figure(self, measure, **kwargs):
		fig = create_violin(
			self.results_a[measure],
			self.results_b[measure],
			label=self.scope.shortname(measure),
			points=False,
			a_name=self.a_name(),
			b_name=self.b_name(),
			**kwargs,
		)
		return fig

	def a_name(self):
		name = getattr(self.a, 'name', 'A')
		if isinstance(name, str):
			return name
		else:
			return 'A'

	def b_name(self):
		name = getattr(self.b, 'name', 'B')
		if isinstance(name, str):
			return name
		else:
			return 'B'


def _compute_kde_1(x_a, bw_method=None):
	lo = numpy.min(x_a)
	hi = numpy.max(x_a)
	span = lo - (hi-lo)*0.07, hi + (hi-lo)*0.07
	kernel_base_a = scipy.stats.gaussian_kde(x_a, bw_method=bw_method)
	x_support = numpy.linspace(*span, 250)
	y_a = kernel_base_a(x_support)
	return x_support, y_a

def _compute_kde_2(x_a, x_b, bw_method=None):
	lo = min(numpy.min(x_a), numpy.min(x_b))
	hi = max(numpy.max(x_a), numpy.max(x_b))
	span = lo - (hi-lo)*0.07, hi + (hi-lo)*0.07
	kernel_base_a = scipy.stats.gaussian_kde(x_a, bw_method=bw_method)
	common_bw = kernel_base_a.covariance_factor()
	kernel_base_b = scipy.stats.gaussian_kde(x_b, bw_method=common_bw)
	x_support = numpy.linspace(*span, 250)
	y_a = kernel_base_a(x_support)
	y_b = kernel_base_b(x_support)
	return x_support, y_a, y_b


def create_kde_1_figure(x, bw_method=None):

	x_support, y_a = _compute_kde_1(x, bw_method=bw_method)
	fig = go.Figure(
		data=[
			go.Scatter(
				x=x_support,
				y=y_a,
				name='Overall',
				fill='tozeroy',
				marker_color=colors.DEFAULT_BASE_COLOR,
			),
		],
		layout=dict(
			showlegend=False,
			margin=dict(l=10, r=10, t=10, b=10),
			yaxis_showticklabels=False,
			**styles.figure_dims,
		),
	)
	fig._figure_kind = 'kde'
	return fig

def create_kde_2_figure(x_a, x_b, bw_method=None):

	x_support, y_a, y_b = _compute_kde_2(x_a, x_b, bw_method=bw_method)
	fig = go.Figure(
		data=[
			go.Scatter(
				x=x_support,
				y=y_a,
				name='Overall',
				fill='tozeroy',
				marker_color=colors.DEFAULT_BASE_COLOR,
			),
			go.Scatter(
				x=x_support,
				y=y_b,
				name='Inside',
				fill='tozeroy',  #fill='tonexty',
				marker_color=colors.DEFAULT_HIGHLIGHT_COLOR,
			),
		],
		layout=dict(
			showlegend=False,
			margin=dict(l=10, r=10, t=10, b=10),
			yaxis_showticklabels=False,
			**styles.figure_dims,
		),
	)
	fig._figure_kind = 'kde'
	return fig



def create_violin(
		x_a,
		x_b,
		label=None,
		points='suspectedoutliers',
		points_diff=False,
		a_name=None,
		b_name=None,
		showlegend=False,
		orientation='v',
		orientation_raw='v',
		orientation_diff='v',
		width=490,
		height=300,
):
	"""

	Args:
		x_a:
		x_b:
		label:
		points ({'all', 'outliers', 'suspectedoutliers', False}):
			Which points to display on the figure.
	Returns:
		Figure
	"""
	from plotly.subplots import make_subplots

	if label is None:
		label = getattr(x_a, 'name', getattr(x_b, 'name', 'Measure'))

	if a_name is None: a_name='A'
	if b_name is None: b_name='B'

	if orientation == 'h':
		fig = make_subplots(rows=2, cols=1, vertical_spacing=0.03, row_heights=[0.65, 0.35])
	elif orientation == 'v':
		fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02, column_widths=[0.55, 0.45])
	else:
		raise ValueError("orientation must be 'h' or 'v'")

	if orientation_raw is None:
		orientation_raw=orientation

	if orientation_diff is None:
		orientation_diff=orientation


	kwargs_common = dict(
		legendgroup='Raw',
		scalegroup='Raw',
		orientation=orientation_raw,
		box_visible=True,
		meanline_visible=True,
		hoveron='points',
		width=1,
	)
	if orientation_raw == 'h':
		kwargs_common['y0'] = 0
		kwargs_common['xaxis'] = 'x1'
	elif orientation_raw == 'v':
		kwargs_common['x0'] = 0
		kwargs_common['yaxis'] = 'y1'

	kwargs_a = dict(
		name=a_name,
		side='positive',
		line_color=_left_top_color,
		pointpos=0.5,
		**kwargs_common,
	)

	kwargs_b = dict(
		name=b_name,
		side='negative',
		line_color=_right_bottom_color,
		pointpos=-0.5,
		**kwargs_common,
	)

	if orientation_raw == 'h':
		kwargs_a['x'] = x_a
		kwargs_b['x'] = x_b
	elif orientation_raw == 'v':
		kwargs_a['y'] = x_a
		kwargs_b['y'] = x_b


	fig.add_trace(
		go.Violin(**kwargs_a),
		row=1, col=1,
	)
	fig.add_trace(
		go.Violin(**kwargs_b),
		row=1, col=1,
	)

	kwargs_diff = dict(
		legendgroup='Diff',
		scalegroup='Diff',
		name='Difference',
		line_color='red',
		box_visible=True,
		meanline_visible=True,
		pointpos=0.5,
		hoveron='points',
		side='positive',
		orientation=orientation_diff,
		width=1,
	)
	if orientation_diff == 'h':
		kwargs_diff['x'] = x_a - x_b
		kwargs_diff['y0'] = 1
		kwargs_diff['xaxis'] = 'x2'
	elif orientation_diff == 'v':
		kwargs_diff['y'] = x_a - x_b
		kwargs_diff['x0'] = 1
		kwargs_diff['yaxis'] = 'y2'

	if orientation == 'h':
		fig.add_trace(
			go.Violin(**kwargs_diff),
			row=2, col=1,
		)
	elif orientation == 'v':
		fig.add_trace(
			go.Violin(**kwargs_diff),
			row=1, col=2,
		)


	fig.update_traces(meanline_visible=True,
					  jitter=0.5,  # add some jitter on points for better visibility
					  scalemode='count')  # scale violin plot area with total count
	fig.data[0].points = points
	fig.data[1].points = points
	fig.data[2].points = points_diff

	fig.update_layout(
		violingap=0,
		violinmode='overlay',
		showlegend=showlegend,
		margin=styles.figure_margins,
		width=width,
		height=height,
		title={
			'text': label,
			# 'y': 0.9,
			'x': 0.5,
			'xanchor': 'center',
			'yanchor': 'top',
		}
	)

	if orientation == 'h':
		fig.update_layout(
			xaxis1=dict(
				# title=label,
				# side="top",
				visible=(orientation_raw == 'h'),
			),
			xaxis2=dict(
				title="Differences",
				side="bottom",
			),
			yaxis1=dict(
				visible=(orientation_raw == 'v'),
			),
			yaxis2=dict(
				visible=False,
			),
		)
	elif orientation == 'v':
		fig.update_layout(
			yaxis1=dict(
			# 	title=label,
			# 	side="left",
				visible=(orientation_raw == 'v'),
			),
			yaxis2=dict(
				# title="Differences",
				# side="right",
				visible=False,
			),
			xaxis1=dict(
				visible=(orientation_raw == 'h'),
			),
			xaxis2=dict(
				visible=True,
				title="Differences",
				side="bottom",
			),
		)

	return fig






class _ChooserRow(widget.Box):

	def __init__(self, tag, a_widget, b_widget, orientation='h', offwidget=None):
		self.tag = tag
		self.description = widget.Label(tag, layout=widget.Layout(
			width="135px"
		))
		# self.description = widget.HTML(
		# 	value=f'<div style="transform: rotate(-15deg);">{tag}</div>',
		# )
		self.a_widget = a_widget
		self.b_widget = b_widget
		if offwidget is None:
			self.offwidget = widget.Label("background")
		elif isinstance(offwidget, str):
			self.offwidget = widget.Label(offwidget)
		else:
			self.offwidget = offwidget
		self.offwidget.add_class("EMAT_NOSHOW")
		self.active_status_box = widget.ToggleButton(
			value=True,
			disabled=False,
			button_style='',  # 'success', 'info', 'warning', 'danger' or ''
			tooltip='Activate Parameter',
			icon='toggle-on',  # (FontAwesome names without the `fa-` prefix)
			layout = widget.Layout(width="40px"),
		)
		self.link_status_box = widget.ToggleButton(
			value=True,
			# description='Click me',
			disabled=False,
			button_style='',  # 'success', 'info', 'warning', 'danger' or ''
			tooltip='Link Values',
			icon='link',  # (FontAwesome names without the `fa-` prefix)
			layout = widget.Layout(width="40px"),
		)
		self.toggle_link()
		super().__init__(children=(
			self.description,
			self.active_status_box,
			self.a_widget,
			self.link_status_box,
			self.b_widget,
			self.offwidget,
		))
		self.layout.display = 'flex'
		self.layout.align_items = 'stretch'
		if orientation == 'v':
			self.layout.flex_flow = 'column'
		self.active_status_box.observe(self.toggle_active)
		self.link_status_box.observe(self.toggle_link)

	def toggle_link(self, payload=None):
		if payload is None:
			payload = {'name':'value', 'new':self.link_status_box.value}
		if payload.get('name') == 'value':
			if payload.get('new'):
				try:
					self.link_status = widget.jslink((self.a_widget, 'value'), (self.b_widget, 'value'))
				except TypeError:
					self.link_status = widget.link((self.a_widget, 'value'), (self.b_widget, 'value'))
			else:
				try:
					self.link_status.unlink()
				except:
					pass

	def toggle_active(self, payload=None):
		if payload is None:
			payload = {'name':'value', 'new':self.active_status_box.value}
		if payload.get('name') == 'value':
			if payload.get('new'):
				self.a_widget.disabled = False
				self.b_widget.disabled = False
				self.link_status_box.disabled = False
				# self.a_widget.layout.visibility = 'visible'
				# self.b_widget.layout.visibility = 'visible'
				# self.offwidget.layout.visibility = 'hidden'
				# self.link_status_box.layout.visibility = 'visible'
				# self.a_widget.layout.display = 'block'
				# self.b_widget.layout.display = 'block'
				# self.offwidget.layout.display = 'none'
				# self.link_status_box.layout.display = 'block'
				self.a_widget.remove_class("EMAT_NOSHOW")
				self.b_widget.remove_class("EMAT_NOSHOW")
				self.offwidget.add_class("EMAT_NOSHOW")
				self.link_status_box.remove_class("EMAT_NOSHOW")
				self.active_status_box.icon = 'toggle-on'
			else:
				self.a_widget.disabled = True
				self.b_widget.disabled = True
				self.link_status_box.disabled = True
				# self.a_widget.layout.visibility = 'hidden'
				# self.b_widget.layout.visibility = 'hidden'
				# self.offwidget.layout.visibility = 'visible'
				# self.link_status_box.layout.visibility = 'hidden'
				# self.a_widget.layout.display = 'none'
				# self.b_widget.layout.display = 'none'
				# self.offwidget.layout.display = 'block'
				# self.link_status_box.layout.display = 'none'
				self.a_widget.add_class("EMAT_NOSHOW")
				self.b_widget.add_class("EMAT_NOSHOW")
				self.offwidget.remove_class("EMAT_NOSHOW")
				self.link_status_box.add_class("EMAT_NOSHOW")
				self.active_status_box.icon = 'toggle-off'

_ox = {
	'h': 'horizontal',
	'v': 'vertical',
}

class AB_Chooser(widget.Box):

	def __init__(self, scope, orientation='h'):

		assert orientation in ('v', 'h')
		self.orientation = orientation
		a_slides = []
		b_slides = []
		rows = [
			widget.HTML("<style> .EMAT_NOSHOW {display: none} </style>")
		]
		assert isinstance(scope, Scope)

		def _add_param(param):
			if param.ptype == 'constant':
				return
			if param.dtype in ('cat', 'bool'):
				a = widget.SelectionSlider(
					value=param.default,
					options=param.values,
					continuous_update=False,
					orientation=_ox.get(orientation),
					style=dict(handle_color=_left_top_color),
				)
				b = widget.SelectionSlider(
					value=param.default,
					options=param.values,
					continuous_update=False,
					orientation=_ox.get(orientation),
					style=dict(handle_color=_right_bottom_color),
				)
			elif param.dtype == 'int':
				a = widget.IntSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					continuous_update=False,
					orientation=_ox.get(orientation),
					style=dict(handle_color=_left_top_color),
				)
				b = widget.IntSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					continuous_update=False,
					orientation=_ox.get(orientation),
					style=dict(handle_color=_right_bottom_color),
				)

			else:
				a = widget.FloatSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					step=(param.max - param.min)/20,
					continuous_update=False,
					orientation=_ox.get(orientation),
					style=dict(handle_color=_left_top_color),
				)
				b = widget.FloatSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					step=(param.max - param.min) / 20,
					continuous_update=False,
					orientation=_ox.get(orientation),
					style=dict(handle_color=_right_bottom_color),
				)
			a_slides.append(a)
			b_slides.append(b)
			rows.append(_ChooserRow(
				param.name,
				a_slides[-1],
				b_slides[-1],
				orientation=orientation,
				offwidget=scope[param.name].dist_description
			))
			if param.ptype != 'lever':
				rows[-1].active_status_box.value = False

		rows.append(widget.HTML("<b>Uncertainties</b>"))
		for param in scope.get_uncertainties():
			_add_param(param)
		rows.append(widget.HTML("<b>Policy Levers</b>"))
		for param in scope.get_levers():
			_add_param(param)

		super().__init__(children=rows)
		self.layout.display = 'flex'
		self.layout.align_items = 'stretch'
		if orientation == 'h': # intentionally backwards
			self.layout.flex_flow = 'column'

	def get_ab(self):
		a, b = {}, {}
		for row in self.children:
			try:
				if row.active_status_box.value:
					a[row.tag] = row.a_widget.value
					b[row.tag] = row.b_widget.value
			except AttributeError:
				pass
		return a, b


class AB_Viewer():

	def __init__(
			self,
			model,
			background=None,
			scope=None,
			figure_kwargs=None,
	):
		self.model = model
		self.scope = scope or model.scope
		self._chooser = AB_Chooser(self.scope)
		a, b = self._chooser.get_ab()
		ab = tuple(sorted(a.items())), tuple(sorted(b.items()))
		self._ab = ab
		self.contrast = AB_Contrast(
			self.model,
			a,
			b,
			background=background or DEFAULT_BACKGROUND,
			scope=self.scope,
		)
		self._figures = {}
		self._ab = None
		self.figure_kwargs = figure_kwargs or {}
		self.figure_kwargs.setdefault('orientation', 'v')
		self.figure_kwargs.setdefault('orientation_raw', 'h')
		self.figure_kwargs.setdefault('orientation_diff', 'h')
		self._compute_button = widget.Button(
			description="Recompute",
			layout=widget.Layout(margin="10px 0px 0px 0px"), # top right bottom left
		)
		self._compute_button.on_click(self.compute)
		self.interface = widget.VBox(
			[
				self._chooser,
				self._compute_button,
			],
			layout=dict(
				justify_content = 'space-between',
				align_items = 'stretch',
			),
		)

	def compute(self, payload=None):
		try:
			if payload.get('name') != 'value':
				print("NO ACTION -- ",payload)
				return
			print("CHANGE -- ",payload)
		except:
			pass
		self._compute_button.description = "running..."
		self._compute_button.disabled = True
		try:
			a, b = self._chooser.get_ab()
			ab = tuple(sorted(a.items())), tuple(sorted(b.items()))
			if self._ab == ab and isinstance(payload, dict) and 'force' not in payload:
				return
			self._ab = ab
			background = getattr(self.contrast, 'background', DEFAULT_BACKGROUND)
			self.contrast = AB_Contrast(self.model, a, b, background=background)
			for measure in self._figures.keys():
				self.get_figure(measure, **self.figure_kwargs)
		finally:
			self._compute_button.description = "Recompute"
			self._compute_button.disabled = False


	def get_figure(self, measure, **kwargs):
		if self.contrast is None:
			self.compute()
		kwargs.update(self.figure_kwargs)
		fig = self.contrast.get_figure(measure, **kwargs)
		if measure not in self._figures:
			self._figures[measure] = go.FigureWidget(fig)
		else:
			with self._figures[measure].batch_update():
				main_min = min(fig.data[0]['x'].min(), fig.data[1]['x'].min())
				main_max = min(fig.data[0]['x'].max(), fig.data[1]['x'].max())
				main_span = main_max - main_min
				diff_min = fig.data[2]['x'].min()
				diff_max = fig.data[2]['x'].max()
				diff_span = diff_max - diff_min
				if diff_span < main_span * 0.01:
					buffer = main_span * 0.005 - diff_span * 0.5
					diff_range = (diff_min-buffer, diff_max+buffer)
				else:
					diff_range = None
				for i in range(len(fig.data)):
					if self._figures[measure].data[i].orientation == 'h':
						self._figures[measure].data[i].x = fig.data[i]['x']
					else:
						self._figures[measure].data[i].y = fig.data[i]['y']
				self._figures[measure].layout = fig.layout
				if diff_range:
					self._figures[measure].layout.xaxis2.range = diff_range
		return self._figures[measure]

	def get_figures(self, *measures, **kwargs):
		if len(measures) == 1 and not isinstance(measures[0], str):
			measures = measures[0]
		w = [self.get_figure(m, **kwargs) for m in measures]
		return widget.Box(w, layout=widget.Layout(flex_flow='row wrap'))

	def set_parameter(self, name, value, side='b', unlink=True, recompute=True):
		for row in self._chooser.children:
			if getattr(row, 'tag', None) == name:
				if not unlink:
					row.link_status_box.value = True
				else:
					row.link_status_box.value = False
				if side.lower() in ('a', 'left', 'l'):
					row.a_widget.value = value
				elif side.lower() in ('b', 'right', 'r'):
					row.b_widget.value = value
		if recompute:
			self.compute()