
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

	Returns:
		pandas.DataFrame or 2-tuple of pandas.DataFrame
	"""

	def __init__(self, model, a, b, background, test_name=None):

		self.model = model
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
			background = model.design_experiments(
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

		results_a = self.model.run_experiments(self.design_a)
		results_b = self.model.run_experiments(self.design_b)

		measure_names = model.scope.get_measure_names()
		not_measures = [i for i in results_a.columns if i not in measure_names]
		self.results_a = results_a.drop(columns=not_measures)
		self.results_b = results_b.drop(columns=not_measures)

	def get_figure(self, measure, **kwargs):
		fig = create_violin(
			self.results_a[measure],
			self.results_b[measure],
			label=self.model.scope.shortname(measure),
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
		x_a, x_b,
		label=None,
		points='suspectedoutliers',
		points_diff=False,
		a_name=None,
		b_name=None,
		showlegend=False,
		orientation='h',
		orientation_raw='v',
		orientation_diff='v',
		width=300,
		height=400,
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
		fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.03, column_widths=[0.65, 0.35])
	else:
		raise ValueError("orientation must be 'h' or ''")

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
		line_color=colors.DEFAULT_BASE_COLOR,
		pointpos=0.5,
		**kwargs_common,
	)

	kwargs_b = dict(
		name=b_name,
		side='negative',
		line_color=colors.DEFAULT_HIGHLIGHT_COLOR,
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
				title="Differences",
				side="right",
			),
			xaxis1=dict(
				visible=(orientation_raw == 'h'),
			),
			xaxis2=dict(
				visible=False,
			),
		)

	return fig






class _ChooserRow(widget.Box):

	def __init__(self, tag, a_widget, b_widget, orientation='h'):
		self.tag = tag
		self.description = widget.Label(tag, layout=widget.Layout(
			width="135px"
		))
		# self.description = widget.HTML(
		# 	value=f'<div style="transform: rotate(-15deg);">{tag}</div>',
		# )
		self.a_widget = a_widget
		self.b_widget = b_widget
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
				self.a_widget.layout.visibility = 'visible'
				self.b_widget.layout.visibility = 'visible'
				self.link_status_box.layout.visibility = 'visible'
				self.active_status_box.icon = 'toggle-on'
			else:
				self.a_widget.disabled = True
				self.b_widget.disabled = True
				self.link_status_box.disabled = True
				self.a_widget.layout.visibility = 'hidden'
				self.b_widget.layout.visibility = 'hidden'
				self.link_status_box.layout.visibility = 'hidden'
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
		rows = []
		assert isinstance(scope, Scope)
		for param in scope.get_parameters():
			if param.ptype == 'constant':
				continue
			if param.dtype in ('cat', 'bool'):
				a = widget.SelectionSlider(
					value=param.default,
					options=param.values,
					continuous_update=False,
					orientation=_ox.get(orientation),
				)
				b = widget.SelectionSlider(
					value=param.default,
					options=param.values,
					continuous_update=False,
					orientation=_ox.get(orientation),
				)
			elif param.dtype == 'int':
				a = widget.IntSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					continuous_update=False,
					orientation=_ox.get(orientation),
				)
				b = widget.IntSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					continuous_update=False,
					orientation=_ox.get(orientation),
				)

			else:
				a = widget.FloatSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					step=(param.max - param.min)/20,
					continuous_update=False,
					orientation=_ox.get(orientation),
				)
				b = widget.FloatSlider(
					value=param.default,
					min=param.min,
					max=param.max,
					step=(param.max - param.min) / 20,
					continuous_update=False,
					orientation=_ox.get(orientation),
				)
			a_slides.append(a)
			b_slides.append(b)
			rows.append(_ChooserRow(
				param.name,
				a_slides[-1],
				b_slides[-1],
				orientation=orientation,
			))
			if param.ptype != 'lever':
				rows[-1].active_status_box.value = False

		super().__init__(children=rows)
		self.layout.display = 'flex'
		self.layout.align_items = 'stretch'
		if orientation == 'h': # intentionally backwards
			self.layout.flex_flow = 'column'

	def get_ab(self):
		a, b = {}, {}
		for row in self.children:
			if row.active_status_box.value:
				a[row.tag] = row.a_widget.value
				b[row.tag] = row.b_widget.value
		return a, b

def _AB_Viewer_compute(self, payload=None):
	a, b = self.chooser.get_ab()
	ab = tuple(sorted(a.items())), tuple(sorted(b.items()))
	if self._ab == ab: return
	self._ab = ab
	background = getattr(self.contrast, 'background', 500)
	self.contrast = AB_Contrast(self.model, a, b, background=background)
	for measure in self._figures.keys():
		self.get_figure(measure, **self.figure_kwargs)


class AB_Viewer():

	def __init__(
			self,
			model,
			figure_kwargs=None,
	):
		self.model = model
		self.chooser = AB_Chooser(model.scope)
		a, b = self.chooser.get_ab()
		ab = tuple(sorted(a.items())), tuple(sorted(b.items()))
		self._ab = ab
		self.contrast = AB_Contrast(self.model, a, b, background=500)
		self._figures = {}
		self._ab = None
		self.figure_kwargs = figure_kwargs or {}
		self.figure_kwargs.setdefault('orientation', 'h')
		self.figure_kwargs.setdefault('orientation_raw', 'v')
		self.figure_kwargs.setdefault('orientation_diff', 'h')
		for row in self.chooser.children:
			row.a_widget.observe(self.compute)
			row.b_widget.observe(self.compute)

	def compute(self, payload=None):
		return _AB_Viewer_compute(self, payload)

	def get_figure(self, measure, **kwargs):
		if self.contrast is None:
			self.compute()
		kwargs.update(self.figure_kwargs)
		fig = self.contrast.get_figure(measure, **kwargs)
		if measure not in self._figures:
			self._figures[measure] = go.FigureWidget(fig)
		else:
			with self._figures[measure].batch_update():
				for i in range(len(fig.data)):
					if self._figures[measure].data[i].orientation == 'h':
						self._figures[measure].data[i].x = fig.data[i]['x']
					else:
						self._figures[measure].data[i].y = fig.data[i]['y']
				self._figures[measure].layout = fig.layout
		return self._figures[measure]

	def get_figures(self, *measures, **kwargs):
		w = [self.get_figure(m, **kwargs) for m in measures]
		return widget.Box(w, layout=widget.Layout(flex_flow='row wrap'))

