
import pandas, numpy
import functools
from plotly import graph_objs as go
from plotly.subplots import make_subplots

X_FIGURE_BUFFER = 0.03

from ....viz import colors, _pick_color
from .... import styles
from ....util.naming import multiindex_to_strings, reset_multiindex_to_strings
from ....scope.box import Bounds
from ....util.si import si_units, get_float

import logging
_logger = logging.getLogger('EMAT.explore')

def _y_maximum(fig):
	return sum(t.y for t in list(fig.select_traces())[:2]).max()


def fig_existing_lines(fig):
	lines = []
	if 'shapes' in fig['layout']:
		for s in fig['layout']['shapes']:
			if s['type'] == 'line':
				lines.append(s)
	return lines


def embolden(text, bold=True):
	if text is None:
		return None
	if bold:
		if "<b>" in text:
			return text
		else:
			return f"<b>{text}</b>" if text else text
	else:
		if "<b>" not in text:
			return text
		else:
			return text.replace("<b>","").replace("</b>","")


def compute_earth_mover_distance(x, y):
	"""
	Compute the earth mover distance between two digitized continuous distributions.

	Args:
		x, y (array-like):
			Two arrays of the same shape describing two distributions
			on the same range.  All values should be non-negative.
			If either array sums to zero, no result is computed.

	Returns:
		float or None
	"""

	x_cum = numpy.asanyarray(x, dtype=numpy.float).cumsum()
	y_cum = numpy.asanyarray(y, dtype=numpy.float).cumsum()
	if x_cum[-1] == 0 or y_cum[-1] == 0:
		return None
	try:
		x_cum /= x_cum[-1]
		y_cum /= y_cum[-1]
	except TypeError:
		return -1.0
	return numpy.absolute(x_cum - y_cum)[:-1].mean()

def compute_categorical_similarity(x,y):
	"""
	Compute the categorical similarity between two discrete distributions.

	Args:
		x, y (array-like): Two arrays of the same shape

	Returns:
		float
	"""
	x = numpy.asarray(x, dtype=numpy.float).copy()
	y = numpy.asarray(y, dtype=numpy.float).copy()
	try:
		x /= x.sum()
		y /= y.sum()
	except TypeError:
		return -1.0
	return numpy.absolute(x - y).sum()/2.0 # divide by two for double-counting


def new_histogram_figure(
		selection,
		data_column,
		bins=20,
		*,
		marker_line_width=None,
		selected_color=None,
		unselected_color=None,
		title_text=None,
		on_select=None,     # lambda *a: self._on_select_from_histogram(*a,name=col)
		on_deselect=None,   # lambda *a: self._on_deselect_from_histogram(*a,name=col)
		figure_class=None,
		box=None,
		ref_point=None,
		ghost_fraction=0.2,
		earth_movers_dist=False,
):
	"""
	Create a new histogram figure for use with the visualizer.

	Args:
		selection (pandas.Series):
			The currently selected subset of the data.
		data_column (pandas.Series):
			The column of data used to define the histogram.
		bins (int or array-like):
			The number of histogram bins, or the precomputed bin edges.
		marker_line_width:
		selected_color:
		unselected_color:
		title_text:
		on_select:
		on_deselect:
		figure_class ({go.FigureWidget, go.Figure}, optional):
			The class type of figure to generate. If not given,
			a go.FigureWidget is created.

	Returns:
		go.FigureWidget or go.Figure
	"""
	unselected_color = colors.Color(unselected_color, default=colors.DEFAULT_BASE_COLOR)
	selected_color = colors.Color(selected_color, default=colors.DEFAULT_HIGHLIGHT_COLOR_RGB)
	if figure_class is None:
		figure_class = go.FigureWidget
	data_column_missing_data = data_column.isna()
	data_column_legit_data = ~data_column_missing_data

	if bins is None:
		bins = 20
	try:
		bar_heights, bar_x = numpy.histogram(data_column[data_column_legit_data], bins=bins)
	except:
		_logger.error("ERROR IN COMPUTING HISTOGRAM")
		_logger.error(f"  bins = {bins}")
		_logger.error(f"  data_column.name = {getattr(data_column,'name','name not found')}")
		_logger.error(f"  data_column = {data_column}")
		raise
	bins_left = bar_x[:-1]
	bins_width = bar_x[1:] - bar_x[:-1]
	bar_heights_select, bar_x = numpy.histogram(data_column[selection][data_column_legit_data], bins=bar_x)

	ghost_x, ghost_y = [], []
	ghost_mode = (numpy.sum(bar_heights_select) / numpy.sum(bar_heights) < ghost_fraction)
	if ghost_mode:
		max_height = numpy.max(bar_heights)
		max_select_height = numpy.max(bar_heights_select)
		ghost_scale = max_height/max_select_height
		if ghost_scale < 2.0:
			ghost_mode = False
	if ghost_mode:
		ghost_x, ghost_y = pseudo_bar_data(bar_x, bar_heights_select * ghost_scale)
	meta = dict()
	if earth_movers_dist:
		meta['emd'] = compute_earth_mover_distance(bar_heights_select, bar_heights)
	fig = figure_class(
		data=[
			go.Bar(
				x=bins_left + bins_width/2,
				y=bar_heights_select,
				width=bins_width,
				name='Inside',
				marker_color=selected_color.rgb(),
				marker_line_width=marker_line_width,
				hoverinfo='skip',
			),
			go.Bar(
				x=bins_left + bins_width/2,
				y=bar_heights - bar_heights_select,
				width=bins_width,
				name='Outside',
				marker_color=unselected_color.rgb(),
				marker_line_width=marker_line_width,
				hoverinfo='skip',
			),
			go.Scatter(
				x=ghost_x,
				y=ghost_y,
				mode='lines',
				line=dict(
					color=selected_color.rgb(),
					width=2,
					dash='2px,1px',
				),
				hoverinfo='skip',
				fill='tozeroy',
				fillcolor=selected_color.rgba(0.15),
			),
		],
		layout=dict(
			barmode='stack',
			showlegend=False,
			margin=styles.figure_margins,
			yaxis_showticklabels=False,
			title_text=title_text,
			title_x=0.5,
			title_xanchor='center',
			selectdirection='h',
			dragmode='select',
			meta=meta if len(meta) else None,
			**styles.figure_dims,
		),
	)
	# fig._bins = bins
	# fig._figure_kind = 'histogram'
	if on_select is not None:
		fig.data[1].on_selection(on_select)
	if on_deselect is not None:
		fig.data[1].on_deselect(on_deselect)
	_y_max = _y_maximum(fig)
	fig.layout.yaxis.range = (
		-_y_max * 0.03,
		_y_max * 1.05,
	)
	x_range = (
		fig.data[0].x[0] - (fig.data[0].width[0] / 2),
		fig.data[0].x[-1] + (fig.data[0].width[-1] / 2),
	)
	x_width = x_range[1] - x_range[0]

	# When ref_point is outside the range, expand the
	# range to include it.
	if ref_point is not None:
		if ref_point < x_range[0]:
			x_range = (ref_point, x_range[1])
		elif ref_point > x_range[1]:
			x_range = (x_range[0], ref_point)

	# Set the plotted range slightly wider than the
	# actual range of the data, to accommodate drawing
	# a box just beyond the range if needed.
	fig.layout.xaxis.range = (
		x_range[0] - x_width * X_FIGURE_BUFFER,
		x_range[1] + x_width * X_FIGURE_BUFFER,
	)
	col = getattr(data_column, 'name', None)
	fig = add_boxes_to_figure(box, col, fig, ref_point=ref_point)
	return fig

def update_histogram_figure(
		fig,
		selection,
		data_column,
		rerange_y=False,
		box=None,
		ref_point=None,
		selected_color=None,
		unselected_color=None,
		ghost_fraction=0.2,
		earth_movers_dist=False,
):
	"""
	Update an existing figure used in the visualizer.

	Args:
		fig:
		selection:
		data_column:

	Returns:
		fig
	"""
	unselected_color = colors.Color(unselected_color)
	selected_color = colors.Color(selected_color)
	bins = list(fig['data'][0]['x'] - fig['data'][0]['width']/2)
	bins.append(fig['data'][0]['x'][-1] + fig['data'][0]['width'][-1])
	data_column_missing_data = data_column.isna()
	data_column_legit_data = ~data_column_missing_data
	bar_heights, bar_x = numpy.histogram(data_column[data_column_legit_data], bins=bins)
	bar_heights_select, bar_x = numpy.histogram(
		data_column[selection][data_column_legit_data],
		bins=bar_x,
	)
	if 'meta' in fig['layout']:
		meta = fig['layout']['meta']
		if meta is None:
			meta = fig['layout']['meta'] = dict()
	else:
		meta = fig['layout']['meta'] = dict()
	if earth_movers_dist or 'emd' in meta:
		meta['emd'] = compute_earth_mover_distance(bar_heights_select, bar_heights)
	fig['data'][0]['y'] = bar_heights_select
	fig['data'][1]['y'] = bar_heights - bar_heights_select
	if rerange_y:
		_y_max = numpy.max(bar_heights)
		fig['layout']['yaxis']['range'] = (
			-_y_max * 0.03,
			_y_max * 1.05,
		)
	existing_lines = fig_existing_lines(fig) if ref_point is None else []
	col = getattr(data_column, 'name', None)
	fig = add_boxes_to_figure(box, col, fig, ref_point=ref_point, existing_shapes=existing_lines)
	if unselected_color is not None:
		fig['data'][1]['marker']['color'] = unselected_color.rgb()
	if selected_color is not None:
		fig['data'][0]['marker']['color'] = selected_color.rgb()
	ghost_mode = (numpy.sum(bar_heights_select) / numpy.sum(bar_heights) < ghost_fraction)
	if ghost_mode:
		max_height = numpy.max(bar_heights)
		max_select_height = numpy.max(bar_heights_select)
		ghost_scale = max_height/max_select_height
		if ghost_scale < 2.0:
			ghost_mode = False
	if ghost_mode:
		ghost_x, ghost_y = pseudo_bar_data(bar_x, bar_heights_select * ghost_scale)
		# manipulate existing ghost line
		fig['data'][2]['x'] = ghost_x
		fig['data'][2]['y'] = ghost_y
		if selected_color is not None:
			fig['data'][2]['line']['color'] = selected_color.rgb()
			fig['data'][2]['fillcolor'] = selected_color.rgba(0.15)
	else:
		fig['data'][2]['x'] = []
		fig['data'][2]['y'] = []
	return fig



def pseudo_bar_data(x_bins, y, gap=0):
	"""
	Parameters
	----------
	x_bins : array-like, shape=(N,) or (N+1,)
		The bin boundaries
	y : array-like, shape=(N,)
		The bar heights

	Returns
	-------
	x, y
	"""
	if len(x_bins) == len(y):
		# add a width
		width = (x_bins[-1] - x_bins[0]) / (len(x_bins)-1)
		x_bins = numpy.asarray(list(x_bins) + [x_bins[-1]+width])
	else:
		width = 0

	if gap:
		x_doubled = numpy.zeros(((x_bins.shape[0] - 1) * 4), dtype=numpy.float)
		x_doubled[::4] = x_bins[:-1]
		x_doubled[1::4] = x_bins[:-1]
		x_doubled[2::4] = x_bins[1:] - gap
		x_doubled[3::4] = x_bins[1:] - gap
		y_doubled = numpy.zeros(((y.shape[0]) * 4), dtype=y.dtype)
		y_doubled[1::4] = y
		y_doubled[2::4] = y
	else:
		x_doubled = numpy.zeros((x_bins.shape[0] - 1) * 2, dtype=x_bins.dtype)
		x_doubled[::2] = x_bins[:-1]
		x_doubled[1::2] = x_bins[1:]
		y_doubled = numpy.zeros((y.shape[0]) * 2, dtype=y.dtype)
		y_doubled[::2] = y
		y_doubled[1::2] = y
	return x_doubled-(width/2), y_doubled


def interpret_histogram_selection(name, selection_range, box, data, scope):

	select_min, select_max = selection_range

	min_value, max_value = None, None
	# Extract min and max from scope if possible
	if scope is not None and name not in scope.get_measure_names():
		min_value = scope[name].min
		max_value = scope[name].max
	# Extract min and max from .data if still missing
	if min_value is None:
		min_value = data[name].min()
	if max_value is None:
		max_value = data[name].max()

	close_to_max_value = max_value - 0.03 * (max_value - min_value)
	close_to_min_value = min_value + 0.03 * (max_value - min_value)
	_logger.debug("name: %s  limits: %f - %f", name, close_to_min_value, close_to_max_value)

	if select_min is not None and select_min <= close_to_min_value:
		select_min = None
	if select_max is not None and select_max >= close_to_max_value:
		select_max = None

	_logger.debug("name: %s  final range: %f - %f", name, select_min or numpy.nan, select_max or numpy.nan)

	box.set_bounds(name, select_min, select_max)
	return box


def new_frequencies_figure(
		selection,
		data_column,
		labels,
		*,
		marker_line_width=None,
		selected_color=None,
		unselected_color=None,
		title_text=None,
		on_select=None,    # lambda *a: self._on_select_from_freq(*a, name=col)
		on_deselect=None,  # lambda *a: self._on_deselect_from_histogram(*a, name=col)
		on_click=None,
		figure_class=None,
		label_name_map=None,
		box=None,
		ref_point=None,
		ghost_fraction=0.2,
		categorical_similarity=False,
):
	unselected_color = colors.Color(unselected_color, default=colors.DEFAULT_BASE_COLOR)
	selected_color = colors.Color(selected_color, default=colors.DEFAULT_HIGHLIGHT_COLOR_RGB)
	if figure_class is None:
		figure_class = go.FigureWidget
	if label_name_map is None:
		label_name_map = {True: 'True', False: 'False'}

	v = data_column.astype(
		pandas.CategoricalDtype(categories=labels, ordered=False)
	).cat.codes
	bar_heights, bar_x = numpy.histogram(v, bins=numpy.arange(0, len(labels) + 1))
	bar_heights_select, _ = numpy.histogram(v[selection], bins=numpy.arange(0, len(labels) + 1))
	original_labels = labels
	labels = [label_name_map.get(i, i) for i in labels]
	ghost_x, ghost_y = [], []
	ghost_mode = (numpy.sum(bar_heights_select) / numpy.sum(bar_heights) < ghost_fraction)
	if ghost_mode:
		max_height = numpy.max(bar_heights)
		max_select_height = numpy.max(bar_heights_select)
		ghost_scale = max_height / max_select_height
		if ghost_scale < 2.0:
			ghost_mode = False
	if ghost_mode:
		ghost_x = labels
		ghost_y = bar_heights_select * ghost_scale
	meta = dict(
		x_tick_values=original_labels,
	)
	if categorical_similarity:
		meta['cat_sim'] = compute_categorical_similarity(bar_heights_select, bar_heights)
	fig = figure_class(
		data=[
			go.Bar(
				x=labels,
				y=bar_heights_select,
				name='Inside',
				marker_color=selected_color.rgb(),
				marker_line_width=marker_line_width,
				hoverinfo='none',
			),
			go.Bar(
				x=labels,
				y=bar_heights - bar_heights_select,
				name='Outside',
				marker_color=unselected_color.rgb(),
				marker_line_width=marker_line_width,
				hoverinfo='none',
			),
			go.Scatter(
				x=ghost_x,
				y=ghost_y,
				mode='lines',
				line=dict(
					color=selected_color.rgb(),
					width=2,
					dash='2px,1px',
					shape='hvh',
				),
				hoverinfo='skip',
				fill='tozeroy',
				fillcolor=selected_color.rgba(0.15),
			),
		],
		layout=dict(
			barmode='stack',
			showlegend=False,
			margin=styles.figure_margins,
			yaxis_showticklabels=False,
			title_text=title_text,
			title_x=0.5,
			title_xanchor='center',
			selectdirection='h',
			dragmode='select',
			**styles.figure_dims,
			meta=meta,
		),
	)
	if on_select is not None:
		fig.data[1].on_selection(on_select)
	if on_deselect is not None:
		fig.data[1].on_deselect(on_deselect)
	if on_click is not None:
		fig.data[0].on_click(on_click)
		fig.data[1].on_click(on_click)
	_y_max = _y_maximum(fig)
	fig.layout.yaxis.range = (
		-_y_max * 0.03,
		_y_max * 1.05,
	)
	x_range = (
		-0.5,
		len(fig.data[0].x) - 0.5
	)
	x_width = x_range[1] - x_range[0]
	fig.layout.xaxis.range = (
		x_range[0] - x_width * X_FIGURE_BUFFER,
		x_range[1] + x_width * X_FIGURE_BUFFER,
	)
	col = getattr(data_column, 'name', None)
	fig = add_boxes_to_figure(box, col, fig, ref_point=ref_point)
	return fig

def update_frequencies_figure(
		fig,
		selection,
		data_column,
		rerange_y=False,
		box=None,
		ref_point=None,
		selected_color=None,
		unselected_color=None,
		ghost_fraction=0.2,
		categorical_similarity=False,
):
	unselected_color = colors.Color(unselected_color)
	selected_color = colors.Color(selected_color)
	labels = list(fig['layout']['meta']['x_tick_values'])
	v = data_column.astype(
		pandas.CategoricalDtype(categories=labels, ordered=False)
	).cat.codes
	bar_heights, bar_x = numpy.histogram(v, bins=numpy.arange(0, len(labels) + 1))
	bar_heights_select, _ = numpy.histogram(v[selection], bins=numpy.arange(0, len(labels) + 1))
	fig['data'][0]['y'] = bar_heights_select
	fig['data'][1]['y'] = bar_heights - bar_heights_select
	if rerange_y:
		_y_max = numpy.max(bar_heights)
		fig['layout']['yaxis']['range'] = (
			-_y_max * 0.03,
			_y_max * 1.05,
		)
	existing_lines = fig_existing_lines(fig) if ref_point is None else []
	col = getattr(data_column, 'name', None)
	fig = add_boxes_to_figure(box, col, fig, ref_point=ref_point, existing_shapes=existing_lines)
	if unselected_color is not None:
		fig['data'][1]['marker']['color'] = unselected_color.rgb()
	if selected_color is not None:
		fig['data'][0]['marker']['color'] = selected_color.rgb()
	if 'meta' in fig['layout']:
		meta = fig['layout']['meta']
		if meta is None:
			meta = fig['layout']['meta'] = dict()
	else:
		meta = fig['layout']['meta'] = dict()
	if categorical_similarity or 'cat_sim' in meta:
		meta['cat_sim'] = compute_categorical_similarity(bar_heights_select, bar_heights)
	ghost_mode = (numpy.sum(bar_heights_select) / numpy.sum(bar_heights) < ghost_fraction)
	if ghost_mode:
		max_height = numpy.max(bar_heights)
		max_select_height = numpy.max(bar_heights_select)
		ghost_scale = max_height / max_select_height
		if ghost_scale < 2.0:
			ghost_mode = False
	if ghost_mode:
		ghost_y = bar_heights_select * ghost_scale
		# manipulate existing ghost line
		fig['data'][2]['x'] = fig['data'][0]['x']
		fig['data'][2]['y'] = ghost_y
		if selected_color is not None:
			fig['data'][2]['line']['color'] = selected_color.rgb()
			fig['data'][2]['fillcolor'] = selected_color.rgba(0.15)
	else:
		fig['data'][2]['x'] = []
		fig['data'][2]['y'] = []
	return fig


def add_boxes_to_figure(box, col, fig, ref_point=None, existing_shapes=None):
	if existing_shapes is None:
		existing_shapes = []

	box_shapes = []
	ref_shapes = []
	_y_max = sum(t['y'] for t in fig['data'][:2]).max()
	y_range = (
		-_y_max * 0.02,
		_y_max * 1.04,
	)
	if box is not None and col in box.thresholds:
		x_lo, x_hi = None, None
		thresh = box.thresholds.get(col)
		if isinstance(thresh, Bounds):
			x_lo, x_hi = thresh
		if isinstance(thresh, set):
			x_lo, x_hi = [], []
			for tickval, ticktext in enumerate(fig['layout']['meta']['x_tick_values']):
				if ticktext in thresh:
					x_lo.append(tickval - 0.45)
					x_hi.append(tickval + 0.45)

		try:
			x_range = (
				fig['data'][0]['x'][0] - (fig['data'][0]['width'][0] / 2),
				fig['data'][0]['x'][-1] + (fig['data'][0]['width'][-1] / 2),
			)
		except (TypeError, KeyError):
			x_range = (
				-0.5,
				len(fig['data'][0]['x']) + 0.5
			)
		x_width = x_range[1] - x_range[0]
		if x_lo is None:
			x_lo = x_range[0] - x_width * 0.02
		if x_hi is None:
			x_hi = x_range[1] + x_width * 0.02
		if not isinstance(x_lo, list):
			x_lo = [x_lo]
		if not isinstance(x_hi, list):
			x_hi = [x_hi]

		y_lo, y_hi = None, None
		y_width = y_range[1] - y_range[0]
		if y_lo is None:
			y_lo = y_range[0] - y_width * 0
		if y_hi is None:
			y_hi = y_range[1] + y_width * 0
		if not isinstance(y_lo, list):
			y_lo = [y_lo]
		if not isinstance(y_hi, list):
			y_hi = [y_hi]

		x_pairs = list(zip(x_lo, x_hi))
		y_pairs = list(zip(y_lo, y_hi))

		box_shapes.extend([
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
		])

		box_shapes.extend([
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
					width=2,
					color=colors.DEFAULT_BOX_LINE_COLOR,
				),
				fillcolor='rgba(0,0,0,0)',
				opacity=1.0,
			)
			for x_pair in x_pairs
			for y_pair in y_pairs
		])



	if ref_point is not None:
		try:
			label_values = list(fig['layout']['meta']['x_tick_values'])
			label_text = list(fig['data'][0]['x'])
		except:
			pass
		else:
			for x_val, x_txt in zip(label_values, label_text):
				ref_point_ = str(ref_point).lower()
				if ref_point == x_val or ref_point_ == str(x_val).lower() or ref_point_ == str(x_txt).lower():
					ref_point = x_txt
					break
		ref_shapes.append(
			go.layout.Shape(
				type="line",
				xref="x1",
				yref="y1",
				x0=ref_point,
				y0=y_range[0],
				x1=ref_point,
				y1=y_range[1],
				**colors.DEFAULT_REF_LINE_STYLE,
			)
		)

	if 'title' in fig['layout']:
		if box_shapes:
			fig['layout']['title']['font']['color'] = colors.DEFAULT_BOX_LINE_COLOR
			fig['layout']['title']['text'] = embolden(fig['layout']['title']['text'], True)
		else:
			fig['layout']['title']['font']['color'] = None
			fig['layout']['title']['text'] = embolden(fig['layout']['title']['text'], False)

	fig['layout']['shapes'] = existing_shapes + ref_shapes + box_shapes
	return fig





from ....viz.perturbation import perturb_categorical

def axis_info(x, range_padding=0.0, epsilon=None, refpoint=None):
	if hasattr(x, 'dtype'):
		if epsilon is None:
			s_ = x.size * 0.01
			s_ = s_ / (1 + s_)
			epsilon = 0.05 + 0.20 * s_
		if isinstance(x.dtype, pandas.CategoricalDtype):
			x_categories = x.cat.categories
			x_ticktext = list(x_categories)
			x_tickvals = list(range(len(x_ticktext)))
			x_range = [-epsilon - range_padding, x_tickvals[-1] + epsilon + range_padding]
			return x_ticktext, x_tickvals, x_range
		if numpy.issubdtype(x.dtype, numpy.bool_):
			x_range = [-epsilon - range_padding, 1 + epsilon + range_padding]
			return ["False", "True"], [0,1], x_range
	x_range = [x.min(), x.max()]
	if refpoint is not None:
		if refpoint < x_range[0]:
			x_range[0] = refpoint
		if refpoint > x_range[1]:
			x_range[1] = refpoint
	x_span = x_range[1] - x_range[0]
	if x_span <= 0: x_span = 1
	x_range = [x_range[0] - x_span*0.07, x_range[1] + x_span*0.07]
	return None, None, x_range

def perturb_categorical_df(df, col=None, suffix="perturb"):
	if col is None:
		cols = list(df.columns)
	else:
		cols = [col]
	for i in cols:
		if f"_{i}_{suffix}" in df.columns: continue
		x, x_ticktext, x_tickvals, x_range, _ = perturb_categorical(df[i],
																	add_variance=(suffix=="perturb"))
		if x_ticktext is not None:
			df[f"_{i}_{suffix}"] = x
	if col is not None:
		if f"_{col}_{suffix}" in df.columns:
			return df[f"_{col}_{suffix}"]
		else:
			return df[col]
	return df

def _get_or_none(mapping, key):
	if mapping is None:
		return None
	return mapping.get(key, None)

def new_splom_figure(
		scope,
		data,
		rows="LX",
		cols="M",
		use_gl=True,
		mass=250,
		row_titles='top',
		size=150,
		selection=None,
		box=None,
		refpoint=None,
		figure_class=None,
		on_select=None,  # lambda *a: self._on_select_from_histogram(*a,name=col)
		on_deselect=None,  # lambda *a: self._on_deselect_from_histogram(*a,name=col)
		selected_color=None,
		unselected_color=None,
		marker_size=3,
):
	if unselected_color is None:
		unselected_color = colors.DEFAULT_BASE_COLOR
	if selected_color is None:
		selected_color = colors.DEFAULT_HIGHLIGHT_COLOR
	selected_color_str = ", ".join(str(int(i)) for i in colors.interpret_color(selected_color))
	unselected_color_str = ", ".join(str(int(i)) for i in colors.interpret_color(unselected_color))

	def _make_axis_list(j):
		if isinstance(j, str):
			if set('XLM').issuperset(j.upper()):
				use = []
				for i in j.upper():
					if i=='X':
						use += scope.get_uncertainty_names()
					elif i=='L':
						use += scope.get_lever_names()
					if i=='M':
						use += scope.get_measure_names()
				return use
			return [j]
		return j
	rows = _make_axis_list(rows)
	cols = _make_axis_list(cols)
	rows = [i for i in rows if i in data.columns]
	cols = [i for i in cols if i in data.columns]

	row_titles_top = (row_titles=='top')

	subplot_titles = []
	specs = []
	for rownum, row in enumerate(rows, start=1):
		specs.append([])
		for colnum, col in enumerate(cols, start=1):
			specs[-1].append({
				# "type": "xy",
				# 'l':0.03,
				# 'r':0.03,
				# 't':0.03,
				# 'b':0.03,
			})
			if colnum == 1 and row_titles_top:
				subplot_titles.append(scope.tagged_shortname(row))
			else:
				subplot_titles.append(None)

	if len(cols)==0 or len(rows)==0:
		fig = go.Figure()
	else:
		fig = make_subplots(
			rows=len(rows), cols=len(cols),
			shared_xaxes=True,
			shared_yaxes=True,
			vertical_spacing=(0.18 if row_titles_top else 0.1)/len(rows),
			horizontal_spacing=0.1/len(cols),
			subplot_titles=subplot_titles,
			specs=specs,
		)
		if row_titles_top:
			for rowtitle in fig['layout']['annotations']:
				rowtitle['x'] = 0
				rowtitle['xanchor'] = 'left'
		if size is not None:
			fig['layout']['height'] = size * len(rows) + 75
			fig['layout']['width'] = size * len(cols) + 100

		Scatter = go.Scattergl if use_gl else go.Scatter

		marker_opacity = _splom_marker_opacity(
			data.index,
			selection,
			mass=mass,
		)

		if selection is None:
			marker_color = None
		else:
			marker_color = pandas.Series(data=unselected_color, index=data.index)
			marker_color[selection] = selected_color

		experiment_name = "Experiment"
		if data.index.name:
			experiment_name = data.index.name

		n = 0
		extra_y_ax = len(rows) * len(cols)

		for rownum, row in enumerate(rows, start=1):
			for colnum, col in enumerate(cols, start=1):
				n += 1

				x = perturb_categorical_df(data, col)
				y = perturb_categorical_df(data, row)
				x_ticktext, x_tickvals, x_range = axis_info(data[col], range_padding=0.3,
				                                            refpoint=_get_or_none(refpoint, col))
				y_ticktext, y_tickvals, y_range = axis_info(data[row], range_padding=0.3,
				                                            refpoint=_get_or_none(refpoint, row))

				if row == col:
					extra_y_ax += 1
					import scipy.stats
					try:
						kde0 = scipy.stats.gaussian_kde(data[~selection][row])
						kde1 = scipy.stats.gaussian_kde(data[selection][row])
					except TypeError:
						kde0 = scipy.stats.gaussian_kde(data[~selection][row].cat.codes)
						kde1 = scipy.stats.gaussian_kde(data[selection][row].cat.codes)
					except ValueError:
						if selection.all():
							kde0 = lambda z: numpy.zeros_like(z)
							kde1 = scipy.stats.gaussian_kde(data[row])
						elif (~selection).all():
							kde0 = scipy.stats.gaussian_kde(data[row])
							kde1 = lambda z: numpy.zeros_like(z)
					x_fill = numpy.linspace(*x_range, 200)
					y_0 = kde0(x_fill)
					y_1 = kde1(x_fill)
					topline = max(y_0.max(), y_1.max())
					y_range_kde = (-0.07 * topline, 1.07 * topline)

					layout_updates = {}
					layout_updates[f'yaxis{extra_y_ax}'] = dict(
						domain=fig['layout'][f'yaxis{n}']['domain'],
						anchor=f'free',
						showticklabels=False,
						range=y_range_kde,
					)
					fig.update_layout(**layout_updates)

					fig.add_trace(
						go.Scatter(
							x=[],
							y=[],
							mode='markers',
							showlegend=False,
						),
						row=rownum, col=colnum,
					)

					fig.add_trace(
						go.Scatter(
							x=x_fill,
							y=y_0,
							yaxis=f"y{extra_y_ax}",
							xaxis=f"x{n}",
							showlegend=False,
							line_color=f'rgb({unselected_color_str})',
							fill='tozeroy',
						)
					)
					fig.add_trace(
						go.Scatter(
							x=x_fill,
							y=y_1,
							yaxis=f"y{extra_y_ax}",
							xaxis=f"x{n}",
							showlegend=False,
							line_color=f'rgb({selected_color_str})',
							fill='tozeroy',
						)
					)
				else:

					if marker_color is None:
						color = _pick_color(scope, row, col)
					else:
						color = marker_color

					if x_ticktext is not None or y_ticktext is not None:
						hovertemplate = (
							f'<b>{scope.shortname(row)}</b>: %{{meta[1]}}<br>' +
							f'<b>{scope.shortname(col)}</b>: %{{meta[2]}}' +
							f'<extra>{experiment_name} %{{meta[0]}}</extra>'
						)
						meta = reset_multiindex_to_strings(data[[row,col]]).to_numpy()
					else:
						hovertemplate = (
							f'<b>{scope.shortname(row)}</b>: %{{y}}<br>' +
							f'<b>{scope.shortname(col)}</b>: %{{x}}' +
							f'<extra>{experiment_name} %{{meta}}</extra>'
						)
						meta = multiindex_to_strings(data.index)

					fig.add_trace(
						Scatter(
							x=x,
							y=y,
							mode='markers',
							marker=dict(
								size=marker_size,
								opacity=marker_opacity,
								color=color,
							),
							showlegend=False,
							hovertemplate=hovertemplate,
							meta=meta,
						),
						row=rownum, col=colnum,
					)
					if on_select is not None:
						fig.data[-1].on_selection(functools.partial(on_select, col, row))
					if on_deselect is not None:
						fig.data[-1].on_deselect(functools.partial(on_deselect, col, row))

					if box is not None:
						shapes = _splom_part_boxes(
							box, n,
							x, col, x_tickvals, x_ticktext, x_range,
							y, row, y_tickvals, y_ticktext, y_range,
							background_opacity=0.05,
						)
						for s in shapes:
							fig.add_shape(s)

					if refpoint is not None:
						shapes = _splom_part_ref_point(
							n,
							x, x_range, x_tickvals, x_ticktext, refpoint.get(col, None),
							y, y_range, y_tickvals, y_ticktext, refpoint.get(row, None),
						)
						for s in shapes: fig.add_shape(s)

				if colnum == 1:
					fig.update_yaxes(
						range=y_range,
						row=rownum,
						col=colnum,
					)
					if not row_titles_top:
						fig.update_yaxes(
							title_text=scope.tagged_shortname(row, wrap_width=18),
							row=rownum,
							col=colnum,
						)
					if y_ticktext is not None:
						fig.update_yaxes(
							row=rownum,
							col=colnum,
							tickmode = 'array',
							ticktext = y_ticktext,
							tickvals = y_tickvals,
						)
				if rownum == len(rows):
					fig.update_xaxes(
						title_text=scope.tagged_shortname(col, wrap_width=18),
						row=rownum,
						col=colnum,
						range=x_range,
					)
					if x_ticktext is not None:
						fig.update_xaxes(
							row=rownum,
							col=colnum,
							tickmode='array',
							ticktext=x_ticktext,
							tickvals=x_tickvals,
						)
	fig.update_layout(
		margin=dict(
			l=10, r=10, t=30 if row_titles_top else 10, b=10,
		),
		plot_bgcolor=colors.DEFAULT_PLOT_BACKGROUND_COLOR,
	)
	metadata = dict(
		rows=rows,
		cols=cols,
		selected_color=selected_color,
		unselected_color=unselected_color,
		use_gl=use_gl,
		row_titles=row_titles,
		size=size,
		refpoint=refpoint,
		marker_size=marker_size,
	)
	if isinstance(mass, int):
		metadata['mass'] = mass
	fig.update_layout(meta=metadata)
	if figure_class is not None:
		fig = figure_class(fig)
	return fig





def _splom_marker_opacity(
		data_index,
		selection,
		mass=1000,
):
	if isinstance(mass, int):
		from ....viz import ScatterMass
		mass = ScatterMass(mass)
	data_index = pandas.Index(data_index)
	if selection is None:
		marker_opacity = mass.get_opacity(data_index)
	else:
		mo = [
			mass.get_opacity(data_index[~selection]),
			mass.get_opacity(data_index[selection]),
		]
		marker_opacity = pandas.Series(data=mo[0], index=data_index)
		marker_opacity[selection] = mo[1]
	return marker_opacity

def _splom_part_boxes(
		box, ax_num,
		x, x_label, x_tickvals, x_ticktext, x_range,
		y, y_label, y_tickvals, y_ticktext, y_range,
		background_opacity=0.2,
):
	background_shapes, foreground_shapes = [], []
	if box is None: return []

	if x_label in box.thresholds or y_label in box.thresholds:
		x_lo, x_hi = None, None
		thresh = box.thresholds.get(x_label)
		if isinstance(thresh, Bounds):
			x_lo, x_hi = thresh
		if isinstance(thresh, set):
			x_lo, x_hi = [], []
			for tickval, ticktext in zip(x_tickvals, x_ticktext):
				if ticktext in thresh:
					x_lo.append(tickval -0.33)
					x_hi.append(tickval +0.33)
		if x_range is None:
			x_range = [x.min(), x.max()]
			x_width_buffer = (x_range[1] - x_range[0]) * 0.02
		else:
			x_width_buffer = -(x_range[1] - x_range[0]) * 0.02
		if x_lo is None:
			x_lo = x_range[0] - x_width_buffer
		if x_hi is None:
			x_hi = x_range[1] + x_width_buffer
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
			for tickval, ticktext in zip(y_tickvals, y_ticktext):
				if ticktext in thresh:
					y_lo.append(tickval -0.33)
					y_hi.append(tickval +0.33)
		if y_range is None:
			y_range = [y.min(), y.max()]
			y_width_buffer = (y_range[1] - y_range[0]) * 0.02
		else:
			y_width_buffer = -(y_range[1] - y_range[0]) * 0.02
		if y_lo is None:
			y_lo = y_range[0] - y_width_buffer
		if y_hi is None:
			y_hi = y_range[1] + y_width_buffer
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
				xref=f"x{ax_num}",
				yref=f"y{ax_num}",
				x0=x_pair[0],
				y0=y_pair[0],
				x1=x_pair[1],
				y1=y_pair[1],
				line=dict(
					width=0,
				),
				fillcolor=colors.DEFAULT_BOX_BG_COLOR,
				opacity=background_opacity,
				layer="below",
			)
			for x_pair in x_pairs
			for y_pair in y_pairs
		]

		foreground_shapes += [
			# Rectangle reference to the axes
			go.layout.Shape(
				type="rect",
				xref=f"x{ax_num}",
				yref=f"y{ax_num}",
				x0=x_pair[0],
				y0=y_pair[0],
				x1=x_pair[1],
				y1=y_pair[1],
				line=dict(
					width=2,
					color=colors.DEFAULT_BOX_LINE_COLOR,
				),
				fillcolor='rgba(0,0,0,0)',
				opacity=1.0,
			)
			for x_pair in x_pairs
			for y_pair in y_pairs
		]
	return background_shapes + foreground_shapes

def _splom_part_ref_point(
		ax_num,
		x, x_range, x_tickvals, x_ticktext, x_refpoint,
		y, y_range, y_tickvals, y_ticktext, y_refpoint,
):
	foreground_shapes = []

	if x_refpoint is not None:
		if isinstance(x_refpoint, (bool, numpy.bool_)):
			x_refpoint = str(x_refpoint)
		if x_tickvals is not None and x_ticktext is not None:
			for tickval, ticktext in zip(x_tickvals, x_ticktext):
				if ticktext == x_refpoint:
					x_refpoint = tickval
					break
		if y_range is None:
			y_range = [y.min(), y.max()]
			if y_refpoint is not None:
				if y_refpoint < y_range[0]:
					y_range[0] = y_refpoint
				if y_refpoint > y_range[1]:
					y_range[1] = y_refpoint
			y_width_buffer = (y_range[1] - y_range[0]) * 0.02
		else:
			y_width_buffer = -(y_range[1] - y_range[0]) * 0.02
		y_lo = y_range[0] - y_width_buffer
		y_hi = y_range[1] + y_width_buffer
		foreground_shapes.append(
			go.layout.Shape(
				type="line",
				xref=f"x{ax_num}",
				yref=f"y{ax_num}",
				y0=y_lo,
				x0=x_refpoint,
				y1=y_hi,
				x1=x_refpoint,
				**colors.DEFAULT_REF_LINE_STYLE,
			)
		)

	if y_refpoint is not None:
		if isinstance(y_refpoint, (bool, numpy.bool_)):
			y_refpoint = str(y_refpoint)
		if y_tickvals is not None and y_ticktext is not None:
			for tickval, ticktext in zip(y_tickvals, y_ticktext):
				if ticktext == y_refpoint:
					y_refpoint = tickval
					break
		if x_range is None:
			x_range = [x.min(), x.max()]
			if x_refpoint is not None:
				if x_refpoint < x_range[0]:
					x_range[0] = x_refpoint
				if x_refpoint > x_range[1]:
					x_range[1] = x_refpoint
			x_width_buffer = (x_range[1] - x_range[0]) * 0.02
		else:
			x_width_buffer = -(x_range[1] - x_range[0]) * 0.02
		x_lo = x_range[0] - x_width_buffer
		x_hi = x_range[1] + x_width_buffer
		foreground_shapes.append(
			go.layout.Shape(
				type="line",
				xref=f"x{ax_num}",
				yref=f"y{ax_num}",
				x0=x_lo,
				y0=y_refpoint,
				x1=x_hi,
				y1=y_refpoint,
				**colors.DEFAULT_REF_LINE_STYLE,
			)
		)

	return foreground_shapes


def update_splom_figure(
		scope,
		data,
		fig,
		selection,
		box,
		mass=None,
		rows=None,
		cols=None,
		selected_color=None,
		unselected_color=None,
		size=None,
):
	existing_rows = fig['layout']['meta']['rows']
	existing_cols = fig['layout']['meta']['cols']
	if selected_color is None:
		selected_color = fig['layout']['meta'].get('selected_color', colors.DEFAULT_HIGHLIGHT_COLOR)
	if unselected_color is None:
		unselected_color = fig['layout']['meta'].get('unselected_color', colors.DEFAULT_BASE_COLOR)
	if mass is None:
		mass = fig['layout']['meta'].get('mass', 250)
	change_dims = False
	if rows is None:
		rows = existing_rows
	else:
		if rows != existing_rows:
			change_dims = True
	if cols is None:
		cols = existing_cols
	else:
		if cols != existing_cols:
			change_dims = True
	if change_dims:
		new_fig = new_splom_figure(
			scope,
			data,
			rows=rows,
			cols=cols,
			use_gl=fig['layout']['meta'].get('use_gl', True),
			mass=mass,
			row_titles=fig['layout']['meta'].get('row_titles', 'side'),
			size=size if size is not None else fig['layout']['meta'].get('size', None),
			selection=selection,
			box=box,
			refpoint=fig['layout']['meta'].get('refpoint', None),
			figure_class=None,
			on_select=None,  # lambda *a: self._on_select_from_histogram(*a,name=col)
			on_deselect=None,  # lambda *a: self._on_deselect_from_histogram(*a,name=col)
			selected_color=selected_color,
			unselected_color=unselected_color,
			marker_size=fig['layout']['meta'].get('marker_size', 3),
		)
		fig['data'] = new_fig['data']
		fig['layout'] = new_fig['layout']
		return fig

	existing_lines = fig_existing_lines(fig)
	marker_opacity = _splom_marker_opacity(
		data.index,
		selection,
		mass=mass,
	)
	if selection is None:
		marker_color = None
	else:
		marker_color = pandas.Series(data=unselected_color, index=data.index)
		marker_color[selection] = selected_color
	n = 0
	trace_n = 0
	box_shapes = []
	extra_y_ax = len(rows) * len(cols)

	for rownum, row in enumerate(rows, start=1):
		for colnum, col in enumerate(cols, start=1):
			if row == col:
				x_ticktext, x_tickvals, x_range = axis_info(data[col], range_padding=0.3)
				extra_y_ax += 1
				n += 1
				__update_univar_cell_in_splom(
					data,
					selection,
					row,
					x_range,
					fig,
					trace_n,
					unselected_color,
					selected_color,
					extra_y_ax,
					n,
				)
				trace_n += 3
				continue

			if marker_color is None:
				color = _pick_color(scope, row, col)
			else:
				color = marker_color
			x = perturb_categorical_df(data, col)
			y = perturb_categorical_df(data, row)
			x_ticktext, x_tickvals, x_range = axis_info(data[col], range_padding=0.3)
			y_ticktext, y_tickvals, y_range = axis_info(data[row], range_padding=0.3)
			fig['data'][trace_n]['x'] = x
			fig['data'][trace_n]['y'] = y
			fig['data'][trace_n]['marker']['color'] = color
			fig['data'][trace_n]['marker']['opacity'] = marker_opacity
			n += 1
			trace_n += 1
			box_shapes.extend(_splom_part_boxes(
				box, n,
				x, col, x_tickvals, x_ticktext, x_range,
				y, row, y_tickvals, y_ticktext, y_range,
				background_opacity=0.05,
			))
	fig['layout']['shapes'] = box_shapes + existing_lines
	return fig

def __update_univar_cell_in_splom(
		data,
		selection,
		row,
		x_range,
		fig,
		trace_n,
		unselected_color_str,
		selected_color_str,
		extra_y_ax,
		n,
):
	import scipy.stats
	# recompute KDEs
	try:
		kde0 = scipy.stats.gaussian_kde(data[~selection][row])
		kde1 = scipy.stats.gaussian_kde(data[selection][row])
	except TypeError:
		kde0 = scipy.stats.gaussian_kde(data[~selection][row].cat.codes)
		kde1 = scipy.stats.gaussian_kde(data[selection][row].cat.codes)
	except ValueError:
		if selection.all():
			kde0 = lambda z: numpy.zeros_like(z)
			kde1 = scipy.stats.gaussian_kde(data[row])
		elif (~selection).all():
			kde0 = scipy.stats.gaussian_kde(data[row])
			kde1 = lambda z: numpy.zeros_like(z)
	x_fill = numpy.linspace(*x_range, 200)
	y_0 = kde0(x_fill)
	y_1 = kde1(x_fill)
	topline = max(y_0.max(), y_1.max())

	fig['data'][trace_n + 1]['y'] = y_0
	fig['data'][trace_n + 2]['y'] = y_1
	fig['data'][trace_n + 1]['line']['color'] = unselected_color_str
	fig['data'][trace_n + 2]['line']['color'] = selected_color_str

	y_range_kde = (-0.07 * topline, 1.07 * topline)
	layout_updates = {}
	layout_updates[f'yaxis{extra_y_ax}'] = dict(
		domain=fig['layout'][f'yaxis{n}']['domain'],
		anchor=f'free',
		showticklabels=False,
		range=y_range_kde,
	)
	fig.update_layout(**layout_updates)


def _hue_mix(selected_array, unselected_array, selected_rgb, unselected_rgb):
	selected_rgb = numpy.asanyarray(selected_rgb)
	unselected_rgb = numpy.asanyarray(unselected_rgb)
	use255 = False
	if selected_rgb.max() > 1 or unselected_rgb.max() > 1:
		selected_rgb = selected_rgb/255
		unselected_rgb = unselected_rgb/255
		use255 = True

	selected_array = numpy.asanyarray(selected_array)
	unselected_array = numpy.asanyarray(unselected_array)
	selection_total = selected_array + unselected_array
	selection_intensity = numpy.nan_to_num(selected_array / (selection_total+0.00001))
	# selection_total /= numpy.max(selection_total)
	selection_total = selection_total / numpy.percentile(selection_total, 99)
	selection_total = numpy.clip(selection_total, 0, 1)

	from matplotlib.colors import LinearSegmentedColormap
	cmap = LinearSegmentedColormap.from_list("BlOr", [unselected_rgb, selected_rgb])
	hue_array = cmap(selection_intensity)
	if use255:
		hue_array = numpy.round(hue_array*255)
	hue_array[...,-1] = selection_total
	return hue_array

from .... import configuration

def _get_bins_and_range(ticktext, label, in_range, scope):
	bins = 20
	range_ = in_range
	if ticktext is not None:
		bins = len(ticktext) * 2 + 1
		range_ = (in_range[0] - 0.25, in_range[1] + 0.25)
	else:
		param = scope[label]
		try:
			range_ = (param.min, param.max)
		except AttributeError:
			pass
		try:
			this_type = scope.get_dtype(label)
		except:
			this_type = 'float'
		if this_type == 'int':
			if param.max - param.min + 1 <= bins * configuration.config.get("integer_bin_ratio", 4):
				bins = param.max - param.min + 1
			range_ = (param.min-0.5, param.max+0.5)
	return bins, range_

def new_hmm_figure(
		scope,
		data,
		rows="LX",
		cols="M",
		row_titles='top',
		size=150,
		selection=None,
		box=None,
		refpoint=None,
		figure_class=None,
		on_select=None,  # lambda *a: self._on_select_from_histogram(*a,name=col)
		on_deselect=None,  # lambda *a: self._on_deselect_from_histogram(*a,name=col)
		selected_color=None,
		unselected_color=None,
		emph_selected=True,
		show_points=50,
		show_points_frac=0.1,
		marker_size=5,
		with_hover=True,
):
	import datashader as ds  # optional dependency

	if unselected_color is None:
		unselected_color = colors.DEFAULT_BASE_COLOR_RGB
	else:
		unselected_color = colors.Color(unselected_color)
	if selected_color is None:
		selected_color = colors.DEFAULT_HIGHLIGHT_COLOR_RGB
	else:
		selected_color = colors.Color(selected_color)

	def _make_axis_list(j):
		if isinstance(j, str):
			if set('XLM').issuperset(j.upper()):
				use = []
				for i in j.upper():
					if i=='X':
						use += scope.get_uncertainty_names()
					elif i=='L':
						use += scope.get_lever_names()
					if i=='M':
						use += scope.get_measure_names()
				return use
			return [j]
		return j
	rows = _make_axis_list(rows)
	cols = _make_axis_list(cols)

	row_titles_top = (row_titles=='top')

	subplot_titles = []
	specs = []
	for rownum, row in enumerate(rows, start=1):
		specs.append([])
		for colnum, col in enumerate(cols, start=1):
			specs[-1].append({
				# "type": "xy",
				# 'l':0.03,
				# 'r':0.03,
				# 't':0.03,
				# 'b':0.03,
			})
			if colnum == 1 and row_titles_top:
				subplot_titles.append(scope.tagged_shortname(row, wrap_width=18))
			else:
				subplot_titles.append(None)

	fig = make_subplots(
		rows=len(rows), cols=len(cols),
		shared_xaxes=True,
		shared_yaxes=True,
		vertical_spacing=(0.18 if row_titles_top else 0.1)/len(rows),
		horizontal_spacing=0.1/len(cols),
		subplot_titles=subplot_titles,
		specs=specs,
	)
	if row_titles_top:
		for rowtitle in fig['layout']['annotations']:
			rowtitle['x'] = 0
			rowtitle['xanchor'] = 'left'
	fig['layout']['height'] = size * len(rows) + 75
	fig['layout']['width'] = size * len(cols) + 100

	if figure_class is not None:
		fig = figure_class(fig)

	experiment_name = "Experiment"
	if data.index.name:
		experiment_name = data.index.name

	if selection is None:
		selection = pandas.Series(True, index=data.index)

	if selection is not None:
		n_selected = numpy.sum(selection)
		n_unselected = numpy.sum(~selection)
	else:
		n_selected = 0
		n_unselected = len(data)

	n = 0
	extra_y_ax = len(rows) * len(cols)

	# saved_bins = {}
	for rownum, row in enumerate(rows, start=1):
		for colnum, col in enumerate(cols, start=1):
			n += 1

			# x = perturb_categorical_df(data, col, suffix="numberize")
			# y = perturb_categorical_df(data, row, suffix="numberize")
			x_points = perturb_categorical_df(data, col)
			y_points = perturb_categorical_df(data, row)
			x_ticktext, x_tickvals, x_range = axis_info(data[col], range_padding=0.25, epsilon=0.25)
			y_ticktext, y_tickvals, y_range = axis_info(data[row], range_padding=0.25, epsilon=0.25)

			if row == col:
				extra_y_ax += 1

				import scipy.stats
				try:
					kde0 = scipy.stats.gaussian_kde(data[~selection][row])
					kde1 = scipy.stats.gaussian_kde(data[selection][row])
				except TypeError:
					kde0 = scipy.stats.gaussian_kde(data[~selection][row].cat.codes)
					kde1 = scipy.stats.gaussian_kde(data[selection][row].cat.codes)
				except ValueError:
					if selection.all():
						kde0 = lambda z: numpy.zeros_like(z)
						kde1 = scipy.stats.gaussian_kde(data[row])
					elif (~selection).all():
						kde0 = scipy.stats.gaussian_kde(data[row])
						kde1 = lambda z: numpy.zeros_like(z)
				x_fill = numpy.linspace(*x_range, 200)
				y_0 = kde0(x_fill)
				y_1 = kde1(x_fill)
				topline = max(y_0.max(), y_1.max())
				y_range_kde = (-0.07*topline, 1.07*topline)

				layout_updates = {}
				layout_updates[f'yaxis{extra_y_ax}'] = dict(
					domain=fig['layout'][f'yaxis{n}']['domain'],
					anchor=f'free',
					showticklabels=False,
					range=y_range_kde,
				)
				fig.update_layout(**layout_updates)

				fig.add_trace(
					go.Scatter(
						x=[],
						y=[],
						mode='markers',
						showlegend=False,
					),
					row=rownum, col=colnum,
				)

				fig.add_trace(
					go.Scatter(
						x=x_fill,
						y=y_0,
						yaxis=f"y{extra_y_ax}",
						xaxis=f"x{n}",
						showlegend=False,
						line_color=unselected_color.rgb(),
						fill='tozeroy',
					)
				)
				fig.add_trace(
					go.Scatter(
						x=x_fill,
						y=y_1,
						yaxis=f"y{extra_y_ax}",
						xaxis=f"x{n}",
						showlegend=False,
						line_color=selected_color.rgb(),
						fill='tozeroy',
					)
				)

			else:

				x_bins, x_range_ = _get_bins_and_range(x_ticktext, col, x_range, scope)
				y_bins, y_range_ = _get_bins_and_range(y_ticktext, row, y_range, scope)

				# saved_bins[(rownum, colnum)] = (x_bins, x_range_, y_bins, y_range_)
				cvs = ds.Canvas(plot_width=x_bins, plot_height=y_bins, x_range=x_range_, y_range=y_range_)

				_col = f"_{col}_perturb" if f"_{col}_perturb" in data.columns else col
				_row = f"_{row}_perturb" if f"_{row}_perturb" in data.columns else row

				agg1 = cvs.points(data[selection], _col, _row)
				agg0 = cvs.points(data[~selection], _col, _row)

				if x_ticktext is not None:
					x_arr = data[col].to_numpy().astype('U')
					x_hovertag = "%{meta[2]}"
				else:
					x_arr = None
					x_hovertag = "%{x:.3s}"

				if y_ticktext is not None:
					y_arr = data[row].to_numpy().astype('U')
					y_hovertag = "%{meta[3]}" if x_hovertag=="%{meta[2]}" else "%{meta[2]}"
				else:
					y_arr = None
					y_hovertag = "%{y:.3s}"
				if with_hover:
					hovertemplate = (
						f"<b>{scope.shortname(col)}</b>: {x_hovertag}<br>" +
						f"<b>{scope.shortname(row)}</b>: {y_hovertag}" +
						"<extra>%{meta[0]} selected<br>%{meta[1]} unselected</extra>"
					)
				else:
					hovertemplate = (
						f"<b>{scope.shortname(col)}</b>: {{x}}<br>" +
						f"<b>{scope.shortname(row)}</b>: {{y}}"
					)
				agg0_arr = numpy.asanyarray(agg0)
				agg1_arr = numpy.asanyarray(agg1)
				wtype_def = [
					('ns', agg1_arr.dtype),
					('nu', agg0_arr.dtype),
				]
				if x_arr is not None:
					wtype_def.append(
						('x', x_arr.dtype)
					)
				if y_arr is not None:
					wtype_def.append(
						('y', y_arr.dtype)
					)
				wtype = numpy.dtype(wtype_def)
				if with_hover:
					meta = numpy.empty(agg0_arr.shape, dtype=wtype)
					meta['ns'] = agg1_arr
					meta['nu'] = agg0_arr
					if x_ticktext is not None:
						meta[:,1::2]['x']=x_ticktext
					if y_ticktext is not None:
						meta[1::2,:]['y']=numpy.asarray(y_ticktext)[:,None]
				else:
					meta = None
				y_label, x_label = agg0.dims[0], agg0.dims[1]
				# np.datetime64 is not handled correctly by go.Heatmap
				for ax in [x_label, y_label]:
					if numpy.issubdtype(agg0.coords[ax].dtype, numpy.datetime64):
						agg0.coords[ax] = agg0.coords[ax].astype(str)
				x = agg0.coords[x_label]
				y = agg0.coords[y_label]

				if not emph_selected:
					fig.add_trace(
						go.Image(
							z=_hue_mix(agg1, agg0, selected_color, unselected_color),
							x0=float(x[0]),
							dx=float(x[1]-x[0]),
							y0=float(y[0]),
							dy=float(y[1]-y[0]),
							hovertemplate=hovertemplate,
							meta=meta,
							colormodel='rgba',
						),
						row=rownum, col=colnum,
					)
				else:
					zmax = max(numpy.percentile(agg0_arr, 98), numpy.percentile(agg1_arr, 98))
					agg0_arr = agg0_arr.astype(numpy.float32)
					agg0_arr[agg0_arr==0] = numpy.nan
					fig.add_trace(
						go.Heatmap(
							x=x,
							y=y,
							z=agg0_arr,
							showlegend=False,
							hovertemplate=hovertemplate,
							meta=meta,
							coloraxis=f"coloraxis{n*2}",
							hoverongaps=False,
							zmax=zmax,
							zmin=0,
						),
						row=rownum, col=colnum,
					)

					agg1_arr = agg1_arr.astype(numpy.float32)
					agg1_arr[agg1_arr == 0] = numpy.nan
					fig.add_trace(
						go.Heatmap(
							x=x,
							y=y,
							z=agg1_arr,
							showlegend=False,
							hovertemplate=hovertemplate,
							meta=meta,
							coloraxis=f"coloraxis{n * 2 - 1}",
							hoverongaps=False,
							zmax=zmax,
							zmin=0,
						),
						row=rownum, col=colnum,
					)

					show_points = min(show_points, len(data)*show_points_frac)
					if n_selected <= show_points:
						_x_points_selected = x_points[selection]
						_y_points_selected = y_points[selection]
					else:
						_x_points_selected = [None]
						_y_points_selected = [None]

					if x_ticktext is not None or y_ticktext is not None:
						hovertemplate_s = (
								f'<b>{scope.shortname(row)}</b>: %{{meta[1]}}<br>' +
								f'<b>{scope.shortname(col)}</b>: %{{meta[2]}}' +
								f'<extra>{experiment_name} %{{meta[0]}}</extra>'
						)
						meta_s = data[selection][[row, col]].reset_index().to_numpy()
					else:
						hovertemplate_s = (
								f'<b>{scope.shortname(row)}</b>: %{{y}}<br>' +
								f'<b>{scope.shortname(col)}</b>: %{{x}}' +
								f'<extra>{experiment_name} %{{meta}}</extra>'
						)
						meta_s = multiindex_to_strings(data[selection].index)

					fig.add_trace(
						go.Scatter(
							x=_x_points_selected,
							y=_y_points_selected,
							mode='markers',
							marker=dict(
								color=selected_color.rgb(),
								size=marker_size,
							),
							showlegend=False,
							hovertemplate=hovertemplate_s,
							meta=meta_s,
						),
						row=rownum, col=colnum,
					)
					fig.update_layout({
						f"coloraxis{n*2-1}": {
							'showscale': False,
							'colorscale': [
								[0.0, selected_color.rgba(0.0)],
								[0.5, selected_color.rgba(0.6)],
								[1.0, selected_color.rgba(1.0)],
							],
							'cmax':zmax,
							'cmin':0,
						},
						f"coloraxis{n*2}": {
							'showscale': False,
							'colorscale': [
								[0.0, unselected_color.rgba(0.0)],
								[0.5, unselected_color.rgba(0.6)],
								[1.0, unselected_color.rgba(1.0)],
							],
							'cmax': zmax,
							'cmin': 0,
						},
					})
				if on_select is not None:
					fig.data[-1].on_selection(lambda *args: on_select(col, row, *args))
				if on_deselect is not None:
					fig.data[-1].on_deselect(lambda *args: on_deselect(col, row, *args))


				if box is not None:
					shapes = _splom_part_boxes(
						box, n,
						x, col, x_tickvals, x_ticktext, x_range,
						y, row, y_tickvals, y_ticktext, y_range,
						background_opacity=0.05,
					)
					for s in shapes:
						fig.add_shape(s)

				if refpoint is not None:
					shapes = _splom_part_ref_point(
						n,
						x, x_range, x_tickvals, x_ticktext, refpoint.get(col, None),
						y, y_range, y_tickvals, y_ticktext, refpoint.get(row, None),
					)
					for s in shapes:
						fig.add_shape(s)

			if colnum == 1:
				fig.update_yaxes(
					range=y_range,
					row=rownum,
					col=colnum,
				)
				if not row_titles_top:
					fig.update_yaxes(
						title_text=scope.tagged_shortname(row, wrap_width=18),
						row=rownum,
						col=colnum,
					)
				if y_ticktext is not None:
					fig.update_yaxes(
						row=rownum,
						col=colnum,
						tickmode = 'array',
						ticktext = y_ticktext,
						tickvals = y_tickvals,
					)
			# elif (colnum-1)%3==0 and len(cols)>4:
			# 	fig.update_yaxes(
			# 		title_text=scope.tagged_shortname(row, wrap_width=18),
			# 		title_font_size=7,
			# 		title_standoff=0,
			# 		row=rownum,
			# 		col=colnum,
			# 	)
			if rownum == len(rows):
				fig.update_xaxes(
					title_text=scope.tagged_shortname(col, wrap_width=18),
					row=rownum,
					col=colnum,
					range=x_range,
				)
				if x_ticktext is not None:
					fig.update_xaxes(
						row=rownum,
						col=colnum,
						tickmode='array',
						ticktext=x_ticktext,
						tickvals=x_tickvals,
					)
		# elif rownum%3==0:
			# 	fig.update_xaxes(
			# 		title_text=scope.tagged_shortname(col, wrap_width=18),
			# 		title_font_size=7,
			# 		title_standoff=2,
			# 		row=rownum,
			# 		col=colnum,
			# 	)
	fig.update_layout(margin=dict(
		l=10, r=10, t=30 if row_titles_top else 10, b=10,
	))

	metadata = dict(
		rows=rows,
		cols=cols,
		selected_color=selected_color.rgb(),
		unselected_color=unselected_color.rgb(),
		emph_selected=emph_selected,
		show_points=show_points,
		# saved_bins=saved_bins,
		marker_size=marker_size,
		row_titles=row_titles,
		size=size,
		refpoint=refpoint,
	)
	fig.update_layout(meta=metadata)
	return fig

def update_hmm_figure(
		scope,
		data,
		fig,
		selection,
		box,
		mass=None,
		rows=None,
		cols=None,
		selected_color=None,
		unselected_color=None,
):
	existing_emph_selected = fig['layout']['meta']['emph_selected']
	existing_show_points = fig['layout']['meta']['show_points']
	# saved_bins = fig['layout']['meta']['saved_bins']

	existing_rows = fig['layout']['meta']['rows']
	existing_cols = fig['layout']['meta']['cols']
	if selected_color is None:
		selected_color =colors.Color(
			fig['layout']['meta'].get('selected_color'),
			default=colors.DEFAULT_HIGHLIGHT_COLOR_RGB,
		)
	if unselected_color is None:
		unselected_color = colors.Color(
			fig['layout']['meta'].get('unselected_color'),
			default=colors.DEFAULT_BASE_COLOR_RGB,
		)
	if rows is None:
		rows = existing_rows
	if cols is None:
		cols = existing_cols

	change_dims = False
	if rows is None:
		rows = existing_rows
	else:
		if rows != existing_rows:
			change_dims = True
	if cols is None:
		cols = existing_cols
	else:
		if cols != existing_cols:
			change_dims = True
	if change_dims:
		new_fig = new_hmm_figure(
			scope,
			data,
			rows=rows,
			cols=cols,
			row_titles=fig['layout']['meta'].get('row_titles', 'side'),
			size=fig['layout']['meta'].get('size', None),
			selection=selection,
			box=box,
			refpoint=fig['layout']['meta'].get('refpoint', None),
			figure_class=None,
			on_select=None,  # lambda *a: self._on_select_from_histogram(*a,name=col)
			on_deselect=None,  # lambda *a: self._on_deselect_from_histogram(*a,name=col)
			selected_color=selected_color,
			unselected_color=unselected_color,
			marker_size=fig['layout']['meta'].get('marker_size', 3),
		)
		fig['data'] = new_fig['data']
		fig['layout'] = new_fig['layout']
		return fig

	replacement = new_hmm_figure(
		scope,
		data,
		rows=rows,
		cols=cols,
		row_titles=fig['layout']['meta']['row_titles'],
		size=fig['layout']['meta']['size'],
		selection=selection,
		box=box,
		refpoint=fig['layout']['meta']['refpoint'],
		figure_class=None,
		on_select=None,
		on_deselect=None,
		selected_color=selected_color,
		unselected_color=unselected_color,
		emph_selected=existing_emph_selected,
		show_points=existing_show_points,
		marker_size=fig['layout']['meta']['marker_size'],
	)

	copy_features = {
		'heatmap': ('name', 'x','y','z','showlegend','hovertemplate','meta','coloraxis','hoverongaps','zmax','zmin'),
		'scatter': ('name', 'x','y','mode','marker','showlegend','hovertemplate','meta'),
	}

	for old_trace, new_trace in zip(fig['data'], replacement['data']):
		classname = old_trace.__class__.__name__
		if classname == 'dict':
			classname = old_trace.get('type', 'NO_TYPE')
		for attr in copy_features[classname.lower()]:
			try:
				replace_attr = getattr(new_trace, attr)
			except AttributeError:
				pass
			else:
				if isinstance(old_trace, dict):
					old_trace[attr] = replace_attr
				else:
					setattr(old_trace, attr, replace_attr)

	fig['layout'] = replacement['layout']

	return fig


import textwrap
def _wrap_with_br(text, width=70, **kwargs):
	return "<br>   ".join(textwrap.wrap(text, width=width, **kwargs))

def new_parcoords_figure(
		scope,
		data,
		coords=(),
		flip_dims=(),
		robustness_functions=None,
		color_dim=None,
		color_data=None,
		colorscale=None,
		show_colorscale_bar=None,
		title=None,
		selection=None,
		unselected_color=None,
		selected_color=None,
		figure_class=None,
):
	"""
	Generate a parallel coordinates figure.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot.
	scope : emat.Scope
		Categorical levers and uncertainties are extracted from the scope.
	"""

	def _make_axis_list(j):
		if isinstance(j, str):
			if set('XLM').issuperset(j.upper()):
				use = []
				for i in j.upper():
					if i=='X':
						use += scope.get_uncertainty_names()
					elif i=='L':
						use += scope.get_lever_names()
					if i=='M':
						use += scope.get_measure_names()
				return use
			return [j]
		return j
	coords = _make_axis_list(coords)
	coords = [i for i in coords if i in data.columns]

	if unselected_color is None:
		unselected_color = colors.DEFAULT_BASE_COLOR
	if selected_color is None:
		selected_color = colors.DEFAULT_HIGHLIGHT_COLOR

	if show_colorscale_bar is None and colorscale is None:
		colorscale = [[0, unselected_color], [1, selected_color]]
		show_colorscale_bar = False

	# Change the range from plain min/max to something else
	column_ranges = {}
	tickvals = {}
	ticktext = {}

	df = data[coords].copy()
	flips = set(flip_dims)

	from ....viz.parcoords import _prefix_symbols
	prefix_chars = _prefix_symbols(scope, robustness_functions)

	for colnum, col in enumerate(coords, start=1):
		df[col] = perturb_categorical_df(data, col)
		x_ticktext, x_tickvals, x_range = axis_info(data[col], range_padding=0.3)
		col_range = column_ranges[col] = x_range
		if x_tickvals is not None:
			tickvals[col] = [col_range[0]] + list(x_tickvals) + [col_range[1]]
			ticktext[col] = [""] + [str(i) for i in x_ticktext] + [""]

	from ....workbench.em_framework.outcomes import ScalarOutcome

	# flip all MINIMIZE outcomes (or unflip them if previously marked as flip)
	if robustness_functions is not None:
		for k in robustness_functions:
			if k.kind == ScalarOutcome.MINIMIZE:
				if k.name in flips:
					flips.remove(k.name)
				else:
					flips.add(k.name)
	if scope is not None:
		for k in scope.get_measures():
			if k.kind == ScalarOutcome.MINIMIZE:
				if k.name in flips:
					flips.remove(k.name)
				else:
					flips.add(k.name)

	parallel_dims = [
		dict(
			range=column_ranges.get(
				col, [
					df[col].min(),
					df[col].max(),
				] if col not in flips else [
					df[col].max(),
					df[col].min(),
				]
			),
			label=_wrap_with_br(prefix_chars.get(col, '') + (col if scope is None else scope.shortname(col)),
								width=24),
			values=df[col],
			tickvals=tickvals.get(col, None),
			ticktext=ticktext.get(col, None),
			name=col,
		)
		for col in df.columns if col not in scope.get_constant_names()
	]

	## Line coloring dimension
	if isinstance(color_dim, int):
		color_dim = df.columns[color_dim]

	color_data_min = None
	color_data_max = None
	if color_data is None and selection is not None:
		color_data = selection.astype(float)
		color_data_min = 0
		color_data_max = 1

	if color_dim is not None:
		if color_dim in df.columns:
			color_data = df[color_dim]

	if color_data is None:
		parallel_line = dict(
			showscale=False,
		)
	elif isinstance(color_data, list):
		parallel_line = dict(
			color=color_data,
			showscale=False,
		)
	else:
		parallel_line = dict(
			color=color_data,
			colorscale=colorscale,
			showscale=show_colorscale_bar,
			reversescale=False,
			cmin=color_data_min if color_data_min is not None else color_data.min(),
			cmax=color_data_max if color_data_max is not None else color_data.max(),
			colorbar_title_text=getattr(color_data, 'name', ''),
			colorbar_title_side='right',
		)

	pc = go.Parcoords(
		line=parallel_line,
		dimensions=parallel_dims,
		labelfont=dict(
			color="#AA0000",
		),
		labelangle=-90,
		domain=dict(
			y=[0, 0.7],
		)
	)

	fig = go.Figure(
		[pc],
		layout=dict(
			title=title,
		)
	)

	if figure_class is not None:
		fig = figure_class(fig)
	return fig



def _to_thing(s, thing=float):
	if s == 'up' or s is None:
		return None
	try:
		return thing(s)
	except ValueError:
		return None


def convert_rangestring_to_tuple(rs, thing=get_float):
	if rs == 'any value':
		return (None, None)
	dn, up = None, None
	for sep in ["to", "and", "--", "—", ","]:
		if sep in rs:
			dn, up = rs.split(sep)
			dn = dn.strip()
			up = up.strip()
			break
	return (_to_thing(dn, thing), _to_thing(up, thing))

def convert_bounds_to_rangestring(b):
	if b is None:
		return ""
	if b.lowerbound is None:
		if b.upperbound is None:
			return ""
		else:
			return "up to " + si_units(b.upperbound)
	else: # b.lowerbound is not None
		if b.upperbound is None:
			return si_units(b.lowerbound) + " and up"
		else:
			return si_units(b.lowerbound) + " to " + si_units(b.upperbound)

def convert_set_to_rangestring(s):
	included = []
	if not s:
		return ""
	for i in s:
		i_ = str(i)
		if "," in i_:
			included.append(f'"{i_}"')
		else:
			included.append(i_)
	return ", ".join(included)

def convert_rangestring_to_set(rs):
	if rs == 'any value' or rs is None or rs == '':
		return None

	if '"' in rs:
		import shlex
		result = set(
			i.strip('"')
			for i in shlex.shlex(instream=rs, punctuation_chars=",;")
			if i != ","
		)
	else:
		result = set(
			i.strip()
			for i in rs.split(',')
		)
	return result
