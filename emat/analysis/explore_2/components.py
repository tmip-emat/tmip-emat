
import pandas, numpy
from plotly import graph_objs as go

X_FIGURE_BUFFER = 0.03

from ...viz import colors
from ... import styles

def _y_maximum(fig):
	return sum(t.y for t in fig.select_traces()).max()


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
	if unselected_color is None:
		unselected_color = colors.DEFAULT_BASE_COLOR
	if selected_color is None:
		selected_color = colors.DEFAULT_HIGHLIGHT_COLOR
	if figure_class is None:
		figure_class = go.FigureWidget

	if bins is None:
		bins = 20
	bar_heights, bar_x = numpy.histogram(data_column, bins=bins)
	bins_left = bar_x[:-1]
	bins_width = bar_x[1:] - bar_x[:-1]
	bar_heights_select, bar_x = numpy.histogram(data_column[selection], bins=bar_x)

	fig = figure_class(
		data=[
			go.Bar(
				x=bins_left,
				y=bar_heights_select,
				width=bins_width,
				name='Inside',
				marker_color=selected_color,
				marker_line_width=marker_line_width,
				hoverinfo='skip',
			),
			go.Bar(
				x=bins_left,
				y=bar_heights - bar_heights_select,
				width=bins_width,
				name='Outside',
				marker_color=unselected_color,
				marker_line_width=marker_line_width,
				hoverinfo='skip',
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
	fig.layout.xaxis.range = (
		x_range[0] - x_width * X_FIGURE_BUFFER,
		x_range[1] + x_width * X_FIGURE_BUFFER,
	)
	# self._figures_hist[col] = fig
	# self._draw_boxes_on_figure(col)
	return fig

def update_histogram_figure(
		fig,
		selection,
		data_column,
		rerange_y=False,
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
	bins = list(fig['data'][0]['x'])
	bins.append(fig['data'][0]['x'][-1] + fig['data'][0]['width'][-1])
	bar_heights, bar_x = numpy.histogram(data_column, bins=bins)
	bar_heights_select, bar_x = numpy.histogram(data_column[selection], bins=bar_x)
	fig['data'][0]['y'] = bar_heights_select
	fig['data'][1]['y'] = bar_heights - bar_heights_select
	if rerange_y:
		_y_max = numpy.max(bar_heights)
		fig['layout']['yaxis']['range'] = (
			-_y_max * 0.03,
			_y_max * 1.05,
		)
	return fig



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
		figure_class=None,
		label_name_map=None,
):
	if unselected_color is None:
		unselected_color = colors.DEFAULT_BASE_COLOR
	if selected_color is None:
		selected_color = colors.DEFAULT_HIGHLIGHT_COLOR
	if figure_class is None:
		figure_class = go.FigureWidget
	if label_name_map is None:
		label_name_map = {}

	v = data_column.astype(
		pandas.CategoricalDtype(categories=labels, ordered=False)
	).cat.codes
	bar_heights, bar_x = numpy.histogram(v, bins=numpy.arange(0, len(labels) + 1))
	bar_heights_select, _ = numpy.histogram(v[selection], bins=numpy.arange(0, len(labels) + 1))

	labels = [label_name_map.get(i, i) for i in labels]
	fig = figure_class(
		data=[
			go.Bar(
				x=labels,
				y=bar_heights_select,
				name='Inside',
				marker_color=selected_color,
				marker_line_width=marker_line_width,
				hoverinfo='none',
			),
			go.Bar(
				x=labels,
				y=bar_heights - bar_heights_select,
				name='Outside',
				marker_color=unselected_color,
				marker_line_width=marker_line_width,
				hoverinfo='none',
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
		),
	)
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
		-0.5,
		len(fig.data[0].x) - 0.5
	)
	x_width = x_range[1] - x_range[0]
	fig.layout.xaxis.range = (
		x_range[0] - x_width * X_FIGURE_BUFFER,
		x_range[1] + x_width * X_FIGURE_BUFFER,
	)
	return fig

def update_frequencies_figure(
		fig,
		selection,
		data_column,
		rerange_y=False,
):
	labels = list(fig['data'][0]['x'])
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
	return fig

from ...viz.perturbation import perturb_categorical

def categorical_ticks(x, range_padding=0.0):
	if hasattr(x, 'dtype'):
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
	return None, None, None

def perturb_categorical_df(df, col=None):
	if col is None:
		cols = list(df.columns)
	else:
		cols = [col]
	for i in cols:
		if f"_{i}_perturb" in df.columns: continue
		x, x_ticktext, x_tickvals, x_range, _ = perturb_categorical(df[i])
		if x_ticktext is not None:
			df[f"_{i}_perturb"] = x
	if col is not None:
		if f"_{col}_perturb" in df.columns:
			return df[f"_{col}_perturb"]
		else:
			return df[col]
	return df

def new_splom(
		scope,
		data,
		rows="LX",
		cols="M",
		use_gl=True,
		mass=1000,
		row_titles='top',
		size=150,
		selection=None,
):
	from plotly.subplots import make_subplots
	import plotly.graph_objects as go
	from ...viz import _pick_color

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
				subplot_titles.append(scope.shortname(row))
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
		# x_title="Performance Measures",
		# y_title="Parameters",
	)
	if row_titles_top:
		for rowtitle in fig['layout']['annotations']:
			rowtitle['x'] = 0
			rowtitle['xanchor'] = 'left'
	fig['layout']['height'] = size * len(rows) + 75
	fig['layout']['width'] = size * len(cols) + 100

	Scatter = go.Scattergl if use_gl else go.Scatter

	if isinstance(mass, int):
		from ...viz import ScatterMass
		mass = ScatterMass(mass)

	if selection is None:
		marker_opacity = mass.get_opacity(data)
	else:
		mo = [mass.get_opacity(data[~selection]), mass.get_opacity(data[selection])]
		marker_opacity = pandas.Series(data=mo[0], index=data.index)
		marker_opacity[selection] = mo[1]
	experiment_name = "Experiment"
	if data.index.name:
		experiment_name = data.index.name

	for rownum, row in enumerate(rows, start=1):
		for colnum, col in enumerate(cols, start=1):

			if selection is None:
				color = _pick_color(scope, row, col)
			else:
				color = pandas.Series(data=colors.DEFAULT_BASE_COLOR, index=data.index)
				color[selection] = colors.DEFAULT_HIGHLIGHT_COLOR

			x = perturb_categorical_df(data, col)
			y = perturb_categorical_df(data, row)
			x_ticktext, x_tickvals, x_range = categorical_ticks(data[col], range_padding=0.3)
			y_ticktext, y_tickvals, y_range = categorical_ticks(data[row], range_padding=0.3)

			if x_ticktext is not None or y_ticktext is not None:
				hovertemplate = (
					f'<b>{scope.shortname(row)}</b>: %{{meta[1]}}<br>' +
					f'<b>{scope.shortname(col)}</b>: %{{meta[2]}}' +
					f'<extra>{experiment_name} %{{meta[0]}}</extra>'
				)
				meta = data[[row,col]].reset_index().to_numpy()
			else:
				hovertemplate = (
					f'<b>{scope.shortname(row)}</b>: %{{y}}<br>' +
					f'<b>{scope.shortname(col)}</b>: %{{x}}' +
					f'<extra>{experiment_name} %{{meta}}</extra>'
				)
				meta = data.index

			fig.add_trace(
				Scatter(
					x=x,
					y=y,
					mode='markers',
					marker=dict(
						# size=s,
						# sizemode=sizemode,
						# sizeref=sizeref,
						# sizemin=sizemin,
						opacity=marker_opacity,
						color=color,
					),
					showlegend=False,
					hovertemplate=hovertemplate,
					meta=meta,
				),
				row=rownum, col=colnum,
			)
			if colnum == 1:
				if not row_titles_top:
					fig.update_yaxes(
						title_text=scope.shortname(row),
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
						range = y_range,
					)
			# elif (colnum-1)%3==0 and len(cols)>4:
			# 	fig.update_yaxes(
			# 		title_text=scope.shortname(row),
			# 		title_font_size=7,
			# 		title_standoff=0,
			# 		row=rownum,
			# 		col=colnum,
			# 	)
			if rownum == len(rows):
				fig.update_xaxes(
					title_text=scope.shortname(col),
					row=rownum,
					col=colnum,
				)
				if x_ticktext is not None:
					fig.update_xaxes(
						row=rownum,
						col=colnum,
						tickmode='array',
						ticktext=x_ticktext,
						tickvals=x_tickvals,
						range=x_range,
					)
		# elif rownum%3==0:
			# 	fig.update_xaxes(
			# 		title_text=scope.shortname(col),
			# 		title_font_size=7,
			# 		title_standoff=2,
			# 		row=rownum,
			# 		col=colnum,
			# 	)
	fig.update_layout(margin=dict(
		l=10, r=10, t=30 if row_titles_top else 10, b=10,
	))
	return fig