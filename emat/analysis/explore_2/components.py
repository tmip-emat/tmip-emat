
import pandas, numpy
from plotly import graph_objs as go
from plotly.subplots import make_subplots

X_FIGURE_BUFFER = 0.03

from ...viz import colors, _pick_color
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

def axis_info(x, range_padding=0.0):
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
	x_range = [x.min(), x.max()]
	x_span = x_range[1] - x_range[0]
	if x_span <= 0: x_span = 1
	x_range = [x_range[0] - x_span*0.07, x_range[1] + x_span*0.07]
	return None, None, x_range

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

def new_splom_figure(
		scope,
		data,
		rows="LX",
		cols="M",
		use_gl=True,
		mass=1000,
		row_titles='top',
		size=150,
		selection=None,
		box=None,
		refpoint=None,
		figure_class=None,
		on_select=None,  # lambda *a: self._on_select_from_histogram(*a,name=col)
		on_deselect=None,  # lambda *a: self._on_deselect_from_histogram(*a,name=col)
):

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
	)
	if row_titles_top:
		for rowtitle in fig['layout']['annotations']:
			rowtitle['x'] = 0
			rowtitle['xanchor'] = 'left'
	fig['layout']['height'] = size * len(rows) + 75
	fig['layout']['width'] = size * len(cols) + 100

	if figure_class is not None:
		fig = figure_class(fig)

	Scatter = go.Scattergl if use_gl else go.Scatter

	marker_opacity = _splom_marker_opacity(
		data.index,
		selection,
		mass=mass,
	)

	if selection is None:
		marker_color = None
	else:
		marker_color = pandas.Series(data=colors.DEFAULT_BASE_COLOR, index=data.index)
		marker_color[selection] = colors.DEFAULT_HIGHLIGHT_COLOR

	experiment_name = "Experiment"
	if data.index.name:
		experiment_name = data.index.name

	n = 0
	for rownum, row in enumerate(rows, start=1):
		for colnum, col in enumerate(cols, start=1):
			n += 1
			if marker_color is None:
				color = _pick_color(scope, row, col)
			else:
				color = marker_color

			x = perturb_categorical_df(data, col)
			y = perturb_categorical_df(data, row)
			x_ticktext, x_tickvals, x_range = axis_info(data[col], range_padding=0.3)
			y_ticktext, y_tickvals, y_range = axis_info(data[row], range_padding=0.3)

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
			if on_select is not None:
				fig.data[-1].on_selection(lambda *args: on_select(col, row, *args))
			if on_deselect is not None:
				fig.data[-1].on_deselect(lambda *args: on_deselect(col, row, *args))

			if box is not None:
				shapes = _splom_part_boxes(
					box, n,
					x, col, x_tickvals, x_ticktext, x_range,
					y, row, y_tickvals, y_ticktext, y_range,
				)
				for s in shapes: fig.add_shape(s)

			if refpoint is not None:
				shapes = _splom_part_ref_point(
					n,
					x, x_range, refpoint.get(col, None),
					y, y_range, refpoint.get(row, None),
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
			# 		title_text=scope.shortname(col),
			# 		title_font_size=7,
			# 		title_standoff=2,
			# 		row=rownum,
			# 		col=colnum,
			# 	)
	fig.update_layout(margin=dict(
		l=10, r=10, t=30 if row_titles_top else 10, b=10,
	))
	fig.update_layout(meta=dict(
		rows=rows,
		cols=cols,
	))
	return fig




from ...scope.box import Bounds

def _splom_marker_opacity(
		data_index,
		selection,
		mass=1000,
):
	if isinstance(mass, int):
		from ...viz import ScatterMass
		mass = ScatterMass(mass)
	if selection is None:
		marker_opacity = mass.get_opacity(data_index)
	else:
		mo = [mass.get_opacity(data_index[~selection]), mass.get_opacity(data_index[selection])]
		marker_opacity = pandas.Series(data=mo[0], index=data_index)
		marker_opacity[selection] = mo[1]
	return marker_opacity

def _splom_part_boxes(
		box, ax_num,
		x, x_label, x_tickvals, x_ticktext, x_range,
		y, y_label, y_tickvals, y_ticktext, y_range,
):
	background_shapes, foreground_shapes = [], []


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
		x, x_range, x_refpoint,
		y, y_range, y_refpoint,
):
	foreground_shapes = []

	if x_refpoint is not None:
		if y_range is None:
			y_range = [y.min(), y.max()]
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
		if x_range is None:
			x_range = [x.min(), x.max()]
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
		mass=1000,
		rows=None,
		cols=None,
):
	existing_rows = fig['layout']['meta']['rows']
	existing_cols = fig['layout']['meta']['cols']
	if rows is None:
		rows = existing_rows
	else:
		raise NotImplementedError("TODO: if rows change, remake figure")
	if cols is None:
		cols = existing_cols
	else:
		raise NotImplementedError("TODO: if cols change, remake figure")
	existing_lines = [s for s in fig['layout']['shapes'] if s['type']=='line']
	data_index = fig['data'][0]['meta']
	marker_opacity = _splom_marker_opacity(
		data_index,
		selection,
		mass=mass,
	)
	if selection is None:
		marker_color = None
	else:
		marker_color = pandas.Series(data=colors.DEFAULT_BASE_COLOR, index=data_index)
		marker_color[selection] = colors.DEFAULT_HIGHLIGHT_COLOR
	n = 0
	box_shapes = []
	for rownum, row in enumerate(rows, start=1):
		for colnum, col in enumerate(cols, start=1):
			if marker_color is None:
				color = _pick_color(scope, row, col)
			else:
				color = marker_color
			x = perturb_categorical_df(data, col)
			y = perturb_categorical_df(data, row)
			x_ticktext, x_tickvals, x_range = axis_info(data[col], range_padding=0.3)
			y_ticktext, y_tickvals, y_range = axis_info(data[row], range_padding=0.3)
			fig['data'][n]['marker']['color'] = color
			fig['data'][n]['marker']['opacity'] = marker_opacity
			n += 1
			box_shapes.extend(_splom_part_boxes(
				box, n,
				x, col, x_tickvals, x_ticktext, x_range,
				y, row, y_tickvals, y_ticktext, y_range,
			))
	fig['layout']['shapes'] = box_shapes + existing_lines
	return fig
