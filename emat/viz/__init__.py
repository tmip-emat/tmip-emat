
from ..database.database import Database
from .scatter import scatter_graph_row, ScatterMass
from ..util import xmle
from plotly.colors import DEFAULT_PLOTLY_COLORS
import itertools

COLOR_BLUE = "rgb(31, 119, 180)"
COLOR_RED = 'rgb(227, 20, 20)'
COLOR_GREEN = "rgb(44, 160, 44)"


def _pick_color(scope, x, y):
	lev = scope.get_lever_names()
	unc = scope.get_uncertainty_names()
	if y in lev:
		if x in lev:
			return COLOR_BLUE
		if x in unc:
			return COLOR_RED
		return COLOR_BLUE
	elif y in unc:
		if x in lev:
			return COLOR_BLUE
		if x in unc:
			return COLOR_RED
		return COLOR_RED
	else: # y in meas
		if x in lev:
			return COLOR_BLUE
		if x in unc:
			return COLOR_RED
		return COLOR_GREEN


def scatter_graphs(
		column,
		data,
		scope,
		db=None,
		contrast='infer',
		marker_opacity=None,
		mass=1000,
		render=None,
		use_gl=True,
):
	"""
	Generate a row of scatter plots comparing one column against others.

	Args:
		column (str):
			The name of the principal parameter or measure to analyze.
		data (pandas.DataFrame or str): The experimental results to plot.
			Can be given as a DataFrame or as the name of a design (in
			which case the results are loaded from the provided `db`).
		scope (Scope, optional): The exploratory scope.
		db (Database, optional):
			The database containing the results. This is ignored unless
			`data` is a string.
		contrast (str or list):
			The contrast columns to plot the principal parameter or
			measure against.  Can be given as a list of columns that
			appear in the data, or one of {'uncertainties', 'levers',
			'parameters', 'measures', 'infer'}. If set to 'infer', the
			contrast will be 'measures' if `column` is a parameter,
			and 'parameters' if `columns` is a measure.
		marker_opacity (float, optional):
			The opacity to use for markers. If the number of markers is
			large, the figure may appear as a solid blob; by setting opacity
			to less than 1.0, the figure can more readily show relative
			density in various regions.  If not specified, marker_opacity
			is set based on `mass` instead.
		mass (int or emat.viz.ScatterMass, default 1000):
			The target number of rendered points in each figure. Setting
			to a number less than the number of experiments will make
			each scatter point partially transparent, which will help
			visually convey relative density when there are a very large
			number of points.
		render (str or dict, optional):
			If given, the graph[s] will be rendered to a static image
			using `plotly.io.to_image`.  For default settings, pass
			'png', or give a dictionary that specifies keyword arguments
			to that function.  See `emat.util.rendering.render_plotly`
			for more details.
		use_gl (bool, default True):
			Use Plotly's `Scattergl` instead of `Scatter`, which may
			provide some performance benefit for large data sets.

	Returns:
		FigureWidget or xmle.Elem

	Raises:
		ValueError: If `contrast` is 'infer' but `column` is neither a parameter
			nor a measure.
	"""

	if contrast == 'infer':
		if column in scope.get_uncertainty_names():
			contrast = 'measures'
		elif column in scope.get_lever_names():
			contrast = 'measures'
		elif column in scope.get_measure_names():
			contrast = 'parameters'
		else:
			raise ValueError('cannot infer what to contrast against')

	if contrast == 'uncertainties':
		contrast = scope.get_uncertainty_names()
	elif contrast == 'levers':
		contrast = scope.get_lever_names()
	elif contrast == 'parameters':
		contrast = scope.get_uncertainty_names() + scope.get_lever_names()
	elif contrast == 'measures':
		contrast = scope.get_measure_names()

	if isinstance(data, str):
		if db is None:
			raise ValueError('db cannot be None if data is a design name')
		data = db.read_experiment_all(data)

	if isinstance(mass, int):
		mass = ScatterMass(mass)

	if marker_opacity is None:
		marker_opacity = mass.get_opacity(data)

	y_title = column
	if scope is not None:
		y_title = scope.shortname(y_title)

	contrast_cols = [c for c in contrast if c in data.columns]
	contrast_color = [_pick_color(scope, c, column) for c in contrast_cols]

	fig = scatter_graph_row(
		contrast_cols,
		column,
		df = data,
		marker_opacity=marker_opacity,
		y_title=y_title,
		layout=dict(
			margin=dict(l=50, r=2, t=5, b=40)
		),
		short_name_func=scope.shortname if scope is not None else None,
		use_gl=use_gl,
		C=contrast_color,
	)

	if render:
		if render == 'png':
			render = dict(format='png', width=200*len(contrast_cols), height=270, scale=2)

		if render == 'svg':
			render = dict(format='svg', width=200*len(contrast_cols), height=270)

		from ..util.rendering import render_plotly
		return render_plotly(fig, render)

	return fig


def scatter_graphs_2(
		column,
		datas,
		scope,
		db=None,
		contrast='infer',
		render=None,
		colors=None,
		use_gl=True,
		mass=1000,
):
	"""
	Generate a row of scatter plots comparing multiple datasets.

	This function is similar to `scatter_graphs`, but accepts
	multiple data sets and plots them using different colors.

	Args:
		column (str):
			The name of the principal parameter or measure to analyze.
		datas (Collection[pandas.DataFrame or str]):
			The experimental results to plot. Can be given as a DataFrame
			or as the name of a design (in which case the results are
			loaded from the provided Database `db`).
		scope (Scope, optional): The exploratory scope.
		db (Database, optional):
			The database containing the results.  Ignored unless `data`
			is a string.
		contrast (str or list):
			The contrast columns to plot the principal parameter or
			measure against.  Can be given as a list of columns that
			appear in the data, or one of {'uncertainties', 'levers',
			'parameters', 'measures', 'infer'}. If set to 'infer',
			the contrast will be 'measures' if `column` is a parameter,
			and 'parameters' if `columns` is a measure.
		render (str or dict, optional):
			If given, the graph[s] will be rendered to a static image
			using `plotly.io.to_image`.  For default settings, pass
			'png', or give a dictionary that specifies keyword arguments
			to that function.  See `emat.util.rendering.render_plotly`
			for more details.
		mass (int or emat.viz.ScatterMass, default 1000):
			The target number of rendered points in each figure. Setting
			to a number less than the number of experiments will make
			each scatter point partially transparent, which will help
			visually convey relative density when there are a very large
			number of points.

	Returns:
		plotly.FigureWidget or xmle.Elem:
			The latter is returned if a `render` argument is used.

	Raises:
		ValueError:
			If `contrast` is 'infer' but `column` is neither a parameter
			nor a measure.
	"""

	if contrast == 'infer':
		if column in scope.get_uncertainty_names():
			contrast = 'measures'
		elif column in scope.get_lever_names():
			contrast = 'measures'
		elif column in scope.get_measure_names():
			contrast = 'parameters'
		else:
			raise ValueError('cannot infer what to contrast against')

	if contrast == 'uncertainties':
		contrast = scope.get_uncertainty_names()
	elif contrast == 'levers':
		contrast = scope.get_lever_names()
	elif contrast == 'parameters':
		contrast = scope.get_uncertainty_names() + scope.get_lever_names()
	elif contrast == 'measures':
		contrast = scope.get_measure_names()

	if isinstance(mass, int):
		mass = ScatterMass(mass)

	data_ = []
	marker_opacity_ = []
	fig = 'widget'
	if colors is None:
		colorcycle = itertools.cycle(DEFAULT_PLOTLY_COLORS)
	else:
		colorcycle = itertools.cycle(colors)

	for data in datas:
		if isinstance(data, str):
			if db is None:
				raise ValueError('db cannot be None if data is a design name')
			data = db.read_experiment_all(data)
		data_.append(data)

	for data in data_:
		marker_opacity_.append(mass.get_opacity(data))

		y_title = column
		try:
			y_title = scope[column].shortname
		except AttributeError:
			pass

		fig = scatter_graph_row(
			[(c if c in data.columns else None) for c in contrast],
			column,
			df = data,
			marker_opacity=marker_opacity_[-1],
			y_title=y_title,
			layout=dict(
				margin=dict(l=50, r=2, t=5, b=40)
			),
			output=fig,
			C=colorcycle.__next__(),
			short_name_func=scope.shortname if scope is not None else None,
			use_gl=use_gl,
		)

	if render:
		if render == 'png':
			render = dict(format='png', width=1400, height=270, scale=2)

		if render == 'svg':
			render = dict(format='svg', width=1400, height=270)

		from ..util.rendering import render_plotly
		return render_plotly(fig, render)

	return fig


def heatmap_table(
		data, cmap='viridis', fmt='.3f', linewidths=0.7, figsize=(12,3),
		xlabel=None, ylabel=None, title=None,
		attach_metadata=True,
		scale_color_by_row=True,
		**kwargs
):
	"""
	Generate a SVG heatmap from data.

	Args:
		data (pandas.DataFrame): source data for the heatmap table.
		cmap (str): A colormap for the resulting heatmap.
		fmt (str): how to format values
		linewidths (float): Line widths for the table.
		figsize (tuple): The size of the resulting figure.
		xlabel, ylabel, title (str, optional): Captions for each.
		attach_metadata (bool, default True): Attach `data` to the
			resulting figure as metadata.
		scale_color_by_row (bool, default True):
			Color rows independently.

	Returns:
		xmle.Elem: The xml data for a svg rendering.
	"""
	import seaborn as sns
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots(figsize=figsize)
	if scale_color_by_row:
		coloring = data.div(data.max(1), 0)
	else:
		coloring = data
	axes = sns.heatmap(
		coloring, ax=ax, cmap=cmap, annot=data,
		fmt=fmt, linewidths=linewidths,
		cbar=not scale_color_by_row, **kwargs,
	)
	if xlabel:
		ax.set_xlabel(xlabel, fontweight='bold')
	if ylabel is not None:
		ax.set_ylabel(ylabel, fontweight='bold')
	if title is not None:
		ax.set_title(title, fontweight='bold')
	result= xmle.Elem.from_figure(axes.get_figure())
	if attach_metadata:
		result.metadata = data
	return result
