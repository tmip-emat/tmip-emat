
import numpy
import plotly.graph_objs as go

from ema_workbench.em_framework.parameters import Category, CategoricalParameter, BooleanParameter
from ema_workbench.em_framework.outcomes import ScalarOutcome


def perturb(x, epsilon=0.05):
	return x + numpy.random.uniform(-epsilon, epsilon)


def parallel_coords(
		df,
		model=None,
		flip_dims=(),
		robustness_functions=(),
		color_dim=0,
		colorscale='Viridis',
		title=None,
):
	"""Generate a parallel coordinates figure.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot.
	model : ema_workbench.Model, optional
		Categorical levers and uncertainties are extracted from the model.
	"""

	df = df.copy(deep=True)

	# if model is not None:
	# 	categorical_parameters = [
	# 		i for i in model.levers if isinstance(i, CategoricalParameter)
	# 	] + [
	# 		i for i in model.uncertainties if isinstance(i, CategoricalParameter)
	# 	]
	# else:
	# 	categorical_parameters = []

	categorical_parameters = df.columns[df.dtypes == 'category']
	bool_columns = df.columns[df.dtypes == bool]

	# Change the range from plain min/max to something else
	column_ranges = {}
	tickvals = {}
	ticktext = {}

	prefix_chars = {}
	for rf in robustness_functions:
		if rf.kind < 0:
			prefix_chars[rf.name] = '⊖ '
		elif rf.kind > 0:
			prefix_chars[rf.name] = '⊕ '
		elif rf.kind == 0:
			prefix_chars[rf.name] = '⊙ '
	for col in model.levers.keys():
		prefix_chars[col] = '⎆ ' # ୰


	for c in categorical_parameters:
		df[c] = df[c].apply(lambda z: z.value if isinstance(z,Category) else z)
		n_cats = len(df[c].cat.categories)
		min_cat, max_cat = 0, n_cats-1
		col_range = column_ranges[c] = [min_cat-0.1, max_cat+0.1]
		tickvals[c] = [col_range[0]] + list(range(min_cat, max_cat+1)) + [col_range[1]]
		ticktext[c] = [""] + [str(i) for i in df[c].cat.categories] + [""]
		df[c] = df[c].cat.codes.apply( lambda x: perturb(x) )

	for c in bool_columns:
		df[c] = df[c].astype(float)
		column_ranges[c] = [-0.1, 1.1]
		tickvals[c] = [-0.1, 0, 1, 1.1]
		ticktext[c] = ["", "False", "True",""]
		df[c] = df[c].apply( lambda x: perturb(x) )

	flips = set(flip_dims)

	# flip all MINIMIZE outcomes (or unflip them if previously marked as flip)
	for k in robustness_functions:
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
			label=prefix_chars.get(col, '')+col,
			values=df[col],
			tickvals=tickvals.get(col, None),
			ticktext=ticktext.get(col, None),
		)
		for col in df.columns
	]

	## Line coloring dimension
	if isinstance(color_dim, int):
		color_dim = df.columns[color_dim]

	parallel_line = dict(
		color=df[color_dim],
		colorscale=colorscale,
		showscale=True,
		reversescale=True,
		cmin=df[color_dim].min(),
		cmax=df[color_dim].max(),
	)

	pc = go.Parcoords(
		line=parallel_line,
		dimensions=parallel_dims,
		labelfont=dict(
			color="#AA0000",
		),
	)

	return go.FigureWidget(
		[pc],
		layout=dict(
			title=title,
		)
	)

