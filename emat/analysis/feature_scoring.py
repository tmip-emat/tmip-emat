
import seaborn as sns
import pandas
import numpy
from ..workbench.analysis import feature_scoring
from ..viz import heatmap_table
from ..scope.box import Box

def feature_scores(
		scope,
		design,
		return_type='styled',
		db=None,
		random_state=None,
		cmap='viridis',
):
	"""
	Calculate feature scores based on a design of experiments.

	Args:
		scope (emat.Scope): The scope that defines this analysis.
		design (str or pandas.DataFrame): The name of the design of experiments
			to use for feature scoring, or a single pandas.DataFrame containing the
			experimental design and results.
		return_type ({'styled', 'figure', 'dataframe'}):
			The format to return, either a heatmap figure as an SVG render in and
			xmle.Elem, or a plain pandas.DataFrame, or a styled dataframe.
		db (emat.Database): If `design` is given as a string, extract the experiments
			from this database.
		random_state (int or numpy.RandomState, optional):
			Random state to use.
		cmap (string or colormap, default 'viridis'): matplotlib colormap
			to use for rendering.

	Returns:
		xmle.Elem or pandas.DataFrame:
			Returns a rendered SVG as xml, or a DataFrame,
			depending on the `return_type` argument.

	This function internally uses feature_scoring from the EMA Workbench, which in turn
	scores features using the "extra trees" regression approach.
	"""

	if isinstance(design, str):
		if db is None:
			raise ValueError('must give db to use design name')
		design_name = design
		inputs = db.read_experiment_parameters(scope.name, design)
		outcomes = db.read_experiment_measures(scope.name, design)
	elif isinstance(design, pandas.DataFrame):
		design_name = None
		inputs = design[[c for c in design.columns if c in scope.get_parameter_names()]]
		outcomes = design[[c for c in design.columns if c in scope.get_measure_names()]]
	else:
		raise TypeError('must name design or give DataFrame')

	# remove input columns with NaN's
	drop_inputs = list(inputs.columns[pandas.isna(inputs).sum()>0])

	# remove constant inputs
	for c in scope.get_constant_names():
		if c in inputs.columns and c not in drop_inputs:
			drop_inputs.append(c)

	# remove outcome columns with NaN's,
	drop_outcomes = list(outcomes.columns[pandas.isna(outcomes).sum()>0])

	# remove outcomes that have been removed from the scope
	scope_measures = set(scope.get_measure_names())
	for c in outcomes.columns:
		if c not in scope_measures and c not in drop_outcomes:
			drop_outcomes.append(c)

	outcomes_ = outcomes.drop(columns=drop_outcomes)
	inputs_ = inputs.drop(columns=drop_inputs)

	fs = feature_scoring.get_feature_scores_all(inputs_, outcomes_, random_state=random_state)

	# restore original row/col ordering
	orig_col_order = [c for c in outcomes.columns if c in scope_measures]
	fs = fs.reindex(index=inputs.columns, columns=orig_col_order)

	if return_type.lower() in ('figure','styled'):
		try:
			cmap = sns.light_palette(cmap, as_cmap=True)
		except ValueError:
			pass

	if return_type.lower() == 'figure':
		return heatmap_table(
			fs.T,
			xlabel='Model Parameters', ylabel='Performance Measures',
			title='Feature Scoring' + (f' [{design_name}]' if design_name else ''),
			cmap=cmap,
		)
	elif return_type.lower() == 'styled':
		return fs.T.style.background_gradient(cmap=cmap, axis=1, text_color_threshold=0.5)
	else:
		return fs.T

def box_feature_scores(
		scope,
		box,
		design,
		return_type='styled',
		db=None,
		random_state=None,
		cmap='viridis',
		exclude_measures=True,
):
	"""
	Calculate feature scores for a box, based on a design of experiments.

	Args:
		scope (emat.Scope): The scope that defines this analysis.
		box (emat.Box): The box the defines the target feature.
		design (str or pandas.DataFrame): The name of the design of experiments
			to use for feature scoring, or a single pandas.DataFrame containing the
			experimental design and results.
		return_type ({'styled', 'figure', 'dataframe'}):
			The format to return, either a heatmap figure as an SVG render in and
			xmle.Elem, or a plain pandas.DataFrame, or a styled dataframe.
		db (emat.Database): If `design` is given as a string, extract the experiments
			from this database.
		random_state (int or numpy.RandomState, optional):
			Random state to use.
		cmap (string or colormap, default 'viridis'): matplotlib colormap
			to use for rendering.
		exclude_measures (bool, default True): Exclude measures from feature scoring.

	Returns:
		xmle.Elem or pandas.DataFrame:
			Returns a rendered SVG as xml, or a DataFrame,
			depending on the `return_type` argument.

	This function internally uses feature_scoring from the EMA Workbench, which in turn
	scores features using the "extra trees" classification approach.
	"""
	if isinstance(design, str):
		if db is None:
			raise ValueError('must give db to use design name')
		design_name = design
		design = db.read_experiment_all(scope.name, design)
	elif isinstance(design, pandas.DataFrame):
		design_name = None
	else:
		raise TypeError('must name design or give DataFrame')

	if exclude_measures:
		if not set(box.thresholds.keys()).intersection(scope.get_measure_names()):
			raise ValueError('no measures in box thresholds')

	target = box.inside(design)
	return target_feature_scores(
		scope,
		target,
		design,
		return_type=return_type,
		db=db,
		random_state=random_state,
		cmap=cmap,
		exclude_measures=exclude_measures,
		exclude_parameters=box.thresholds.keys(),
	)

def target_feature_scores(
		scope,
		target,
		design,
		return_type='styled',
		db=None,
		random_state=None,
		cmap='viridis',
		exclude_measures=True,
		exclude_parameters=None,
):
	"""
	Calculate feature scores for a target selection, based on a design of experiments.

	Args:
		scope (emat.Scope): The scope that defines this analysis.
		target (pandas.Series): The target feature, whose dtype should be bool.
		design (str or pandas.DataFrame): The name of the design of experiments
			to use for feature scoring, or a single pandas.DataFrame containing the
			experimental design and results.
		return_type ({'styled', 'figure', 'dataframe'}):
			The format to return, either a heatmap figure as an SVG render in and
			xmle.Elem, or a plain pandas.DataFrame, or a styled dataframe.
		db (emat.Database): If `design` is given as a string, extract the experiments
			from this database.
		random_state (int or numpy.RandomState, optional):
			Random state to use.
		cmap (string or colormap, default 'viridis'): matplotlib colormap
			to use for rendering.
		exclude_measures (bool, default True): Exclude measures from feature scoring.

	Returns:
		xmle.Elem or pandas.DataFrame:
			Returns a rendered SVG as xml, or a DataFrame,
			depending on the `return_type` argument.

	This function internally uses feature_scoring from the EMA Workbench, which in turn
	scores features using the "extra trees" classification approach.
	"""
	import pandas, numpy

	if isinstance(design, str):
		if db is None:
			raise ValueError('must give db to use design name')
		design_name = design
		design = db.read_experiment_all(scope.name, design)
	elif isinstance(design, pandas.DataFrame):
		design_name = None
	else:
		raise TypeError('must name design or give DataFrame')

	# remove design columns with NaN's
	drop_cols = set(design.columns[pandas.isna(design).sum()>0])

	# remove design columns not in the scope
	all_names = set(scope.get_all_names())
	for c in design.columns:
		if c not in all_names:
			drop_cols.add(c)

	# remove constants
	for c in scope.get_constant_names():
		if c in design.columns:
			drop_cols.add(c)

	# remove outcome columns if exclude_measures
	if exclude_measures:
		for meas in scope.get_measure_names():
			if meas in design.columns:
				drop_cols.add(meas)

	if exclude_parameters is not None:
		for meas in design.columns:
			if meas in exclude_parameters:
				drop_cols.add(meas)

	design_ = design.drop(columns=list(drop_cols))

	from ..workbench.analysis.scenario_discovery_util import RuleInductionType

	target_name = getattr(target, 'name', None)
	if not isinstance(target_name, str):
		target_name = 'target'

	fs = feature_scoring.get_feature_scores_all(
		design_,
		{target_name:target},
		random_state=random_state,
		mode=RuleInductionType.CLASSIFICATION,
	)

	# restore original row/col ordering
	# orig_col_order = [c for c in outcomes.columns if c in scope_measures]
	# fs = fs.reindex(
	# 	index=design.columns,
	# 	# columns=orig_col_order,
	# )

	if return_type.lower() in ('figure','styled'):
		try:
			cmap = sns.light_palette(cmap, as_cmap=True)
		except ValueError:
			pass

	if return_type.lower() == 'figure':
		return heatmap_table(
			fs.T,
			xlabel='Model Parameters', ylabel='Target',
			title='Feature Scoring' + (f' [{design_name}]' if design_name else ''),
			cmap=cmap,
		)
	elif return_type.lower() == 'styled':
		return fs.T.style.background_gradient(cmap=cmap, axis=1, text_color_threshold=0.5)
	else:
		return fs.T


def _col_breakpoints(
        data_col,
        min_tail=5,
        max_breaks=20,
		break_spacing='linear',
):
	arr = numpy.asarray(data_col).flatten()
	if arr.size < min_tail*2:
		raise ValueError("array too short for `min_tail`")
	arr_s = numpy.partition(arr, [min_tail,-min_tail])
	lo_end, hi_end = arr_s[[min_tail,-min_tail]]
	inside_size = arr.size - (min_tail*2)
	if arr.size == min_tail*2:
		inside_breaks = 1
	else:
		inside_breaks = max(min(int(numpy.ceil(inside_size/min_tail)), max_breaks),2)
	if break_spacing == 'linear':
		return numpy.linspace(lo_end, hi_end, inside_breaks)
	elif break_spacing == 'percentile':
		qtiles = numpy.linspace(min_tail/arr.size, (arr.size-min_tail)/arr.size, inside_breaks)
		return numpy.quantile(arr, qtiles)
	raise ValueError(f'unknown `break_spacing` value {break_spacing}')

def measure_marginal_feature_scores(
		scope,
		measure_name,
		design,
		return_type='styled',
		db=None,
		random_state=None,
		cmap='viridis',
		*,
		min_tail=5,
		max_breaks=20,
		break_spacing='linear',
):
	if isinstance(design, str):
		if db is None:
			raise ValueError('must give db to use design name')
		design_name = design
		design = db.read_experiment_all(scope.name, design)
	elif isinstance(design, pandas.DataFrame):
		design_name = None
	else:
		raise TypeError('must name design or give DataFrame')

	tracking = {}

	breakpoints = _col_breakpoints(
		design[measure_name],
		min_tail=min_tail,
		max_breaks=max_breaks,
		break_spacing=break_spacing,
	)

	for j in breakpoints:
		tracking[j] = dict(box_feature_scores(
			scope,
			Box(name="", lower_bounds={measure_name: j}),
			design,
			return_type='dataframe',
			db=db,
			random_state=random_state,
			exclude_measures=True,
		).iloc[0])

	result = pandas.DataFrame(tracking)
	if return_type.lower() == 'styled':
		return result.style.background_gradient(cmap=cmap, axis=0, text_color_threshold=0.5)

	if return_type.lower() == 'figure':
		import plotly.graph_objects as go
		import plotly.colors
		colorscheme = getattr(
			plotly.colors.qualitative,
			cmap,
			plotly.colors.qualitative.Light24
		)
		fig = go.Figure()
		for n, i in enumerate(result.index):
			fig.add_trace(go.Scatter(
				x=result.columns, y=result.loc[i],
				mode='lines',
				fillcolor=colorscheme[n % len(colorscheme)],
				line=dict(width=0, color=colorscheme[n % len(colorscheme)]),
				stackgroup='one',  # define stack group
				name=i,
				hovertemplate="%{y:.3f}",
			))
		fig.update_layout(hovermode="x unified", yaxis_range=(0, 1))
		return fig

	return result
