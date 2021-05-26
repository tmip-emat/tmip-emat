
import seaborn as sns
import pandas
import numpy
import warnings
from ..workbench.analysis import feature_scoring
from ..viz import heatmap_table
from ..scope.box import Box
from ..util.arg_processing import design_check
from ..exceptions import MissingMeasuresWarning, MissingMeasuresError


def feature_scores(
		scope,
		design,
		return_type='styled',
		db=None,
		random_state=None,
		cmap='viridis',
		measures=None,
		shortnames=None,
):
	"""
	Calculate feature scores based on a design of experiments.

	Args:
		scope (emat.Scope): The scope that defines this analysis.
		design (str or pandas.DataFrame): The name of the design
			of experiments to use for feature scoring, or a single
			pandas.DataFrame containing the experimental design and
			results.
		return_type ({'styled', 'figure', 'dataframe'}):
			The format to return, either a heatmap figure as an SVG
			render in and xmle.Elem, or a plain pandas.DataFrame,
			or a styled dataframe.
		db (emat.Database): If `design` is given as a string,
			extract the experiments from this database.
		random_state (int or numpy.RandomState, optional):
			Random state to use.
		cmap (string or colormap, default 'viridis'): matplotlib
			colormap to use for rendering.
		measures (Collection, optional): The performance measures
			on which feature scores are to be generated.  By default,
			all measures are included.
		shortnames (Scope or callable):
			If given, use this function to convert the measure
			names into more readable `shortname` values from the
			scope, or by using a function that maps measures
			names to something else.

	Returns:
		xmle.Elem or pandas.DataFrame:
			Returns a rendered SVG as xml, or a DataFrame,
			depending on the `return_type` argument.

	This function internally uses feature_scoring from the EMA Workbench,
	which in turn scores features using the "extra trees" regression
	approach.
	"""

	design = design_check(design, scope, db)

	# Split design into inputs and outcomes
	inputs = design[[c for c in design.columns if c in scope.get_parameter_names()]]
	outcomes = design[[c for c in design.columns if c in scope.get_measure_names()]]

	# Remove experiments that have no outcomes in any dimension
	missing_outcomes_experiments = outcomes.isna().all(axis=1)
	if missing_outcomes_experiments.any():
		n_missing = missing_outcomes_experiments.sum()
		if n_missing == len(outcomes):
			raise MissingMeasuresError(
				f"All {n_missing} experiments have no outcomes",
			)
		inputs = inputs[~missing_outcomes_experiments]
		outcomes = outcomes[~missing_outcomes_experiments]
		warnings.warn(
			f"There are {n_missing} experiments "
			f"with no outcomes, ignoring them",
			stacklevel=2,
			category=MissingMeasuresWarning,
		)

	# prepare to remove input columns with NaN's
	drop_inputs = list(inputs.columns[pandas.isna(inputs).sum()>0])

	# prepare to remove constant inputs
	for c in scope.get_constant_names():
		if c in inputs.columns and c not in drop_inputs:
			drop_inputs.append(c)

	# prepare to remove outcome columns with NaN's,
	drop_outcomes = list(outcomes.columns[pandas.isna(outcomes).sum()>0])

	# prepare to remove outcomes that have been removed from the scope
	scope_measures = set(scope.get_measure_names())
	for c in outcomes.columns:
		if c not in scope_measures and c not in drop_outcomes:
			drop_outcomes.append(c)
		if measures is not None and c not in measures and c not in drop_outcomes:
			drop_outcomes.append(c)

	# execute removals
	outcomes_ = outcomes.drop(columns=drop_outcomes)
	inputs_ = inputs.drop(columns=drop_inputs)

	# use workbench to compute feature scores
	fs = feature_scoring.get_feature_scores_all(inputs_, outcomes_, random_state=random_state)

	# restore original row/col ordering
	orig_col_order = [c for c in outcomes.columns if c in scope_measures]
	fs = fs.reindex(index=inputs.columns, columns=orig_col_order)

	# remove columns and rows that are entirely NA
	drop_c = list(fs.columns[(~pandas.isna(fs)).sum() == 0])
	drop_r = list(fs.index[(~pandas.isna(fs)).sum(axis=1) == 0])
	fs = fs.drop(index=drop_r, columns=drop_c)

	if shortnames is not None:
		if shortnames is True:
			shortnames = scope
		from ..scope.scope import Scope
		if isinstance(shortnames, Scope):
			fs.columns = fs.columns.map(shortnames.shortname)
		else:
			fs.columns = fs.columns.map(shortnames)

	# convert colormap to a light color palette for rendered outputs
	if 'figure' in return_type.lower() or 'styled' in return_type.lower():
		try:
			cmap = sns.light_palette(cmap, as_cmap=True)
		except ValueError:
			pass

	# create output based on `return_type`
	if 'figure' in return_type.lower():
		return heatmap_table(
			fs.T,
			xlabel='Model Parameters', ylabel='Performance Measures',
			title='Feature Scoring' + (f' [{design.design_name_}]' if design.design_name_ else ''),
			cmap=cmap,
		)
	elif return_type.lower() == 'styled':
		from ..util.styling import feature_score_styling
		return feature_score_styling(fs.T, cmap=cmap)
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
	design = design_check(design, scope, db)

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

def threshold_feature_scores(
		scope,
		measure_name,
		design,
		return_type='styled',
		*,
		db=None,
		random_state=None,
		cmap='viridis',
		z_min=0,
		z_max=1,
		min_tail=5,
		max_breaks=20,
		break_spacing='linear',
):
	"""
	Compute and display threshold feature scores for a performance measure.

	This function is useful to detect and understand non-linear relationships
	between performance measures and various input parameters.

	Args:
		scope (emat.Scope): The scope that defines this analysis.
		measure_name (str): The name of an individual performance
			measure to analyze.
		design (str or pandas.DataFrame): The name of the design
			of experiments to use for feature scoring, or a single
			pandas.DataFrame containing the experimental design and
			results.
		return_type (str):
			The format to return:
			- 'dataframe' gives a plain pandas.DataFrame,
			- 'styled' gives a colorized pandas.DataFrame,
			- 'figure' gives a plotly violin plot,
			- 'ridge figure' gives a plotly ridgeline figure.
			Either plotly result can optionally have ".svg"
			or ".png" added to render a static image in those
			formats.
		db (emat.Database): If `design` is given as a string,
			extract the experiments from this database.
		random_state (int or numpy.RandomState, optional):
			Random state to use.
		cmap (string or colormap, default 'viridis'): matplotlib
			colormap to use for rendering. Ignored if `return_type`
			is 'dataframe'.
		z_min, z_max (float, optional): Trim the bottom and top of
			the colormap range, respectively.  Defaults to (0,1) which
			will make the most relevant overall feature colored at the
			top of the colorscale and the least relevant feature at
			the bottom.
		min_tail (int, default 5): The minimum number of observations
			on each side of any threshold point. If this value is too
			small, the endpoint feature scoring results are highly
			unstable, but if it is too large then important nonlinearities
			near the extreme points may not be detected.  This is also
			used as the minimum average number of observations between
			threshold points.
		max_breaks (int, default 20): The maximum number of distinct
			threshold points to use.  Setting this value higher improves
			resolution but also requires more computational time.
		break_spacing ({'linear', 'percentile'}): How to distribute
			threshold breakpoints to test within the min-max range.

	Returns:
		plotly.Figure or DataFrame or styled DataFrame

	"""

	design = design_check(design, scope, db)

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
	name_order = []
	for name in scope.get_parameter_names():
		if name in result.index:
			name_order.append(name)
	for name in result.index:
		if name not in name_order:
			name_order.append(name)
	result = result.reindex(index=name_order)

	if return_type.lower() == 'styled':
		return result.style.background_gradient(cmap=cmap, axis=0, text_color_threshold=0.5)

	if 'figure' in return_type.lower():
		import plotly.graph_objects as go
		from matplotlib import cm

		base_score = feature_scores(
			scope=scope,
			design=design,
			return_type='dataframe',
			db=None,
			random_state=random_state,
			measures=[measure_name],
		)

		traces = []
		max_base_score = base_score.max().max()
		tick_values = []
		tick_labels = []
		colormap = getattr(cm, cmap, cm.viridis)


		if 'ridge' in return_type.lower():
			ridge = True
			gap = numpy.percentile(result.values.flatten(), 95)
			linewidth = 3
			area_alpha = 1.0
		else:
			ridge = False
			gap = numpy.percentile(result.values.flatten(), 95) * 2
			linewidth = 2
			area_alpha = 1.0

		for n_reversed in range(len(result)):
			n = len(result) - n_reversed - 1
			bs = base_score.loc[measure_name, result.index[n]] / max_base_score
			if numpy.isnan(bs):
				bs = 0
			bs = bs * (z_max-z_min) + z_min
			color = colormap(bs, bytes=True)
			dark_color, light_color = _darker_and_lighter_color(numpy.asarray(color)/255)
			linecolor_ = ", ".join(str(i) for i in dark_color[:3])
			areacolor_ = ", ".join(str(i) for i in light_color[:3])
			traces.append(
				go.Scatter(
					y=(numpy.zeros(len(result.columns)) if ridge else -result.iloc[n]) + n * gap,
					x=result.columns,
					fillcolor='rgba(0,0,0,0)',
					visible=True,
					showlegend=False,
					line=dict(color=f'rgba({linecolor_}, 1.0)', width=0 if ridge else linewidth),
					name=result.index[n],
					hovertemplate='%{meta}<br>Rel Import: %{customdata:.3f}<extra>'+measure_name+': %{x:.3s}</extra>',
					meta=[result.index[n]],
					customdata=result.iloc[n],
				)
			)
			traces.append(
				go.Scatter(
					y=result.iloc[n] + n * gap,
					x=result.columns,
					fill='tonexty',
					name=result.index[n],
					fillcolor=f'rgba({areacolor_}, {area_alpha})',
					line=dict(color=f'rgba({linecolor_}, 1.0)', width=linewidth),
					hovertemplate='%{meta}<br>Rel Import: %{customdata:.3f}<extra>'+measure_name+': %{x:.3s}</extra>',
					meta=[result.index[n]],
					customdata=result.iloc[n],
				)
			)
			tick_values.append(n * gap + (gap/3 if ridge else 0))
			tick_labels.append(result.index[n])

		fig = go.Figure()
		fig.add_traces(traces)

		fig.update_layout(
			xaxis_title_text=scope.shortname(measure_name),
			yaxis_showgrid=False,
			yaxis_zeroline=False,
			yaxis_tickvals=tick_values,
			yaxis_ticktext=tick_labels,
			yaxis_tickmode='array',
			showlegend=False,
			margin=dict(t=0,b=0,l=0,r=0),
		)
		from ..util.rendering import render_plotly
		return render_plotly(fig, return_type)

	return result



def _max_luminosity_color(color, max_lum=0.333, bytes=False):
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	new_c = colorsys.hls_to_rgb(c[0], min(c[1], max_lum), c[2])
	if bytes:
		lev = lambda x: max(0,min(255,int(numpy.round(x*255))))
		return tuple(lev(i) for i in new_c)
	else:
		return new_c

def _darker_and_lighter_color(color, lum_diff=0.3, bytes=False):
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	hls = colorsys.rgb_to_hls(*mc.to_rgb(c))
	dark = hls[1] * (1-lum_diff)
	light = dark + lum_diff
	new_dark_c = colorsys.hls_to_rgb(hls[0], dark, hls[2])
	new_light_c = colorsys.hls_to_rgb(hls[0], light, hls[2])
	if bytes:
		lev = lambda x: max(0,min(255,int(numpy.round(x*255))))
		return tuple(lev(i) for i in new_dark_c), tuple(lev(i) for i in new_light_c)
	else:
		return new_dark_c, new_light_c