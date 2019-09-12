
import seaborn as sns
from ema_workbench.analysis import feature_scoring
from ..viz import heatmap_table


def feature_scores(
		scope,
		design,
		return_type='figure',
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
		return_type ({'figure', 'dataframe', 'styled'}):
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
	import pandas

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

	fs = feature_scoring.get_feature_scores_all(inputs, outcomes, random_state=random_state)

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
