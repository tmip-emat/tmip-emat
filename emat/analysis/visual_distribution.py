
from emat.viz import scatter_graphs, scatter_graphs_2
from emat.viz.scatter import ScatterMass
from IPython.display import HTML, display, display_html


def _shorten_category_names(scope, experiment_results, original_cats):
	for k in scope.get_all_names():
		abbrev = getattr(scope[k], 'abbrev', {})
		if k in experiment_results.columns:
			try:
				is_cat = experiment_results[k].dtype == 'category'
			except TypeError:
				is_cat = False
			if is_cat:
				original_cats[k] = experiment_results[k].cat.categories
				experiment_results[k].cat.categories = [abbrev.get(i, i) for i in experiment_results[k].cat.categories]
	return experiment_results, original_cats


def _restore_category_names(experiment_results, original_cats):
	for k, v in original_cats.items():
		try:
			experiment_results[k].cat.categories = v
		except:
			print(k)
			print(experiment_results[k].cat.categories)
			print(v)
			raise


def display_experiments(
		scope,
		experiment_results=None,
		db=None,
		render='png',
		rows='measures',
		columns='infer',
		mass=1000,
		use_gl=True,
		return_figures=False,
):
	"""
	Render a visualization of experimental results.

	This function will display the outputs in a jupyter notebook,
	but does not actually return any values.

	Parameters
	----------
	scope : emat.Scope
		The scope to use in identifying parameters and performance
		measures.
	experiment_results : pandas.DataFrame or str
		The complete results from a set of experiments,
		including parameter inputs and performance measure
		outputs.  Give a string to name a design in the database.
	db : emat.Database, optional
		When either of the `experiments` arguments is given as
		a string, the experiments are loaded from this database
		using the scope name as well as the given string as the
		design name.
	render : str or dict or None, default 'png'
		If given, the graph[s]
		will be rendered to a static image using `plotly.io.to_image`.
		For default settings, pass 'png', or give a dictionary
		that specifies keyword arguments to that function. If no
		rendering is done (by setting `render` to None), the raw
		plotly figures are returned -- this may result in a very
		large number of javascript figures and may slow down your
		browser.
	rows : {'measures', 'levers', 'uncertainties'} or Collection, default 'measures'
		Give a named group to generate a row of figures for each
		item in that group, or give a collection of individual
		names to generate a row of figures for each named item.
	columns : {'infer', 'measures', 'levers', 'uncertainties'} or Collection, default 'infer'
		Give a named group to generate a column of plots for each
		item in that group, or give a collection of individual
		names to generate a column of plots for each named item.
		The default 'infer' value will select all parameters when
		the row is a measure, and all measures otherwise.
	mass : int or emat.viz.ScatterMass, default 1000
		The target number of rendered points in each figure. Setting
		to a number less than the number of experiments will make
		each scatter point partially transparent, which will help
		visually convey relative density when there are a very large
		number of points.
	return_figures : bool, default False
		Set this to True to return the FigureWidgets instead of
		simply displaying them.
	"""

	if isinstance(experiment_results, str):
		if db is None:
			raise ValueError("must give experiments or db and design_name")
		experiment_results = db.read_experiment_all(
			scope_name=scope.name,
			design_name=experiment_results,
		)

	if rows == 'measures':
		rows = scope.get_measure_names()
	elif rows == 'levers':
		rows = scope.get_lever_names()
	elif rows == 'uncertainties':
		rows = scope.get_uncertainty_names()
	
	original_cats = {}
	try:
		experiment_results, original_cats = _shorten_category_names(scope, experiment_results, original_cats)

		figures = {}
		for row in rows:
			try:
				fig = scatter_graphs(
					row,
					experiment_results,
					scope=scope,
					render=render,
					use_gl=use_gl,
					mass=mass,
					contrast=columns,
				)
			except KeyError:
				continue
			try:
				fig.update_layout(height=250)
			except KeyboardInterrupt:
				raise
			except:
				pass
			if return_figures:
				figures[row] = fig
			else:
				display_html(f'<h4 title="{scope.get_description(row)}">{scope.shortname(row)}</h4>', raw=True)
				display(fig)

		if return_figures:
			return figures

	finally:
		_restore_category_names(experiment_results, original_cats)



def contrast_experiments(
		scope,
		experiments_1,
		experiments_2,
		db=None,
		render='png',
		rows='measures',
		columns='infer',
		mass=1000,
		colors=None,
		use_gl=True,
		return_figures=False,
):
	"""
	Render a visualization of two sets of experimental results.

	This function will display the outputs in a Jupyter notebook,
	but does not actually return any values.

	Args:
		scope (emat.Scope):
			The scope to use in identifying parameters and
			performance measures.
		experiments_1, experiments_2 (str or pandas.DataFrame):
			The complete results from a set of experiments,
			including parameter inputs and performance measure
			outputs.  Give a string to name a design in the
			database instead of passing results explicitly as
			a DataFrame.
		db (emat.Database, optional):
			When either of the `experiments` arguments is given as
			a string, the experiments are loaded from this database
			using the scope name as well as the given string as the
			design name.
		render (str or dict or None, default 'png'):
			If given, the graph[s] will be rendered to a static
			image using `plotly.io.to_image`. For default settings,
			pass 'png', or give a dictionary that specifies keyword
			arguments to that function. If no rendering is done (by
			setting `render` to None), the raw plotly figures are
			returned -- this may result in a very large number of
			javascript figures and may slow down your browser.
		rows (str or Collection, default 'measures'):
			Give a named group {'measures', 'levers', 'uncertainties'}
			to generate a row of figures for each item in that group,
			or give a collection of individual names to generate a
			row of figures for each named item.
		columns (str or Collection, default 'infer'):
			Give a named group {'infer', 'measures', 'levers',
			'uncertainties'} to generate a column of plots for each
			item in that group, or give a collection of individual
			names to generate a column of plots for each named item.
			The default 'infer' value will select all parameters when
			the row is a measure, and all measures otherwise.
		mass (int or emat.viz.ScatterMass, default 1000):
			The target number of rendered points in each figure. Setting
			to a number less than the number of experiments will make
			each scatter point partially transparent, which will help
			visually convey relative density when there are a very large
			number of points.
		colors (2-tuple, optional):
			A pair of colors for the experiments.
		return_figures (bool, default False):
			Set this to True to return the figures instead of
			simply displaying them within a Jupyter notebook.
	"""

	if isinstance(experiments_1, str):
		if db is None:
			raise ValueError("must give experiments or db and design_name")
		experiments_1 = db.read_experiment_all(
			scope_name=scope.name,
			design_name=experiments_1,
		)

	if isinstance(experiments_2, str):
		if db is None:
			raise ValueError("must give experiments or db and design_name")
		experiments_2 = db.read_experiment_all(
			scope_name=scope.name,
			design_name=experiments_2,
		)

	# check that categories align with scope
	scope.ensure_cat_ordering(experiments_1)
	scope.ensure_cat_ordering(experiments_2)

	if rows == 'measures':
		rows = scope.get_measure_names()
	elif rows == 'levers':
		rows = scope.get_lever_names()
	elif rows == 'uncertainties':
		rows = scope.get_uncertainty_names()

	# if measures is None:
	# 	measures = scope.get_measure_names()

	original_cats1 = {}
	original_cats2 = {}
	try:
		experiments_1, original_cats1 = _shorten_category_names(scope, experiments_1, original_cats1)
		experiments_2, original_cats2 = _shorten_category_names(scope, experiments_2, original_cats2)

		figures = {}
		for row in rows:
			if not return_figures:
				display_html(f'<h4 title="{scope.get_description(row)}">{scope.shortname(row)}</h4>', raw=True)
			fig = scatter_graphs_2(
				row,
				[experiments_1, experiments_2],
				scope=scope,
				render=render,
				use_gl=use_gl,
				mass=mass,
				contrast=columns,
				colors=colors,
			)
			if return_figures:
				figures[row] = fig
			else:
				display(fig)

		if return_figures:
			return figures

	finally:
		_restore_category_names(experiments_1, original_cats1)
		_restore_category_names(experiments_2, original_cats2)
