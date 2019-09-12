
from emat.viz import scatter_graphs, scatter_graphs_2
from emat.viz.scatter import ScatterMass
from IPython.display import HTML, display, display_html

def display_experiments(
		scope,
		experiment_results=None,
		db=None,
		render='png',
		measures=None,
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
	render (str or dict or None, default 'png'): If given, the graph[s]
		will be rendered to a static image using `plotly.io.to_image`.
		For default settings, pass 'png', or give a dictionary
		that specifies keyword arguments to that function. If no
		rendering is done (by setting `render` to None), the raw
		plotly figures are returned -- this may result in a very
		large number of javascript figures and may slow down your
		browser.
	measures : Collection, optional
		A subset of measures to include.  If not given, all
		measures from the defined scope are included.
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

	if measures is None:
		measures = scope.get_measure_names()

	figures = {
		meas:scatter_graphs(meas, experiment_results, scope=scope, render=render, use_gl=use_gl, mass=mass)
		for meas in measures
	}

	if return_figures:
		return figures

	for meas, fig in figures.items():
		display_html(f"<h4>{meas}</h4>", raw=True)
		display(fig)




def contrast_experiments(
		scope,
		experiments_1,
		experiments_2,
		db=None,
		render='png',
		measures=None,
		mass=1000,
		use_gl=True,
		return_figures=False,
):
	"""
	Render a visualization of two sets of experimental results.

	This function will display the outputs in a jupyter notebook,
	but does not actually return any values.

	Parameters
	----------
	scope : emat.Scope
		The scope to use in identifying parameters and performance
		measures.
	experiments_1, experiments_2 : str or pandas.DataFrame
		The complete results from a set of experiments,
		including parameter inputs and performance measure
		outputs.  Give a string to name a design in the database.
	db : emat.Database, optional
		When either of the `experiments` arguments is given as
		a string, the experiments are loaded from this database
		using the scope name as well as the given string as the
		design name.
	render (str or dict or None, default 'png'): If given, the graph[s]
		will be rendered to a static image using `plotly.io.to_image`.
		For default settings, pass 'png', or give a dictionary
		that specifies keyword arguments to that function. If no
		rendering is done (by setting `render` to None), the raw
		plotly figures are returned -- this may result in a very
		large number of javascript figures and may slow down your
		browser.
	measures : Collection, optional
		A subset of measures to include.  If not given, all
		measures from the defined scope are included.
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

	if measures is None:
		measures = scope.get_measure_names()

	figures = {
		meas:scatter_graphs_2(meas, [experiments_1, experiments_2], scope=scope, render=render, use_gl=use_gl)
		for meas in measures
	}

	if return_figures:
		return figures

	for meas, fig in figures.items():
		display_html(f"<h4>{meas}</h4>", raw=True)
		display(fig)

		from plotly.subplots import make_subplots