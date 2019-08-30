
from emat.viz import scatter_graphs
from IPython.display import HTML, display, display_html

def display_experiments(
		scope,
		experiment_results=None,
		db=None,
		design_name=None,
		render='png',
		measures=None,
):
	"""
	Render a visualization of experimental results.

	Parameters
	----------
	scope : emat.Scope
	experiment_results : pandas.DataFrame, optional
	db : emat.Database, optional
	design_name : str, optional
	measures : Collection, optional
		A subset of measures to include.  If not given, all
		measures from the defined scope are included.
	"""

	if experiment_results is None:
		if db is None or design_name is None:
			raise ValueError("must give experiments or db and design_name")
		experiment_results = db.read_experiment_all(
			scope_name=scope.name,
			design_name=design_name,
		)

	if measures is None:
		measures = scope.get_measure_names()

	for meas in measures:
		display_html(f"<h4>{meas}</h4>", raw=True)
		display(scatter_graphs(meas, experiment_results, scope=scope, render=render))