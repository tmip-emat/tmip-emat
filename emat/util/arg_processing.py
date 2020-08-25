
import pandas as pd
from ..experiment.experimental_design import ExperimentalDesign

def design_check(design, scope=None, db=None):
	"""
	Ensure a design is an ExperimentalDesign not just a name.

	Args:
		design (str or pd.DataFrame):
			The design to check, given either as a
			name for a design of experiments stored
			in the database, or as a DataFrame containing
			the design itself.
		scope (emat.Scope, optional):
			The scope cooresponding to the design that is
			stored in the database.
		db (emat.Database, optional):
			The database from which to extract the design of
			experiments.  If the design is given as a DataFrame,
			this argument is ignored.

	Returns:
		design (emat.ExperimentalDesign)
	"""
	if isinstance(design, str):
		if db is None:
			raise ValueError('must give db to use design name')
		design_name = design
		if scope is None:
			scope = db.read_scope()
		design = db.read_experiment_all(scope.name, design)
	elif isinstance(design, pd.DataFrame):
		design_name = None
	else:
		raise TypeError('must name design or give DataFrame')

	if not isinstance(design, ExperimentalDesign):
		design = ExperimentalDesign(design)
		if scope is not None:
			design.scope = scope
		if design_name is not None:
			design.design_name = design_name

	return design