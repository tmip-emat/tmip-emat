
import pandas
from ..workbench import Constraint

def batch_contraint_check(
		constraints,
		parameter_frame,
		outcome_frame=None,
		aggregate=True,
		scope=None,
		only_parameters=False,
):
	"""
	Batch check of constraints

	Args:
		constraints (Collection[Constraint]):
			A collection of Constraints to evaluate.
		parameter_frame (pandas.DataFrame, optional):
			The parameters (uncertainties and levers) for a
			batch of experiments. If scope is given, this can be
			split automatically into parameter_frame and
			outcome_frame.
		outcome_frame (pandas.DataFrame, optional):
			The outcomes (performance measures) for a
			batch of experiments.  If both this and `parameter_frame`
			are given, they must have the same indexes.  If not
			given but the scope is given, the outcome_frame
			is created from `parameter_frame`.
		aggregate (bool, default True):
			Return a single boolean series that indicates whether
			all of the constraints are satisfied.  Otherwise,
			a pandas.DataFrame is returned with a column for every
			constraint.
		scope (Scope, optional):
			A Scope used to identify parameters and outcomes.
		only_parameters (bool, default False):
			Only check constraints based exclusively on parameters.

	Returns:
		pandas.Series: If return_agg is True
		pandas.DataFrame: If return_agg is False

	Raises:
		KeyError:
			If a constraint in constraints calls for a parameter or
			outcome name that is not present in parameter_frame or
			outcome_frame, respectively.

	"""
	if scope is not None and outcome_frame is None:
		_p, _o = [], []
		for col in parameter_frame.columns:
			if col in scope.get_parameter_names():
				_p.append(col)
			else:
				_o.append(col)
		parameter_frame, outcome_frame = parameter_frame[_p], parameter_frame[_o]

	if parameter_frame is None and outcome_frame is not None:
		parameter_frame = pandas.DataFrame(index=outcome_frame.index, columns=[])

	if parameter_frame is not None and outcome_frame is None:
		outcome_frame = pandas.DataFrame(index=parameter_frame.index, columns=[])

	assert len(parameter_frame) == len(outcome_frame)

	results = pandas.DataFrame(
		data=True,
		index=parameter_frame.index,
		columns=[c.name for c in constraints],
		dtype=bool,
	)

	if len(parameter_frame):
		for c in constraints:
			assert isinstance(c, Constraint)
			if only_parameters and c.outcome_names:
				continue
			constraint_data = pandas.concat([
				parameter_frame[c.parameter_names],
				outcome_frame[c.outcome_names],
			], axis=1)
			results[c.name] = (constraint_data.apply(c.process, axis=1) == 0)

	if aggregate:
		return results.all(axis=1)
	else:
		return results

