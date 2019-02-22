
import pandas
from ema_workbench import Constraint

def batch_contraint_check(
		constraints,
		parameter_frame,
		outcome_frame,
		aggregate=True,
):
	"""
	Batch check of constraints

	Args:
		constraints (Collection[Constraint]):
			A collection of Constraints to evaluate.
		parameter_frame (pandas.DataFrame, optional):
			The parameters (uncertainties and levers) for a
			batch of experiments.
		outcome_frame (pandas.DataFrame, optional):
			The outcomes (performance measures) for a
			batch of experiments.  If both this and `parameter_frame`
			are given, they must have the same indexes.
		aggregate (bool, default True):
			Return a single boolean series that indicates whether
			all of the constraints are satisfied.  Otherwise,
			a pandas.DataFrame is returned with a column for every
			constraint.

	Returns:
		pandas.Series: If return_agg is True
		pandas.DataFrame: If return_agg is False

	Raises:
		KeyError:
			If a constraint in constraints calls for a parameter or
			outcome name that is not present in parameter_frame or
			outcome_frame, respectively.

	"""
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

	for c in constraints:

		assert isinstance(c, Constraint)

		constraint_data = pandas.concat([
			parameter_frame[c.parameter_names],
			outcome_frame[c.outcome_names],
		], axis=1)

		results[c.name] = (constraint_data.apply(c.process, axis=1) == 0)

	if aggregate:
		return results.all(axis=1)
	else:
		return results

