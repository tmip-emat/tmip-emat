
from ..workbench.em_framework.samplers import sample_uncertainties, sample_levers
import platypus
import numpy

from .optimization_result import OptimizationResult
from ..model.core_model import AbstractCoreModel
from ..scope.measure import Measure
from ..scope.parameter import CategoricalParameter


def robust_optimize(
		model,
		robustness_functions,
		scenarios,
		evaluator=None,
		nfe=10000,
		convergence='default',
		display_convergence=True,
		convergence_freq=100,
		constraints=None,
		epsilons=0.1,
		algorithm=None,
		check_extremes=False,
		**kwargs,
):
	"""
	Perform robust optimization.

	The robust optimization generally a multi-objective optimization task.
	It is undertaken using statistical measures of outcomes evaluated across
	a number of scenarios, instead of using the individual outcomes themselves.
	For each candidate policy, the model is evaluated against all of the considered
	scenarios, and then the robustness measures are evaluated using the
	set of outcomes from the original runs.  The robustness measures
	are aggregate measures that are computed from a set of outcomes.
	For example, this may be expected value, median, n-th percentile,
	minimum, or maximum value of any individual outcome.  It is also
	possible to have joint measures, e.g. expected value of the larger
	of outcome 1 or outcome 2.

	Each robustness function is indicated as a maximization or minimization
	target, where higher or lower values are better, respectively.
	The optimization process then tries to identify one or more
	non-dominated solutions for the possible policy levers.

	Args:
		model (AbstractCoreModel): A core model to use for
			robust optimization.
		robustness_functions (Collection[Measure]): A collection of
			aggregate statistical performance measures.
		scenarios (int or Collection): A collection of scenarios to
			use in the evaluation(s), or give an integer to generate
			that number of random scenarios.
		evaluator (Evaluator, optional): The evaluator to use to
			run the model. If not given, a SequentialEvaluator will
			be created.
		nfe (int, default 10_000): Number of function evaluations.
			This generally needs to be fairly large to achieve stable
			results in all but the most trivial applications.
		convergence ('default', None, or emat.optimization.ConvergenceMetrics):
			A convergence display during optimization.
		display_convergence (bool, default True): Automatically display
			the convergence metric figures when optimizing.
		constraints (Collection[Constraint], optional)
			Solutions will be constrained to only include values that
			satisfy these constraints. The constraints can be based on
			the policy levers, or on the computed values of the robustness
			functions, or some combination thereof.
		epsilons ('auto' or float or array-like): Used to limit the number of
			distinct solutions generated.  Set to a larger value to get
			fewer distinct solutions.  When 'auto', epsilons are set based
			on the standard deviations of a preliminary set of experiments.
		algorithm (platypus.Algorithm or str, optional): Select an
			algorithm for multi-objective optimization.  The algorithm can
			be given directly, or named in a string. See `platypus`
			documentation for details.
		check_extremes (bool or int, default False): Conduct additional
			evaluations, setting individual policy levers to their
			extreme values, for each candidate Pareto optimal solution.
		kwargs: any additional arguments will be passed on to the
			platypus algorithm.

	Returns:
		emat.OptimizationResult:
			The set of non-dominated solutions found.
			When `convergence` is given, the convergence measures are
			included, as a pandas.DataFrame in the `convergence` attribute.

	Raises:
		ValueError:
			If any of the `robustness_functions` are not emat.Measures, or
			do not have a function set, or share a name with any parameter,
			measure, constant, or performance measure in the scope.
		KeyError:
			If any of the `robustness_functions` relies on a named variable
			that does not appear in the scope.
	"""
	if not isinstance(model, AbstractCoreModel):
		raise ValueError(f'model must be AbstractCoreModel subclass, not {type(model)}')

	for rf in robustness_functions:
		if not isinstance(rf, Measure):
			raise ValueError(f'robustness functions must be defined as emat.Measure objects')
		if rf.function is None:
			raise ValueError(f'robustness function must have a function attribute set ({rf.name})')
		if rf.name in model.scope:
			raise ValueError(f'cannot name robustness function the same as any scope name ({rf.name})')
		for rf_v in rf.variable_name:
			if rf_v not in model.scope:
				raise KeyError(rf_v)

	if constraints:
		for c in constraints:
			for pn in c.parameter_names:
				if pn in model.scope.get_uncertainty_names():
					raise ValueError(f"cannot constrain on uncertainties ({c.name})")

	epsilons, convergence, display_convergence, evaluator = model._common_optimization_setup(
		epsilons, convergence, display_convergence, evaluator
	)

	if algorithm is None:
		algorithm = platypus.EpsNSGAII
	if isinstance(algorithm, str):
		algorithm = getattr(platypus, algorithm, algorithm)
		if isinstance(algorithm, str):
			raise ValueError(f"platypus algorithm {algorithm} not found")
	if not issubclass(algorithm, platypus.Algorithm):
		raise ValueError(f"algorithm must be a platypus.Algorithm subclass, not {algorithm}")

	if isinstance(scenarios, int):
		n_scenarios = scenarios
		scenarios = sample_uncertainties(model, n_scenarios)


	with evaluator:
		if epsilons == 'auto':
			trial = model.robust_evaluate(
				robustness_functions=robustness_functions,
				scenarios=scenarios,
				policies=30,
				evaluator=evaluator,
			)
			epsilons = [max(0.1, numpy.std(trial[rf.name]) / 20) for rf in robustness_functions]

		robust_results = evaluator.robust_optimize(
			robustness_functions,
			scenarios,
			nfe=nfe,
			constraints=constraints,
			epsilons=epsilons,
			convergence=convergence,
			convergence_freq=convergence_freq,
			algorithm=algorithm,
			**kwargs,
		)

	if isinstance(robust_results, tuple) and len(robust_results) == 2:
		robust_results, result_convergence = robust_results
	else:
		result_convergence = None

	robust_results = model.ensure_dtypes(robust_results)

	result = OptimizationResult(
		robust_results,
		result_convergence,
		scope=model.scope,
		robustness_functions=robustness_functions,
		scenarios=scenarios,
	)

	if check_extremes:
		result.check_extremes(
			model,
			1 if check_extremes is True else check_extremes,
			evaluator=evaluator,
			constraints=constraints,
		)

	return result

