
from joblib import Memory
from ema_workbench.em_framework.samplers import sample_uncertainties, sample_levers

from .optimization_result import OptimizationResult
from ..model.core_model import AbstractCoreModel
from ..scope.measure import Measure

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
		algorithm (platypus.Algorithm, optional): Select an
			algorithm for multi-objective optimization.  See
			`platypus` documentation for details.
		nfe (int, default 10_000): Number of function evaluations.
			This generally needs to be fairly large to achieve stable
			results in all but the most trivial applications.
		convergence ('default', None, or emat.optimization.ConvergenceMetrics):
			A convergence display during optimization.
		constraints (Collection[Constraint], optional)
			Solutions will be constrained to only include values that
			satisfy these constraints. The constraints can be based on
			the policy levers, or on the computed values of the robustness
			functions, or some combination thereof.
		kwargs: any additional arguments will be passed on to the
			platypus algorithm.

	Returns:
		emat.OptimizationResult:
			The set of non-dominated solutions found.
			When `convergence` is given, the convergence measures are
			included, as a pandas.DataFrame in the `convergence` attribute.
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

	epsilons, convergence, display_convergence, evaluator = model._common_optimization_setup(
		epsilons, convergence, display_convergence, evaluator
	)


	if isinstance(scenarios, int):
		n_scenarios = scenarios
		scenarios = sample_uncertainties(model, n_scenarios)

	with evaluator:
		robust_results = evaluator.robust_optimize(
			robustness_functions,
			scenarios,
			nfe=nfe,
			constraints=constraints,
			epsilons=epsilons,
			convergence=convergence,
			convergence_freq=convergence_freq,
			**kwargs,
		)

	if isinstance(robust_results, tuple) and len(robust_results) == 2:
		robust_results, result_convergence = robust_results
	else:
		result_convergence = None

	robust_results = model.ensure_dtypes(robust_results)

	return OptimizationResult(
		robust_results,
		result_convergence,
		scope=model.scope,
		robustness_functions=robustness_functions,
	)