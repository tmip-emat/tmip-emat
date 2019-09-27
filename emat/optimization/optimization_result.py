
import pandas
from ..viz.parcoords import ParCoordsViewer
from .nondominated import nondominated_solutions

from ema_workbench import Scenario, Policy

from ..util.loggers import get_module_logger
_logger = get_module_logger(__name__)

class OptimizationResult:

	def __init__(self, result, convergence, scope=None, robustness_functions=None, scenarios=None, policies=None):
		self.result = result
		self.convergence = convergence
		self.scope = scope
		self.robustness_functions = robustness_functions
		self.scenarios = scenarios
		self.policies = policies

	@property
	def scenario(self):
		"""Access to a single scenario"""
		if isinstance(self.scenarios, Scenario) or self.scenarios is None:
			return self.scenarios
		raise TypeError("scenario is invalid")

	@property
	def policy(self):
		"""Access to a single policy"""
		if isinstance(self.policies, Policy) or self.policies is None:
			return self.policies
		raise TypeError("policy is invalid")

	def par_coords(self):
		return ParCoordsViewer(
			self.result,
			scope=self.scope,
			robustness_functions=self.robustness_functions,
		)

	def add_solutions(self, alternate_solutions):
		"""
		Incorporate non-dominated solutions from an alternate set of solutions.

		Parameters
		----------
		alternate_solutions : DataFrame or OptimizationResult
		"""
		if isinstance(alternate_solutions, OptimizationResult):
			alternate_solutions = alternate_solutions.result

		_prev_count = len(self.result)

		self.result = nondominated_solutions(
			pandas.concat([self.result, alternate_solutions[self.result.columns]]).reset_index(drop=True),
			self.scope,
			self.robustness_functions,
		)
		net_gain = len(self.result)-_prev_count
		if net_gain >= 0:
			_logger.info(f"add_solutions: net gain of {net_gain} solutions")
		else:
			_logger.info(f"add_solutions: net loss of {-net_gain} solutions")
		return self


	def check_extremes(self, model, n=1, evaluator=None, cache_dir=None, searchover='levers', robust=True):
		from ..scope.parameter import CategoricalParameter
		for i in range(n):
			if searchover == 'levers':
				these_names = model.scope.get_lever_names()
			elif searchover == 'uncertainties':
				these_names = model.scope.get_uncertainty_names()
			else:
				raise ValueError(f"search_over must be levers or uncertainties, not {searchover}")

			_logger.debug(f"checking extremes of {searchover}, pass {i+1} of {n}")
			for lever_name in these_names:
				df = self.result.copy()
				_logger.debug(f"checking extreme {searchover} for {lever_name} on {len(df)} candidate solutions")
				if isinstance(model.scope[lever_name], CategoricalParameter):
					extremes = (model.scope[lever_name].values)
				else:
					extremes = (model.scope[lever_name].min, model.scope[lever_name].max)
				for x in extremes:
					df.loc[:, lever_name] = x
					if robust:
						self.add_solutions(model.robust_evaluate(
							self.robustness_functions,
							scenarios=self.scenarios,
							policies=df,
							evaluator=evaluator,
							cache_dir=cache_dir,
						))
					elif searchover == 'levers':
						if self.scenarios is not None:
							for k in self.scenarios:
								df[k] = self.scenarios[k]
						self.add_solutions(model.run_experiments(
							design=df,
							evaluator=evaluator,
							db=False,
						))
					elif searchover == 'uncertainties':
						if self.policies is not None:
							for k in self.policies:
								df[k] = self.policies[k]
						self.add_solutions(model.run_experiments(
							design=df,
							evaluator=evaluator,
							db=False,
						))
		return self