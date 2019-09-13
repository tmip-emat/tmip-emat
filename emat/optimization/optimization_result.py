
import pandas
from ..viz.parcoords import ParCoordsViewer
from .nondominated import nondominated_solutions

class OptimizationResult:

	def __init__(self, result, convergence, scope=None, robustness_functions=None, scenarios=None):
		self.result = result
		self.convergence = convergence
		self.scope = scope
		self.robustness_functions = robustness_functions
		self.scenarios = scenarios

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

		self.result = nondominated_solutions(
			pandas.concat([self.result, alternate_solutions]).reset_index(drop=True),
			self.scope,
			self.robustness_functions,
		)

		return self


	def check_extremes(self, model, n=1, evaluator=None, cache_dir=None):
		from ..scope.parameter import CategoricalParameter
		for _ in range(n):
			for lever_name in model.scope.get_lever_names():
				df = self.result.copy()
				if isinstance(model.scope[lever_name], CategoricalParameter):
					extremes = (model.scope[lever_name].values)
				else:
					extremes = (model.scope[lever_name].min, model.scope[lever_name].max)
				for x in extremes:
					df.loc[:, lever_name] = x
					self.add_solutions(model.robust_evaluate(
						self.robustness_functions,
						scenarios=self.scenarios,
						policies=df,
						evaluator=evaluator,
						cache_dir=cache_dir,
					))
		return self