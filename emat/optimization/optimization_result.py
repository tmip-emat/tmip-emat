
from ..viz.parcoords import ParCoordsViewer

class OptimizationResult:

	def __init__(self, result, convergence, scope=None, robustness_functions=None):
		self.result = result
		self.convergence = convergence
		self.scope = scope
		self.robustness_functions = robustness_functions

	def par_coords(self):
		return ParCoordsViewer(
			self.result,
			scope=self.scope,
			robustness_functions=self.robustness_functions,
		)

