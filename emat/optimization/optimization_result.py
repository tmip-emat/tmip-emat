
from ..viz.parcoords import ParCoordsViewer

class OptimizationResult:

	def __init__(self, result, convergence, scope=None):
		self.result = result
		self.convergence = convergence
		self.scope = scope

	def par_coords(self):
		return ParCoordsViewer(
			self.result,
			self.scope,
		)

