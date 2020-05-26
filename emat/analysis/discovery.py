
import pandas

import emat.workbench.analysis.prim
import abc
from ..scope.box import Box, Bounds, Boxes
from ..scope.scope import Scope

class ScenarioDiscoveryMixin():

	@abc.abstractmethod
	def boxes_to_dataframe(self, include_stats=False):
		pass

	@abc.abstractmethod
	def stats_to_dataframe(self):
		pass

	def to_emat_boxes(self, scope=None):
		"""
		Export boxes to emat.Boxes

		Args:
			scope (Scope): The scope

		Returns:
			Boxes
		"""
		boxes = Boxes(scope=scope)
		raw_boxes = self.boxes_to_dataframe()
		stats = self.stats_to_dataframe()
		n = 1
		while f'box {n}' in raw_boxes.columns:
			box_df = raw_boxes[f'box {n}']
			b = Box(f'box {n}')
			for row_n, row in box_df.iterrows():
				if isinstance(self.x.dtypes[row.name], pandas.CategoricalDtype):
					if set(self.x[row.name].cat.categories) != row['min']:
						b.replace_allowed_set(row.name, row['min'])
				else:
					if row['min'] != self.x[row.name].min():
						b.set_lower_bound(row.name, row['min'])
					if row['max'] != self.x[row.name].max():
						b.set_upper_bound(row.name, row['max'])
			b.coverage = stats.loc[f'box {n}', 'coverage']
			b.density = stats.loc[f'box {n}', 'density']
			b.mass = stats.loc[f'box {n}', 'mass']
			boxes.add(b)
			n += 1
		return boxes


