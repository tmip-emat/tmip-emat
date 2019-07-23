
import pandas
import operator

import ema_workbench.analysis.prim
from ..scope.box import Box, Bounds, Boxes
from ..scope.scope import Scope
from .discovery import ScenarioDiscoveryMixin

class Prim(ema_workbench.analysis.prim.Prim, ScenarioDiscoveryMixin):

	def find_box(self):
		result = super().find_box()
		result.__class__ = PrimBox
		return result


class PrimBox(ema_workbench.analysis.prim.PrimBox):

	def to_emat_box(self, i=None, name=None):
		if i is None:
			i = self._cur_box

		if name is None:
			name = f'prim box {i}'

		limits = self.box_lims[i]

		b = Box(name)

		for col in limits.columns:
			if isinstance(self.prim.x.dtypes[col], pandas.CategoricalDtype):
				if set(self.prim.x[col].cat.categories) != limits[col].iloc[0]:
					b.replace_allowed_set(col, limits[col].iloc[0])
			else:
				if limits[col].iloc[0] != self.prim.x[col].min():
					b.set_lower_bound(col, limits[col].iloc[0])
				if limits[col].iloc[1] != self.prim.x[col].max():
					b.set_upper_bound(col, limits[col].iloc[1])
		b.coverage = self.coverage
		b.density = self.density
		b.mass = self.mass
		return b

	def __repr__(self):
		i = self._cur_box
		head = f"<{self.__class__.__name__} peel {i+1} of {len(self.peeling_trajectory)}>"

		# make the box definition
		qp_values = self.qp[i]
		uncs = [(key, value) for key, value in qp_values.items()]
		uncs.sort(key=operator.itemgetter(1))
		uncs = [uncs[0] for uncs in uncs]
		box_lim = pandas.DataFrame( index=uncs, columns=['min','max'])
		for unc in uncs:
			values = self.box_lims[i][unc]
			box_lim.loc[unc] = [values[0], values[1]]
		head += f'\n   coverage: {self.coverage:.5f}'
		head += f'\n   density:  {self.density:.5f}'
		head += f'\n   mean: {self.mean:.5f}'
		head += f'\n   mass: {self.mass:.5f}'
		head += f'\n   restricted dims: {self.res_dim}'
		if not box_lim.empty:
			head += "\n     "+str(box_lim).replace("\n", "\n     ")
		return head