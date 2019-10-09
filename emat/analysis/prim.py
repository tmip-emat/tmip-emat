
import numpy
import pandas
import operator

import ema_workbench.analysis.prim
from ..scope.box import Box, Bounds, Boxes
from ..scope.scope import Scope
from .discovery import ScenarioDiscoveryMixin

from plotly import graph_objects as go

class Prim(ema_workbench.analysis.prim.Prim, ScenarioDiscoveryMixin):

	def find_box(self):
		result = super().find_box()
		result.__class__ = PrimBox
		result._explorer = getattr(self, '_explorer', None)
		return result

	def tradeoff_selector(self, n=-1, colorscale='viridis'):
		'''
		Visualize the trade off between coverage and density, for
		a particular PrimBox.

		Parameters
		----------
		n : int, optional
			The index number of the PrimBox to use.  If not given,
			the last found box is used.  If no boxes have been found
			yet, giving any value other than -1 will raise an error.
		colorscale : str, default 'viridis'
			A valid color scale name, as compatible with the
			color_palette method in seaborn.

		Returns
		-------
		FigureWidget
		'''
		try:
			box = self._boxes[n]
		except IndexError:
			if n == -1:
				box = self.find_box()
			else:
				raise
		return box.tradeoff_selector(colorscale=colorscale)

def _discrete_color_scale(name='viridis', n=8):
	import seaborn as sns
	colors = sns.color_palette(name, n)
	colorlist = []
	for i in range(n):
		c = colors[i]
		thiscolor_s = f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
		colorlist.append([i/n, thiscolor_s])
		colorlist.append([(i+1)/n, thiscolor_s])
	return colorlist


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
		b.coverage = self.peeling_trajectory['coverage'][i]
		b.density = self.peeling_trajectory['density'][i]
		b.mass = self.peeling_trajectory['mass'][i]
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

	def _make_tradeoff_selector(self, colorscale='cividis'):
		'''
		Visualize the trade off between coverage and density. Color
		is used to denote the number of restricted dimensions.

		Parameters
		----------
		colorscale : str
			valid seaborn color scale name

		Returns
		-------
		a FigureWidget instance

		'''

		peeling_trajectory = self.peeling_trajectory

		hovertext = pandas.Series('', index=peeling_trajectory.index)

		fig = go.FigureWidget()

		for i in range(len(peeling_trajectory)):
			t = str(self.to_emat_box(i, name=str(i))).replace("\n","<br>")
			hovertext.iloc[i] = f'<span style="font-family:Consolas,monospace">{t}</span>'

		n_colors = max(peeling_trajectory['res_dim'])+1
		color_scale_ = _discrete_color_scale(colorscale, n_colors)
		colortickvals = numpy.arange(0.5, n_colors, 1) * (n_colors-1)/n_colors
		colorticktext = [str(i) for i in range(n_colors)]

		scatter = fig.add_scatter(
			x=peeling_trajectory['coverage'],
			y=peeling_trajectory['density'],
			mode='markers',
			marker=dict(
				color=peeling_trajectory['res_dim'],
				colorscale=color_scale_,
				showscale=True,
				colorbar=dict(
					title="Number of Restricted Dimensions",
					titleside="right",
					tickmode="array",
					tickvals=colortickvals,
					ticktext=colorticktext,
					ticks="outside",
				),
			),
			text=hovertext,
			hoverinfo="text",
		).data[-1]

		fig.update_layout(
			margin=dict(l=10, r=10, t=10, b=10),
			width=600,
			height=400,
			xaxis_title_text='Coverage',
			yaxis_title_text='Density',
		)

		# create callback function
		def select_point(trace, points, selector):
			for i in points.point_inds:
				self.select(i)
				explorer = getattr(self, '_explorer', None)
				if explorer is not None:
					explorer.set_box(self.to_emat_box())

		scatter.on_click(select_point)

		return fig

	def tradeoff_selector(self, colorscale='viridis'):
		'''
		Visualize the trade off between coverage and density. Color
		is used to denote the number of restricted dimensions.

		Parameters
		----------
		colorscale : str, default 'viridis'
			A valid color scale name, as compatible with the
			color_palette method in seaborn.

		Returns
		-------
		FigureWidget
		'''
		if getattr(self, '_tradeoff_widget', None) is None:
			self._tradeoff_widget = self._make_tradeoff_selector(colorscale=colorscale)
		return self._tradeoff_widget

	def explore(self, scope=None, data=None):
		if getattr(self, '_explorer', None) is None:
			from .explore import Explore
			self._explorer = Explore(scope=scope, data=data, box=self.to_emat_box())
		return self._explorer

