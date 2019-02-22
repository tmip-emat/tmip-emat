
from ema_workbench.em_framework.optimization import AbstractConvergenceMetric, Hypervolume, to_dataframe
import pandas
import plotly_nb
from ipywidgets import widgets
from IPython.display import display_html


class EpsilonProgress(AbstractConvergenceMetric):
	'''
	Epsilon progress convergence metric class.

	This convergence metric counts the number of new solutions
	that enter the solution set.
	'''

	def __init__(self):
		super(EpsilonProgress, self).__init__("epsilon_progress")
		self.figure = plotly_nb.line_graph(
			X=None, Y=None, widget=True, title="ε-Progress", xtitle='Generation', interpolate='linear',
		)

	def __call__(self, optimizer):
		self.results.append(optimizer.algorithm.archive.improvements)
		self.figure.data[0].y = self.results

class HyperVolume(AbstractConvergenceMetric):
	"""
	HyperVolume convergence metric class

	This metric is derived from a hyper-volume measure, which describes the
	multi-dimensional volume of space contained within the pareto front. When
	computed with minimum and maximums, it describes the ratio of dominated
	outcomes to all possible outcomes in the extent of the space.  Getting this
	number to be high or low is not necessarily important, as not all outcomes
	within the min-max range will be feasible.  But, having the hypervolume remain
	fairly stable over multiple generations of the evolutionary algorithm provides
	an indicator of convergence.

	Args:
		minimum (array-like):
			The expected minimum values for each dimension of the outcome space.
		maximum (array-like):
			The expected maximum values for each dimension of the outcome space.

	"""

	def __init__(self, minimum, maximum):
		super(HyperVolume, self).__init__("hypervolume")
		self.hypervolume_func = Hypervolume(minimum=minimum, maximum=maximum)
		self.figure = plotly_nb.line_graph(
			X=None, Y=None, widget=True, title = "Hypervolume", xtitle='Generation', interpolate='linear',
		)

	def __call__(self, optimizer):
		self.results.append(self.hypervolume_func.calculate(optimizer.algorithm.archive))
		start = 0
		while start < len(self.results) and self.results[start] == 0:
			start += 1
		figure_data = self.results[start:]
		self.figure.data[0].y = figure_data
		try:
			y_range = min(figure_data), max(figure_data)
		except:
			pass
		else:
			buffer = (y_range[1]-y_range[0])/10
			if buffer:
				y_range = y_range[0]-buffer, y_range[1]+buffer
			if len(figure_data) > 1:
				self.figure.layout.yaxis.range = y_range

	@classmethod
	def from_outcomes(cls, outcomes):
		ranges = [o.expected_range for o in outcomes if o.kind != o.INFO]
		lows = [_[0] for _ in ranges]
		highs = [_[1] for _ in ranges]
		return cls(lows, highs)


class SolutionViewer(AbstractConvergenceMetric):
	"""
	SolutionViewer convergence metric class

	This convergence metric isn't actually a metric,
	but instead just provides a dynamically updated view
	of the current set of solutions.

	Parameters
	----------
	decision_varnames (Collection[str]):
		The names of the decision variables.
	outcome_varnames (Collection[str]):
		The names of the outcome variables.
	dataframe_processor (callable, optional):
		A post-processing function that will be called
		on the resulting DataFrame of solutions, and
		should also return a DataFrame.
	"""

	def __init__(self, decision_varnames, outcome_varnames, dataframe_processor=None):
		super().__init__('archive_viewer')

		self.figure = plotly_nb.table_figure(
			pandas.DataFrame(
				data=None,
				index=[],
				columns=decision_varnames+outcome_varnames
			),
			title="Current Solutions",
		)
		self.decision_varnames = decision_varnames
		self.outcome_varnames = outcome_varnames
		self.dataframe_processor = dataframe_processor
		self.index = 0

	def __call__(self, optimizer):
		self.index += 1

		archive = to_dataframe(optimizer, self.decision_varnames, self.outcome_varnames)
		if self.dataframe_processor is not None:
			archive = self.dataframe_processor(archive)

		self.figure.data[0].cells.values = [archive[c] for c in archive.columns]

	@classmethod
	def from_model_and_outcomes(cls, model, outcomes):

		levers = model.scope.get_lever_names()

		try:
			dataframe_processor = model.ensure_dtypes
		except AttributeError:
			dataframe_processor = None

		return cls(
			levers,
			[i.name for i in outcomes if i.kind != 0],
			dataframe_processor=dataframe_processor,
		)


class ConvergenceMetrics(widgets.HBox):
	"""
	ConvergenceMetrics emulates a list to contain AbstractConvergenceMetrics.

	It also offers a widgets view.
	"""

	def __init__(self, *members):
		self._members = list(members)
		widgets.HBox.__init__(self, [member.figure for member in members])

	def __getitem__(self, item):
		return self._members[item]

	def __setitem__(self, key, value):
		self._members[key] = value
		self.children[key] = value.figure

	def __delitem__(self, key):
		del self._members[key]
		del self.children[key]

	def __len__(self):
		return len(self._members)

	def insert(self, position, value):
		self._members.insert(position, value)
		self.children.insert(position, value.figure)

