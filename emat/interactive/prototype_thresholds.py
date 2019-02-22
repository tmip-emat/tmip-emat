import matplotlib.pyplot as plt
import numpy, pandas
from IPython.display import clear_output
from ipywidgets import (
    VBox, HBox, Label, Layout, interactive_output, Dropdown, Output, Widget,
    Button, ToggleButtons, FloatRangeSlider, Accordion, Box, ToggleButton,
	widget_selection, IntRangeSlider
)
from .prototype_data import ExplorerData
from IPython.display import display_html, HTML, display
from ..scope.box import ChainedBox
from .prototype_pane import ExplorerPane
from .prototype_logging import logger

from ..scope.box import Bounds
from pandas import CategoricalDtype
from .multitoggle import MultiToggleButtons

slider_style = {
    'description_width': '300px',
    'min_width': '600px',
}

slider_layout = Layout(
            width='600px',
        )




class ExplorerThresholds(ExplorerPane):

	def __init__(self, ed:ExplorerData, box_universe, box_name, parent_widget=None):
		super().__init__(parent_widget=parent_widget)
		self.data = ed
		self._box_universe = box_universe
		self._box_name = box_name

		self._suspend_updates = False
		self.initialize_sliders()

		self.next_button.on_click(lambda x: self._parent_widget.load_reports())

	@property
	def joint_data(self):
		return self.data.joint_data

	def initialize_sliders(self, box_universe=None, box_name=None):

		if box_universe is not None:
			self._box_universe = box_universe
		if box_name is not None:
			self._box_name = box_name

		logger.info(f"initialize_sliders( {self._box_name} )")

		try:
			self.joint_filters = pandas.DataFrame(
				numpy.ones_like(self.joint_data, dtype=bool),
				index=self.joint_data.index,
				columns=self.joint_data.columns,
			)

			clusterdef = ChainedBox(self._box_universe, self._box_name)

			self.sliders = []
			self.outboxes = []

			for i in self.joint_data.columns:
				i_dtype = self.scope.get_dtype(i) or 'real'
				if i_dtype =='real':

					current_setting = clusterdef.get(i, (None,None))
					logger.info(f"   initial setting {i} = {current_setting}")

					current_min = self.joint_data[i].min() if current_setting[0] is None else current_setting[0]
					current_max = self.joint_data[i].max() if current_setting[1] is None else current_setting[1]

					controller = FloatRangeSlider(
						value=[current_min, current_max],
						min=self.joint_data[i].min(),
						max=self.joint_data[i].max(),
						step=(self.joint_data[i].max()-self.joint_data[i].min())/20,
						disabled=False,
						continuous_update=False,
						orientation='horizontal',
						readout=True,
						readout_format='.2f',
						description=i,
						style=slider_style,
						layout=slider_layout,
					)
				elif i_dtype == 'int':

					current_setting = clusterdef.get(i, (None,None))
					logger.info(f"   initial setting {i} = {current_setting}")

					current_min = self.joint_data[i].min() if current_setting[0] is None else current_setting[0]
					current_max = self.joint_data[i].max() if current_setting[1] is None else current_setting[1]

					controller = IntRangeSlider(
						value=[current_min, current_max],
						min=self.joint_data[i].min(),
						max=self.joint_data[i].max(),
						#step=(self.joint_data[i].max()-self.joint_data[i].min())/20,
						disabled=False,
						continuous_update=False,
						orientation='horizontal',
						readout=True,
						readout_format='d',
						description=i,
						style=slider_style,
						layout=slider_layout,
					)
				elif i_dtype == 'cat':
					cats = self.scope.get_cat_values(i)
					controller = MultiToggleButtons(
						description=i,
						style=slider_style,
						options=cats,
						disabled=False,
						button_style='',  # 'success', 'info', 'warning', 'danger' or ''
						layout=slider_layout,
					)
					controller.values = cats
				elif i_dtype == 'bool':
					cats = [False, True]
					controller = MultiToggleButtons(
						description=i,
						style=slider_style,
						options=cats,
						disabled=False,
						button_style='',  # 'success', 'info', 'warning', 'danger' or ''
						layout=slider_layout,
					)
					controller.values = cats

				else:  # buttons
					raise NotImplementedError(f"filters for {i}:{i_dtype}")
					controller = ToggleButtons(
						description=i,
						style=slider_style,
						options=['Off', 'On', 'Both'],
						value='Both',
						disabled=False,
						button_style='', # 'success', 'info', 'warning', 'danger' or ''
						tooltips=['Definitely off', 'Definitely on', 'Maybe'],
						layout=slider_layout,
					)
				self.sliders.append(controller)
				self.outboxes.append(
					Output(
						layout=Layout(
							height='1in',
							width='3in',
						),
					)
				)


			for s in range(len(self.sliders)):
				self.sliders[s].observe(self.replot_many)

			self.ui_risks = VBox(
				[
					HBox([s,ob])
					for s,ob in zip(self.sliders, self.outboxes)
					if s.description in self.data.all_risk_factors_
				],
			)

			self.ui_strategies = VBox(
				[
					HBox([s,ob])
					for s,ob in zip(self.sliders, self.outboxes)
					if s.description in self.data.all_strategy_names_
				],
			)

			self.ui_perform = VBox(
				[
					HBox([s,ob])
					for s,ob in zip(self.sliders, self.outboxes)
					if s.description in self.data.all_performance_measures_
				],
			)

			self.accordion = Accordion(children=[self.ui_strategies, self.ui_risks, self.ui_perform])
			self.accordion.set_title(0, 'Policy Levers')
			self.accordion.set_title(1, 'Exogenous Uncertainties' )
			self.accordion.set_title(2, 'Performance Measures')

			#self.footer = Output(layout={ 'border': '1px solid red', } )

			self.stack = VBox([
				self.header_area,
				self.accordion,
				self.footer,
			])

			self.set_header(clusterdef.names)
			self.replot_many(None)  # initial interactive plots

		except:
			logger.exception("error in initialize_sliders")
			raise

	def reset_sliders(self, box_universe=None, box_name=None):

		if box_universe is not None:
			self._box_universe = box_universe
		if box_name is not None:
			self._box_name = box_name

		logger.info(f"reset_sliders( {self._box_name} )")

		try:
			self._suspend_updates = True

			clusterdef = ChainedBox(self._box_universe, self._box_name)

			# Set All Sliders
			for i, controller in zip(self.joint_data.columns, self.sliders):
				i_dtype = self.scope.get_dtype(i) or 'real'
				if i_dtype in ('real', 'int'):

					current_setting = clusterdef.get(i, (None, None))
					logger.info(f"   current setting {i} = {current_setting}")

					current_min = self.joint_data[i].min() if current_setting[0] is None else current_setting[0]
					current_max = self.joint_data[i].max() if current_setting[1] is None else current_setting[1]

					try:
						controller.value = [current_min, current_max]
					except:
						logger.info(f"   controller.value = {controller.value}")
						logger.info(f"   controller.dir = {dir(controller)}")
						raise


					if current_min is not None and current_max is not None:
						self.joint_filters.loc[:, i] = (self.joint_data[i] >= current_min) & (self.joint_data[i] <= current_max)
					elif current_min is not None:
						self.joint_filters.loc[:, i] = (self.joint_data[i] >= current_min)
					elif current_max is not None:
						self.joint_filters.loc[:, i] = (self.joint_data[i] <= current_max)
					else:
						self.joint_filters.loc[:, i] = True
				elif i_dtype == 'cat':
					current_setting = clusterdef.get(i, self.scope.get_cat_values(i))
					controller.set_value(*current_setting)
					t = numpy.in1d(self.joint_data[i], list(current_setting))
					self.joint_filters.loc[:, i] = t
				elif i_dtype == 'bool':
					current_setting = clusterdef.get(i, [False, True])
					controller.set_value(*current_setting)
					t = numpy.in1d(self.joint_data[i], list(current_setting))
					self.joint_filters.loc[:, i] = t

				else: # set option pickers
					raise NotImplementedError(f"reset {i}:{i_dtype}")
					current_setting = clusterdef.get(i, 'Both')
					controller.value = current_setting

					if current_setting == "Off":
						self.joint_filters.loc[:, i] = (self.joint_data[i] == 0)
					elif current_setting == "On":
						self.joint_filters.loc[:, i] = (self.joint_data[i] != 0)
					if current_setting == "Both":
						self.joint_filters.loc[:, i] = True

			self.set_header(clusterdef.names)
			self._suspend_updates = False
			self.replot_many(None)  # initial interactive plots

		except:
			logger.exception("error in reset_sliders")
			raise
		finally:
			self._suspend_updates = False


	def set_header(self, names):
		logger.info(f"ExplorerThresholds.set_header( {names} )")
		self.header.clear_output()
		with self.header:
			for name in names[:-1]:
				display(HTML(f"<h4>▽ {name}</h4>"))
			display(HTML(f"<h1>▷ {names[-1]}</h1>"))

	def replot_single(self, i):
		col = self.joint_data.columns[i]
		self.outboxes[i].clear_output(wait=True)
		with self.outboxes[i]:
			plt.clf()
			fig = plt.gcf()
			fig.set_figwidth(3)
			fig.set_figheight(1)

			if self.scope.get_dtype(col) in ('cat', 'bool'):

				if self.scope.get_dtype(col) == 'cat':
					bar_labels = self.scope.get_cat_values(col)
				else:
					bar_labels = [False, True]

				v = self.joint_data[col].astype(
					CategoricalDtype(categories=bar_labels, ordered=False)
				).cat.codes
				bar_heights, _ = numpy.histogram(v, bins=numpy.arange(0, len(bar_labels) + 1))
				bar_x = numpy.arange(0, len(bar_labels))
				plt.bar(bar_x, bar_heights, 0.8, align='edge')
				filter_vals = self.joint_data.loc[self.joint_filters.all(axis=1), col].astype(
					CategoricalDtype(categories=bar_labels, ordered=False)
				).cat.codes
				bar_heights, _ = numpy.histogram(filter_vals, bins=numpy.arange(0, len(bar_labels) + 1))
				plt.bar(bar_x, bar_heights, 0.8, align='edge')
				plt.xticks(bar_x + 0.4, [str(i) for i in bar_labels])
				plt.show()

				# bar_labels, bar_heights = numpy.unique(self.joint_data[col], return_counts=True)
				# bar_x = numpy.arange(0,len(bar_labels))
				# plt.bar(bar_x, bar_heights, 0.8, align='edge')
				# from pandas import CategoricalDtype
				# filter_vals = self.joint_data.loc[self.joint_filters.all(axis=1), col].astype(
				# 	CategoricalDtype(categories=bar_labels, ordered=False)
				# ).cat.codes
				# bar_heights, _ = numpy.histogram(filter_vals, bins=numpy.arange(0,len(bar_labels)+1))
				# plt.bar(bar_x, bar_heights, 0.8, align='edge')
				# plt.xticks(bar_x+0.4, bar_labels)
				# plt.show()
			else:

				bins = 20 if col not in self.data.strategy_names else 20
				#n, bins, patches = plt.hist(self.joint_data[col], bins=bins)
				bar_heights, bar_x = numpy.histogram(self.joint_data[col], bins=bins)
				plt.bar(bar_x[:-1], bar_heights, bar_x[1:] - bar_x[:-1], align='edge')
				#n, bins, patches = plt.hist(self.joint_data.loc[self.joint_filters.all(axis=1), col], bins=bins)

				bar_heights, bar_x = numpy.histogram(self.joint_data.loc[self.joint_filters.all(axis=1), col], bins=bar_x)
				plt.bar(bar_x[:-1], bar_heights, bar_x[1:] - bar_x[:-1], align='edge')
				plt.show()


	def replot_many(self, content):
		logger.info(f"ExplorerThresholds.replot_many(  )")
		if content is not None and content['name'] == '_property_lock':
			return

		if self._suspend_updates:
			return

		try:
			owner = content['owner']
		except:
			owner = None

		if isinstance(owner, (FloatRangeSlider, IntRangeSlider)):
			aa = Bounds(*owner.value)
			i = owner.description
			self.joint_filters.loc[:, i] = (self.joint_data[i] >= aa[0]) & (self.joint_data[i] <= aa[1])

			self._box_universe[self._box_name].set_bounds(i, aa)
			# if i in self.data.performance_measure_names:
			# 	self._box_universe[self._box_name].measure_thresholds[i] = aa
			# elif i in self.data.risk_factor_names:
			# 	self._box_universe[self._box_name].uncertainty_thresholds[i] = aa
			logger.info(f"for {self._box_name}, changed slider {i} to {aa}")

		elif isinstance(owner, MultiToggleButtons):
			current_setting = owner.value
			i = owner.description
			t = numpy.in1d(self.joint_data[i], list(current_setting))
			self.joint_filters.loc[:, i] = t
			self._box_universe[self._box_name].replace_allowed_set(i, current_setting)

		elif isinstance(owner, ToggleButtons):
			raise NotImplementedError("togglebutttons")
			toggle_position = owner.value
			i = owner.description
			if toggle_position == "Off":
				self.joint_filters.loc[:, i] = (self.joint_data[i] == 0)
			elif toggle_position == "On":
				self.joint_filters.loc[:, i] = (self.joint_data[i] != 0)
			if toggle_position == "Both":
				self.joint_filters.loc[:, i] = True

			if i in self.data.strategy_names:
				self._box_universe[self._box_name].lever_thresholds[i] = toggle_position
			logger.info(f"for {self._box_name}, changed toggle {i} to {toggle_position}")

		logger.info(f"ExplorerThresholds.replot_many:: {len(self.sliders)} sliders")
		for i in range(len(self.sliders)):
			self.replot_single(i)

		self.footer.clear_output()
		with self.footer:
			print(self._box_name)
			print(ChainedBox(self._box_universe, self._box_name))

	def joint_filters_no_strat(self):
		return self.joint_filters.drop(self.data.strategy_names, axis=1).all(axis=1)

	def strategy_filters(self):
		return self.joint_filters.loc[:, self.data.strategy_names]

	def joint_strategy_filters(self):
		return self.joint_filters.loc[:, self.data.strategy_names].all(axis=1)