import matplotlib.pyplot as plt
import numpy, pandas
from IPython.display import clear_output
from ipywidgets import (
    VBox, HBox, Label, Layout, interactive_output, Dropdown, Output,
    Button, ToggleButtons, FloatRangeSlider, Accordion, Checkbox, Select, Text
)
from ipywidgets import HTML as HTML_widget

from .prototype_data import ExplorerData
from IPython.display import display_html, HTML, display
from ..scope.box import ChainedBox
from .prototype_pane import ExplorerPane
from .prototype_logging import logger, execution_tracer
from .snippets import *

slider_style = {
    'description_width': '300px',
    'min_width': '600px',
}

slider_layout = Layout(
            width='600px',
        )

_demanded_tooltip = "Cannot disable features with active thresholds"



#
# labels = []
# sliders = []
# outboxes = []

class ExplorerFeatureSelect(ExplorerPane):

	def __init__(self, ed:ExplorerData, box_universe, box_name, parent_widget=None):
		super().__init__(parent_widget=parent_widget)
		self.data = ed
		self._box_universe = box_universe
		self._box_name = box_name

		self._suspend_updates = False

		try:

			clusterdef = ChainedBox(self._box_universe, self._box_name)

			clusterdef_relevant = clusterdef.relevant_and_demanded_features
			clusterdef_demanded = clusterdef.demanded_features

			self.checkers = []

			for i in self.data.all_performance_measures:
				controller = Checkbox(
					value=i in clusterdef_relevant,
					description=i,
					disabled=i in clusterdef_demanded,
					style = {'description_width': 'initial'},
					tooltip=i if i not in clusterdef_demanded else _demanded_tooltip,
				)
				self.checkers.append(controller)

			for i in self.data.all_strategy_names:
				controller = Checkbox(
					value=i in clusterdef_relevant,
					description=i,
					disabled=i in clusterdef_demanded,
					style = {'description_width': 'initial'},
					tooltip=i if i not in clusterdef_demanded else _demanded_tooltip,
				)
				self.checkers.append(controller)

			for i in self.data.all_risk_factors:
				controller = Checkbox(
					value=i in clusterdef_relevant,
					description=i,
					disabled=i in clusterdef_demanded,
					style = {'description_width': 'initial'},
					tooltip=i if i not in clusterdef_demanded else _demanded_tooltip,
				)
				self.checkers.append(controller)


			for s in range(len(self.checkers)):
				self.checkers[s].observe(self.toggle_check)


			self.ui_risks = VBox(
				[ch for ch in self.checkers if ch.description in self.data.all_risk_factors ],
			)

			self.ui_strategies = VBox(
				[ch for ch in self.checkers if ch.description in self.data.all_strategy_names ],
			)

			self.ui_perform = VBox(
				[ch for ch in self.checkers if ch.description in self.data.all_performance_measures ],
			)

			self.accordion = Accordion(
				children=[self.ui_strategies, self.ui_risks, self.ui_perform],
				layout = Layout(
					width='600px',
				)
			)
			self.accordion.set_title(0, 'Policy Levers')
			self.accordion.set_title(1, 'Exogenous Uncertainties' )
			self.accordion.set_title(2, 'Performance Measures')

			self.accordion_filter = Text(
				value='.*',
				placeholder='.*',
				description='Filter:',
				disabled=False
			)
			self.accordion_filter.observe(self.filter_checkboxes)


			self.current_relevant_attributes = Select(
				options=[],
				# value='OSX',
				rows=30,
				description='',
				disabled=False
			)

			# self.load_current_relevant_attributes_button = Button(
			# 	description=f"Next",
			# 	disabled=False,
			# 	button_style='',  # 'success', 'info', 'warning', 'danger' or ''
			# 	tooltip='Load',
			# 	# icon='check'
			# )
			# self.load_current_relevant_attributes_button.on_click(
			# 	self.load_selected_features_into_thresholds
			# )

			self.next_button.on_click(
				self.load_selected_features_into_thresholds
			)

			# self.stack = VBox([
			# 	self.header_area,
			# 	HBox([
			# 		VBox([
			# 			self.accordion_filter,
			# 			self.accordion,
			# 		]),
			# 		VBox([
			# 			HTML_widget(value=f"<b>{RELEVANT}</b>", placeholder=f"These things will be loaded"),
			# 			self.current_relevant_attributes,
			# 			self.load_current_relevant_attributes_button,
			# 		]),
			# 	]),
			# 	self.footer,
			# ])

			self.make_stack(
				HBox([
					VBox([
						self.accordion_filter,
						self.accordion,
					]),
					VBox([
						HTML_widget(value=f"<b>{RELEVANT}</b>", placeholder=f"These things will be loaded"),
						self.current_relevant_attributes,
						# self.load_current_relevant_attributes_button,
					]),
				]),
			)

			self.set_header(clusterdef.names)
			self.recompile_list_of_active_features(None)  # initial interactive plots

		except:
			logger.exception("error in initialize_sliders")
			raise

	@property
	def joint_data(self):
		return self.data.joint_data

	def set_header(self, names):
		logger.info(f"{self.__class__.__name__}.set_header( {names} )")
		self.header.clear_output()
		with self.header:
			for name in names[:-1]:
				display(HTML(f"<h4>▽ {name}</h4>"))
			display(HTML(f"<h1>▷ {names[-1]}</h1>"))

	def toggle_check(self, action_info):
		try:
			logger.debug(f"toggle_check: {action_info}")
			owner = action_info.get('owner')
			if owner is None:
				return
			label = owner.description
			if action_info.get('name') == 'value':
				if label is not None:
					value = action_info.get('new')
					if value is not None:
						logger.info(f"toggle_check: set {label} to {value}")
						if value:
							self._box_universe[self._box_name].relevant_features.add(label)
							self.recompile_list_of_active_features(None)
						else:
							self._box_universe[self._box_name].relevant_features.discard(label)
							self.recompile_list_of_active_features(None)
		except:
			logger.exception("error in ExplorerFeatureSelect.toggle_check")
			raise

	def recompile_list_of_active_features(self, content):
		if content is not None and content['name'] == '_property_lock':
			return

		if self._suspend_updates:
			return

		try:
			owner = content['owner']
		except:
			owner = None


		on_risks = []
		on_strategies = []
		on_perform = []

		for ch in self.checkers:
			if ch.value:
				if ch.description in self.data.all_risk_factors_:
					on_risks.append(ch.description)
				if ch.description in self.data.all_strategy_names_:
					on_strategies.append(ch.description)
				if ch.description in self.data.all_performance_measures_:
					on_perform.append(ch.description)


		current = []

		if on_strategies:
			current.append("-- STRATEGIES --")
			current.extend(on_strategies)

		if on_risks:
			current.append("-- RISK FACTORS --")
			current.extend(on_risks)

		if on_perform:
			current.append("-- PERFORMANCE MEASURES --")
			current.extend(on_perform)

		self.current_relevant_attributes.options = current



	def filter_checkboxes(self, event):
		if event.get('name') == 'value':
			logger.debug(f"filter_checkboxes:{event}")
			filter = event.get('new')
			if filter is not None:
				if filter == "" or filter=="*":
					for ch in self.checkers:
						# ch.layout.visibility = 'visible'
						ch.layout.display = 'flex'
				else:
					import re
					pattern = re.compile(filter)
					for ch in self.checkers:
						if pattern.search(ch.description):
							#ch.layout.visibility = 'visible'
							ch.layout.display = 'flex'
						else:
							#ch.layout.visibility = 'hidden'
							ch.layout.display = 'none'


	def load_selected_features_into_thresholds(self, event):
		execution_tracer(logger, event)
		try:
			self._parent_widget.load_cluster(self._box_name)
			[h.flush() for h in logger.handlers[0]]
		except:
			logger.exception()
			raise
		execution_tracer(logger, "END")
