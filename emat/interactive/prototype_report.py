
import numpy, pandas
import matplotlib.pyplot as plt
import seaborn
from IPython.display import display_html, HTML, display
from ipywidgets import (
    VBox, HBox, Label, Layout, interactive_output, Dropdown, Output,
    Button, ToggleButtons, FloatRangeSlider, Accordion, IntRangeSlider,
)

from .multitoggle import MultiToggleButtons
from .prototype_thresholds import ExplorerThresholds
from .prototype_pane import ExplorerPane

from ..scope.box import Bounds

def summarize_cluster(es:ExplorerThresholds):

	comments = []

	for owner in es.sliders:

		if isinstance(owner, (FloatRangeSlider, IntRangeSlider)):
			aa = Bounds(*owner.value)
			i = owner.description
			max_binding = (owner.max != aa.upperbound)
			min_binding = (owner.min != aa.lowerbound)
			if max_binding and min_binding:
				comments.append(f"{i} is between {aa.lowerbound:.2f} and {aa.upperbound:.2f}")
			elif max_binding:
				comments.append(f"{i} is below {aa.upperbound:.2f}")
			elif min_binding:
				comments.append(f"{i} is above {aa.lowerbound:.2f}")
		elif isinstance(owner, MultiToggleButtons):
			toggle_position = owner.value
			i = owner.description
			if es.scope.get_dtype(i) == 'bool':
				possible_values = [False, True]
			else:
				possible_values = es.scope.get_cat_values(i)
			set_toggle_position = set(toggle_position)
			if set_toggle_position != set(possible_values):
				if len(set_toggle_position) == 1:
					comments.append(f"{i} is {set_toggle_position.pop()}")
				else:
					comments.append(f"{i} is in {set_toggle_position}")

	return comments



## Conditional on the cluster, what is the breakdown of strategies?

from .util import dicta, Counter

def get_strategizer_view(es:ExplorerThresholds):

	c = Counter()

	for t in es.joint_data.loc[es.joint_filters_no_strat(), es.data.strategy_names].itertuples():
		c.one(tuple(t)[1:])

	strategizer = pandas.DataFrame(
		columns=es.data.strategy_names + ['Count', 'Conditional Possibility']
	)


	for key, val in c.items():
		strategizer.loc[str(key), 'Count'] = val
		for sn, s in enumerate(es.data.strategy_names):
			strategizer.loc[str(key), s] = key[sn]



	strategizer.sort_values('Count', ascending=False, inplace=True)

	strategizer.index = numpy.arange(len(strategizer))
	strategizer.loc[:,'Conditional Possibility'] = strategizer.loc[:,'Count'] / strategizer.loc[:,'Count'].sum()

	strategizer_view = strategizer.drop('Count', axis=1).style.bar(
		subset=['Conditional Possibility'], color='#d65f5f', align='mid'
	).format(
		{'Conditional Possibility': "{:0>.2%}", }
	).set_properties(**{'text-align': 'left', 'color':'white'}, subset=['Conditional Possibility'])

	return strategizer_view


def draw_boxplot(es:ExplorerThresholds, measure):
	plt.figure(figsize=(8, 1.5))
	_ = plt.boxplot(
		[
			es.joint_data.loc[es.joint_strategy_filters(), measure].values,
			es.joint_data.loc[es.joint_filters.all(axis=1), measure].values,
		],
		vert=False,
		notch=True,
		labels=[
			'All Scenarios, Available Policy Levers',
			'Current Box',
		],
		widths=0.75,
	)
	plt.title(measure)
	plt.show()


def draw_boxplots(es:ExplorerThresholds, measures):
	for m in measures:
		draw_boxplot(es, m)


def draw_renormalized_risks(es:ExplorerThresholds):

	for r in es.data.X.columns:
		if r in es.data.all_risk_factors_:
			plt.figure(figsize=(8, 1.5))
			seaborn.kdeplot(es.joint_data.loc[es.joint_strategy_filters(), r].values, label='All Scenarios, Available Policy Levers')
			seaborn.kdeplot(es.joint_data.loc[es.joint_filters.all(axis=1), r].values, label='Current Box')
			plt.title(r)
			plt.show()


def single_report(es:ExplorerThresholds, output_to=None):

	if output_to is None:
		from .util import noop_wrapper
		output_to = noop_wrapper()

	with output_to:

		display(HTML("<h1>Exploratory Modeling Report</h1>"))

		percent_of_space = es.joint_filters.all(axis=1).sum() / es.joint_strategy_filters().sum()
		summ = "<ul>" + "\n".join([f"<li>{t}</li>" for t in summarize_cluster(es)]) + "</ul>"
		display(HTML(summ))
		display(HTML(f"<i>The current box represents {percent_of_space:.1%} of the EMA exploration space.</i>"))

		display(HTML("<h3>Performance Measures</h3>"))
		draw_boxplots(es, es.data.Y.columns)

		display(HTML("<h3>Exogenous Uncertainties</h3>"))
		draw_renormalized_risks(es)

		display(HTML("<h3>Exploratory Modeling Strategizer</h3>"))
		display(HTML(f"<i>The current box represents {percent_of_space:.1%} of the EMA exploration space.</i>"))
		display(get_strategizer_view(es))


#### Build/No Build Reports

def draw_boxplot_bnb(es:ExplorerThresholds, measure, bnb):
	plt.figure(figsize=(8, 1.5))
	_ = plt.boxplot(
		[
			es.joint_data.loc[es.joint_strategy_filters(), measure].values,
			es.joint_data.loc[es.joint_filters.all(axis=1) & (es.joint_data[bnb] == 1), measure].values,
			es.joint_data.loc[es.joint_filters.all(axis=1) & (es.joint_data[bnb] == 0), measure].values,
		],
		vert=False,
		notch=True,
		labels=[
			'All Scenarios, Available Policy Levers',
			'Current Box, Build',
			'Current Box, No Build',
		],
		widths=0.75,
	)
	plt.title(measure)
	plt.show()


def draw_boxplots_bnb(es:ExplorerThresholds, measures, bnb):
	for m in measures:
		draw_boxplot_bnb(es, m, bnb)


def draw_renormalized_risks_bnb(es:ExplorerThresholds, bnb):
	for r in es.data.X.columns:
		if r in es.data.all_risk_factors_:
			plt.figure(figsize=(8, 1.5))
			seaborn.kdeplot(es.joint_data.loc[es.joint_strategy_filters(), r].values, label='All Scenarios, Available Policy Levers')
			seaborn.kdeplot(es.joint_data.loc[es.joint_filters.all(axis=1) & (es.joint_data[bnb] == 1), r].values,
							label='Current Box, Build')
			seaborn.kdeplot(es.joint_data.loc[es.joint_filters.all(axis=1) & (es.joint_data[bnb] == 0), r].values,
							label='Current Box, No Build')
			plt.title(r)
			plt.show()


def build_no_build_report(es:ExplorerThresholds, bnb, output_to=None, ):
	if output_to is None:
		from .util import noop_wrapper
		output_to = noop_wrapper()

	with output_to:

		display(HTML("<h1>Exploratory Modeling Build / No Build Report</h1>"))

		percent_of_space = es.joint_filters.all(axis=1).sum() / es.joint_strategy_filters().sum()
		summ = "<ul>" + "\n".join([f"<li>{t}</li>" for t in summarize_cluster(es)]) + "</ul>"
		display(HTML(summ))
		display(HTML(f"<i>The current box represents {percent_of_space:.1%} of the EMA exploration space.</i>"))

		display(HTML("<h3>Performance Measures</h3>"))
		draw_boxplots_bnb(es, es.data.Y.columns, bnb)

		display(HTML("<h3>Exogenous Uncertainties</h3>"))
		draw_renormalized_risks_bnb(es, bnb)



####



_report_picker_current_selection = 'Current Box Report'
_report_picker_build_nobuild = 'Build / No Build Report'

class ExplorerReports(ExplorerPane):

	def __init__(self, es:ExplorerThresholds, parent_widget=None):
		super().__init__(parent_widget=parent_widget)
		self.selector = es
		self.report_picker = Dropdown(
			options=[
				' ',
				_report_picker_current_selection,
				_report_picker_build_nobuild,
				'Other Report'],
			value=' ',
			description='Choose:',
			disabled=False,
		)
		self.report_picker.observe(self.report_picker_action)

		self.second_picker = Dropdown(
			options=(" ",),
			value=' ',
			description=" ",
			disabled=False,
		)
		self.second_picker.layout.visibility = 'hidden'

		self.report_pane = Output(
			layout=Layout(
				width='7.5in',
			),
		)
		self.make_stack(
			self.report_picker,
			self.second_picker,
			self.report_pane,
		)

		self.next_button.disabled = True

	def redraw_with_no_second_picker(self):
		self.second_picker.layout.visibility = 'hidden'

	def redraw_with_bnb_picker(self, second_options=(' ',), label="Choose:"):

		# self.second_picker = Dropdown(
		# 	options=second_options,
		# 	description=label,
		# 	disabled=True,
		# )
		#
		self.second_picker.layout.visibility = 'visible'

		self.second_picker.options = (' ',) + tuple(second_options)

		self.second_picker.description = "Strategy:"

		#
		# self.stack = VBox([
		# 	self.report_picker,
		# 	self.second_picker,
		# 	self.report_pane,
		# ])

		self.second_picker.observe(self.bnb_picker_action)



	def report_picker_action(self, action_content):
		if 'name' in action_content and action_content['name'] == 'value':
			if 'new' in action_content:
				if action_content['new'] == _report_picker_current_selection:
					self.redraw_with_no_second_picker()
					self.report_pane.clear_output(wait=True)
					single_report(self.selector, output_to=self.report_pane)
				elif action_content['new'] == _report_picker_build_nobuild:
					# self.report_pane.clear_output(wait=True)
					# build_no_build_report(self.selector, output_to=self.report_pane)
					self.redraw_with_bnb_picker(second_options=self.selector.data.strategy_names)
				else:
					self.report_pane.clear_output(wait=True)
					with self.report_pane:
						print(f"{action_content['new']} is not implemented in this version of Explorer")

	def bnb_picker_action(self, action_content):
		if 'name' in action_content and action_content['name'] == 'value':
			if 'new' in action_content:
				if action_content['new'] == " ":
					self.report_pane.clear_output()
				else:
					self.report_pane.clear_output(wait=True)
					build_no_build_report(self.selector, bnb=action_content['new'], output_to=self.report_pane)
