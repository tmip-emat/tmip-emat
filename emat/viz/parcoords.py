
import numpy
import textwrap
import plotly.graph_objs as go
import itertools

from ..workbench.em_framework.parameters import Category, CategoricalParameter, BooleanParameter
from ..workbench.em_framework.outcomes import ScalarOutcome

from ipywidgets import VBox, HBox, Checkbox, Accordion
from ..analysis.widgets import NamedCheckbox
import ipywidgets as widget

def perturb(x, epsilon=0.05):
	return x + numpy.random.uniform(-epsilon, epsilon)

def _wrap_with_br(text, width=70, **kwargs):
	return "<br>   ".join(textwrap.wrap(text, width=width, **kwargs))

SYMBOL_MIMIMIZE = '(M-)'    # '⊖'
SYMBOL_MAXIMIZE = '(M+)'    #'⊕'
SYMBOL_INFOMEASURE = '(M)'  #'⊙'
SYMBOL_LEVER = '(L)'        #'⎆' # ୰
SYMBOL_UNCERTAINTY = '(X)'  #'‽'

def _prefix_symbols(scope, robustness_functions):
	prefix_chars = {}
	if robustness_functions is not None:
		for rf in robustness_functions:
			if rf.kind < 0:
				prefix_chars[rf.name] = SYMBOL_MIMIMIZE+' '
			elif rf.kind > 0:
				prefix_chars[rf.name] = SYMBOL_MAXIMIZE+' '
			elif rf.kind == 0:
				prefix_chars[rf.name] = SYMBOL_INFOMEASURE+' '
	if scope is not None:
		for meas in scope.get_measures():
			if meas.kind < 0:
				prefix_chars[meas.name] = SYMBOL_MIMIMIZE+' '
			elif meas.kind > 0:
				prefix_chars[meas.name] = SYMBOL_MAXIMIZE+' '
			elif meas.kind == 0:
				prefix_chars[meas.name] = SYMBOL_INFOMEASURE+' '
		for col in scope.get_lever_names():
			prefix_chars[col] = SYMBOL_LEVER+' ' # ୰
		for col in scope.get_uncertainty_names():
			prefix_chars[col] = SYMBOL_UNCERTAINTY+' '
	return prefix_chars

def parallel_coords(
		df,
		scope=None,
		flip_dims=(),
		robustness_functions=None,
		color_dim=0,
		colorscale='Viridis',
		title=None,
):
	"""Generate a parallel coordinates figure.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot.
	scope : emat.Scope, optional
		Categorical levers and uncertainties are extracted from the scope.
	"""

	df = df.copy(deep=True)

	categorical_parameters = df.columns[df.dtypes == 'category']
	bool_columns = df.columns[df.dtypes == bool]

	# Change the range from plain min/max to something else
	column_ranges = {}
	tickvals = {}
	ticktext = {}

	prefix_chars = _prefix_symbols(scope, robustness_functions)

	for c in categorical_parameters:
		df[c] = df[c].apply(lambda z: z.value if isinstance(z,Category) else z)
		n_cats = len(df[c].cat.categories)
		min_cat, max_cat = 0, n_cats-1
		col_range = column_ranges[c] = [min_cat-0.1, max_cat+0.1]
		tickvals[c] = [col_range[0]] + list(range(min_cat, max_cat+1)) + [col_range[1]]
		ticktext[c] = [""] + [str(i) for i in df[c].cat.categories] + [""]
		df[c] = df[c].cat.codes.apply( lambda x: perturb(x) )

	for c in bool_columns:
		df[c] = df[c].astype(float)
		column_ranges[c] = [-0.1, 1.1]
		tickvals[c] = [-0.1, 0, 1, 1.1]
		ticktext[c] = ["", "False", "True",""]
		df[c] = df[c].apply( lambda x: perturb(x) )

	flips = set(flip_dims)

	# flip all MINIMIZE outcomes (or unflip them if previously marked as flip)
	if robustness_functions is not None:
		for k in robustness_functions:
			if k.kind == ScalarOutcome.MINIMIZE:
				if k.name in flips:
					flips.remove(k.name)
				else:
					flips.add(k.name)
	if scope is not None:
		for k in scope.get_measures():
			if k.kind == ScalarOutcome.MINIMIZE:
				if k.name in flips:
					flips.remove(k.name)
				else:
					flips.add(k.name)

	parallel_dims = [
		dict(
			range=column_ranges.get(
				col, [
					df[col].min(),
					df[col].max(),
				] if col not in flips else [
					df[col].max(),
					df[col].min(),
				]
			),
			label=_wrap_with_br(prefix_chars.get(col, '')+(col if scope is None else scope.shortname(col)), width=24),
			values=df[col],
			tickvals=tickvals.get(col, None),
			ticktext=ticktext.get(col, None),
			name=col,
		)
		for col in df.columns if col not in scope.get_constant_names()
	]

	## Line coloring dimension
	if isinstance(color_dim, int):
		color_dim = df.columns[color_dim]

	parallel_line = dict(
		color=df[color_dim],
		colorscale=colorscale,
		showscale=True,
		reversescale=True,
		cmin=df[color_dim].min(),
		cmax=df[color_dim].max(),
		colorbar_title_text=color_dim,
		colorbar_title_side='right',
	)

	pc = go.Parcoords(
		line=parallel_line,
		dimensions=parallel_dims,
		labelfont=dict(
			color="#AA0000",
		),
		labelangle=-90,
		domain=dict(
			y=[0,0.7],
		)
	)

	return go.FigureWidget(
		[pc],
		layout=dict(
			title=title,
		)
	)

_NOTHING = ' '
_CLEAR_CONSTRAINT_RANGES = '   Clear Selection Constraints'
_SELECT_ALL_UNCS     = '   Show All Exogenous Uncertainties'
_DESELECT_ALL_UNCS   = '   Hide All Exogenous Uncertainties'
_SELECT_ALL_LEVERS   = '   Show All Policy Levers'
_DESELECT_ALL_LEVERS = '   Hide All Policy Levers'
_SELECT_ALL_MEAS     = '   Show All Performance Measures'
_DESELECT_ALL_MEAS   = '   Hide All Performance Measures'


class ParCoordsViewer(VBox):

	def __init__(
			self,
			data,
			scope,
			robustness_functions=None,
			initial_max_active_measures = 5,
	):
		self.data = data
		self.scope = scope
		self.robustness_functions = robustness_functions
		if self.robustness_functions is None:
			self.robustness_functions = ()

		self.parcoords = parallel_coords(
				self.data,
				scope=self.scope,
				flip_dims=(),
				robustness_functions=robustness_functions,
				color_dim=0,
				colorscale='Viridis',
				title=None,
		)

		self.dim_activators = []
		self.dim_activators_by_name = {}
		self.out_logger = widget.Output()

		prefix_chars = _prefix_symbols(scope, robustness_functions)

		n_active_measures = 0
		measure_names = set(self.scope.get_measure_names())
		if robustness_functions is not None:
			measure_names |= set(rf.name for rf in robustness_functions)
		for i in self.data.columns:
			if i in self.scope.get_constant_names():
				continue
			short_i = i
			if self.scope is not None:
				short_i = self.scope.shortname(i)
			i_value = True
			if i in measure_names:
				if n_active_measures >= initial_max_active_measures:
					i_value = False
					for dim in self.parcoords.data[0].dimensions:
						if dim.name == i:
							dim.visible = False
				else:
					n_active_measures += 1
			cb = NamedCheckbox(description=prefix_chars.get(i,'')+short_i, value=i_value, name=i, description_tooltip=i)
			cb.observe(self._on_dim_choose_toggle, names='value')
			self.dim_activators.append(cb)
			self.dim_activators_by_name[i] = cb

		self.dim_choose = Accordion(
			children=[
				widget.Box(
					self.dim_activators,
					layout=widget.Layout(flex_flow='row wrap')
				)
			],
			layout=widget.Layout(width='100%')
		)
		self.dim_choose.set_title(0, 'Axes')
		self.dim_choose.selected_index = None

		self.color_dim_choose = widget.Dropdown(
			options=['< None >']+list(self.data.columns),
			description='Colorize:',
			value=self.data.columns[0],
		)
		self.color_dim_choose.observe(self._on_color_choose, names='value')

		self.select_menu = widget.Dropdown(
			options=[
				_NOTHING,
				_CLEAR_CONSTRAINT_RANGES,
				"-- (X) Uncertainties --",
				_SELECT_ALL_UNCS    ,
				_DESELECT_ALL_UNCS  ,
				"-- (L) Levers --",
				_SELECT_ALL_LEVERS  ,
				_DESELECT_ALL_LEVERS,
				"-- (M) Measures --",
				_SELECT_ALL_MEAS    ,
				_DESELECT_ALL_MEAS  ,
			],
			description='View:',
			value=_NOTHING,
		)
		self.select_menu.observe(self._on_select_menu, names='value')

		self.menus = HBox([
			self.color_dim_choose,
			self.select_menu,
		])

		self.symbol_legend = widget.HTML(f"""
			{SYMBOL_MIMIMIZE} Performance Measure to Minimize<br>
			{SYMBOL_MAXIMIZE} Performance Measure to Maximize<br>
			{SYMBOL_INFOMEASURE} Performance Measure without preferred direction<br>
			{SYMBOL_LEVER} Policy Lever<br>
			{SYMBOL_UNCERTAINTY} Exogenous Uncertainty
		""")

		super().__init__(
			[
				self.parcoords,
				self.menus,
				self.dim_choose,
				self.symbol_legend,
			],
			layout=dict(
				align_items='center',
			)
		)

	def clear_all_constraint_ranges(self):
		"""
		Clear any constraint ranges across all dimensions.
		"""
		with self.parcoords.batch_update():
			for dim in self.parcoords.data[0].dimensions:
				dim.constraintrange = None

	def _on_dim_choose_toggle(self, payload):
		for dim in self.parcoords.data[0].dimensions:
			if dim.name == payload['owner'].name:
				dim.visible = payload['new']

	def _on_color_choose(self, payload):
		with self.out_logger:
			try:
				with self.parcoords.batch_update():
					color_dim_name = payload['new']
					if color_dim_name == '< None >':
						self.parcoords.data[0].line.color = 'rgb(200,0,0)'
						self.parcoords.data[0].line.showscale = False
					else:
						color_data = self.data[payload['new']]
						self.parcoords.data[0].line.color = color_data
						self.parcoords.data[0].line.showscale = True
						if self.scope is not None:
							self.parcoords.data[0].line.colorbar.title.text = self.scope.shortname(payload['new'])
						else:
							self.parcoords.data[0].line.colorbar.title.text = payload['new']
						if color_data.dtype == numpy.bool_:
							self.parcoords.data[0].line.cmin = 0
							self.parcoords.data[0].line.cmax = 1
						else:
							self.parcoords.data[0].line.cmin = color_data.min()
							self.parcoords.data[0].line.cmax = color_data.max()
			except:
				import traceback
				traceback.print_exc()
				raise

	def _on_select_menu(self, payload):
		with self.out_logger:
			try:
				with self.parcoords.batch_update():
					command_name = payload['new']
					if command_name == _NOTHING:
						pass
					else:
						if command_name == _SELECT_ALL_MEAS:
							for meas in itertools.chain(self.scope.get_measure_names(), [_.name for _ in self.robustness_functions]):
								if meas in self.dim_activators_by_name:
									self.dim_activators_by_name[meas].value = True
						elif command_name == _DESELECT_ALL_MEAS:
							for meas in itertools.chain(self.scope.get_measure_names(), [_.name for _ in self.robustness_functions]):
								if meas in self.dim_activators_by_name:
									self.dim_activators_by_name[meas].value = False
						elif command_name == _SELECT_ALL_LEVERS:
							for meas in self.scope.get_lever_names():
								if meas in self.dim_activators_by_name:
									self.dim_activators_by_name[meas].value = True
						elif command_name == _DESELECT_ALL_LEVERS:
							for meas in self.scope.get_lever_names():
								if meas in self.dim_activators_by_name:
									self.dim_activators_by_name[meas].value = False
						elif command_name == _SELECT_ALL_UNCS:
							for meas in self.scope.get_uncertainty_names():
								if meas in self.dim_activators_by_name:
									self.dim_activators_by_name[meas].value = True
						elif command_name == _DESELECT_ALL_UNCS:
							for meas in self.scope.get_uncertainty_names():
								if meas in self.dim_activators_by_name:
									self.dim_activators_by_name[meas].value = False
						elif command_name == _CLEAR_CONSTRAINT_RANGES:
							self.clear_all_constraint_ranges()
						self.select_menu.value = _NOTHING
			except:
				import traceback
				traceback.print_exc()
				raise
