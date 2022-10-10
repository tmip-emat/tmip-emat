
from plotly.offline import iplot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools
import numpy
import pandas
from typing import Mapping

from .widget import FigureWidget
from .common import get_name, any_names
from .colors import DEFAULT_PLOT_BACKGROUND_COLOR


class ScatterMass:

	def __init__(self, target=1000, minimum=0.1):
		if isinstance(target, ScatterMass):
			self.target = target.target
			self.minimum = target.minimum
		elif isinstance(target, Mapping):
			self.target = target.get('target', 1000)
			self.minimum = target.get('minimum', 0.1)
		else:
			self.target = target
			self.minimum = minimum

	def get_opacity(self, arr):
		"""
		Get opacity for markers.

		Parameters
		----------
		arr: array-like

		Returns
		-------
		alpha
		"""
		alpha = 1.0
		n = arr.shape[0]
		if n > self.target:
			alpha = self.target / n
		if alpha < self.minimum:
			alpha = self.minimum
		return alpha

	def get_opacity_with_selection(self, arr, selection):
		"""
		Get opacity for selected and unselected markers.

		Parameters
		----------
		arr: array-like, shape[N,...]
		selection: array-like[bool], shape[N]

		Returns
		-------
		alpha_selected, alpha_unselected
		"""
		alpha_selected, alpha_unselected = [1.0, 1.0]
		n = arr.shape[0]
		n_selected = int(selection.sum())
		n_unselect = n - n_selected

		if n_unselect > self.target:
			alpha_unselected = self.target / n_unselect
		if alpha_unselected < self.minimum:
			alpha_unselected = self.minimum

		if n_selected > self.target:
			alpha_selected = self.target / n_selected
		if alpha_selected < self.minimum:
			alpha_selected = self.minimum
		return alpha_selected, alpha_unselected



def simple_scatter_explicit(x,y, title=None, xtitle=None, ytitle=None, use_gl=True):

	Scatter = go.Scattergl if use_gl else go.Scatter
	# Create a trace
	trace = Scatter(
		x = x,
		y = y,
		mode = 'markers',
	)

	layout= go.Layout(
		title= title,
		hovermode= 'closest',
		xaxis= dict(
			title= xtitle,
			#         ticklen= 5,
			#         zeroline= False,
			#         gridwidth= 2,
		),
		yaxis=dict(
			title= ytitle,
			#         ticklen= 5,
			#         gridwidth= 2,
		),
		showlegend= False
	)

	fig= go.Figure(data=[trace], layout=layout)
	iplot(fig)
	return fig


def simple_scatter_df(df,x,y, *, size=None, hovertext=None, title=None, xtitle=None, ytitle=None, use_gl=True):
	# Create a trace

	marker = None
	if size is not None:
		marker=dict(
			size=df[size],
			# showscale=True,
		)
	Scatter = go.Scattergl if use_gl else go.Scatter
	trace = Scatter(
		x = df[x],
		y = df[y],
		mode = 'markers',
		marker=marker,
		text=df[hovertext] if hovertext is not None else None,
	)

	layout= go.Layout(
		title= title,
		hovermode= 'closest',
		xaxis= dict(
			title= xtitle or x,
		),
		yaxis=dict(
			title= ytitle or y,
		),
		showlegend= False
	)

	fig= go.Figure(data=[trace], layout=layout)
	iplot(fig)
	return fig

def scatter_graph(
		X,
		Y,
		S=None,
		*,
		df=None,
		title=None,
		showlegend=None,
		paper_bgcolor=None,
		plot_bgcolor=None,
		sizemin=None,
		sizemode='area',
		sizeref=None,
		opacity=1.0,
		cats=None,
		n_cats=None,
		cat_fmt=".2g",
		legend_title=None,
		legend_labels=None,
		axis_labels=True,
		output='widget',
		metadata=None,
		use_gl=True,
		**kwargs
):
	"""Generate a scatter plot.

	Parameters
	----------
	X, Y : array-like, str, list of array-like, or list of str
		The x and y coordinates for each point. If given as str or a list
		of str, the values are taken from the matching named columns of `df`.
	S : array-like, str, list of array-like, or list of str, optional
		The marker size for each point. If given as str or a list
		of str, the values are taken from the matching named columns of `df`.
	df : pandas.DataFrame, optional
		The dataframe from which to draw the data. This must be given if
		any of {X,Y,S} are defined by str instead of passing array-like
		objects directly.
	cats : array-like, str, list of array-like, or list of str, optional
		The category for each point.  The dataframe is actually broken on categories.
	n_cats : int, optional
		If the categories column contains sortable (i.e. numeric) data, it gets broken into
		this number of roughly similar size categories.

	Other Parameters
	----------------
	x_title, y_title: str, optional
		A label to apply to the {x,y} axis. If omitted and the x,y arguments are
		given as strings, those strings are used.
	axis_labels : bool, default True
		Set to False to suppress labelling the axes, even if x,y arguments are
		given as strings.
	paper_bgcolor : color, optional
		Sets the color of paper where the graph is drawn. default: "#fff"
	plot_bgcolor : color, optional
		Sets the color of plotting area in-between x and y axes. default: "#fff"
	legend_title : str, optional
		A title for the legend.
	legend_labels : Iterable
		Labels to use in the legend.  Order should match that in X and Y.
	output : {'widget', 'display', 'figure'}
		How to return the outputs.
		For `widget`, a :ref:`FigureWidget` object is returned.
		For `display`, a figure is displayed using the iplot command from plotly,
		and nothing is returned.
		For `figure`, a :ref:`Figure` object is returned.
	metadata : dict, optional
		A dictionary of meta-data to attach to the FigureWidget.  Only attached
		if the return type is FigureWidget.


	Returns
	-------
	fig
	"""

	x_name = get_name(X, True)
	y_name = get_name(Y, True)

	if x_name != '':
		kwargs['x_title'] = kwargs.get('x_title', x_name)
	if y_name != '':
		kwargs['y_title'] = kwargs.get('y_title', y_name)

	if not isinstance(X, list):
		X = [X]
	if not isinstance(Y, list):
		Y = [Y]
	if not isinstance(S, list):
		S = [S]
	if not isinstance(opacity, list):
		opacity = [opacity]

	longer_XY = X if (len(X) >= len(Y)) else Y

	if cats is None and legend_labels is None:
		if any_names(longer_XY):
			legend_labels = [get_name(i) for i in longer_XY]

	if cats is not None:
		if n_cats is None and isinstance(df, pandas.DataFrame):
			uniq = numpy.unique(df[cats])
			df = {
				k: df[df[cats] == k]
				for k in uniq
			}
		elif n_cats is not None and isinstance(df, pandas.DataFrame):
			breaks = numpy.percentile(df[cats], numpy.linspace(0,100,n_cats+1))
			breaks[-1] += numpy.abs(breaks[-1])*0.001
			df = {
				f"{lowbound:{cat_fmt}}-{highbound:{cat_fmt}}": df[(df[cats] >= lowbound) & (df[cats] < highbound)]
				for lowbound, highbound in zip(breaks[:-1], breaks[1:])
			}

	if isinstance(df, dict):
		DF = df
	else:
		DF = {'':df}


	if showlegend is None:
		showlegend = (len(X) > 1)

	X_data = {}
	Y_data = {}
	S_data = {}
	max_S = 0

	for n,df in DF.items():

		X_data[n] = [(df[i] if isinstance(i, str) and df is not None else i) for i in X]
		Y_data[n] = [(df[i] if isinstance(i, str) and df is not None else i) for i in Y]
		S_data[n] = [(df[i] if isinstance(i, str) and df is not None else i) for i in S]

		if sizeref is None:
			try:
				max_S = max(max_S, max(max(s) for s in S_data[n]))
			except (TypeError, ValueError):
				pass

	if sizeref is None:
		sizeref = 2. * max_S / (40. ** 2)

	traces = []

	if legend_title is None and cats is not None and n_cats is not None:
		legend_title = cats

	Scatter = go.Scattergl if use_gl else go.Scatter

	if legend_title is not None:
		dummy_trace = Scatter(
			x=[None], y=[None],
			name=f'<b>{legend_title}</b>',
			# set opacity = 0
			line={'color': 'rgba(0, 0, 0, 0)'}
		)
		traces.append(dummy_trace)

	for n,df in DF.items():

		traces += [
			Scatter(
				x = x,
				y = y,
				mode = 'markers',
				marker=dict(
					size=s,
					sizemode=sizemode,
					sizeref=sizeref,
					sizemin=sizemin,
					opacity=opaque,
				),
				name=n if legend_label=='' else legend_label,
			)
			for x,y,s,tracenum,legend_label,opaque in zip(
				itertools.cycle(X_data[n]),
				itertools.cycle(Y_data[n]),
				itertools.cycle(S_data[n]),
				range(max(len(X_data[n]), len(Y_data[n]), len(S_data[n]))),
				itertools.cycle(legend_labels) if legend_labels is not None else itertools.cycle(['']),
				itertools.cycle(opacity),
			)
		]


	layout= go.Layout(
		title= title,
		hovermode= 'closest',
		xaxis= {i[2:]:j for i,j in kwargs.items() if i[:2]=='x_'},
		yaxis= {i[2:]:j for i,j in kwargs.items() if i[:2]=='y_'},
		showlegend= showlegend,
		paper_bgcolor=paper_bgcolor,
		plot_bgcolor=plot_bgcolor,
	)

	if not axis_labels:
		layout.xaxis.title = None
		layout.yaxis.title = None

	if output=='widget':
		figwid = FigureWidget(data=traces, layout=layout, metadata=metadata)
		return figwid
	# elif hasattr(output, 'batch_update'):
	# 	with output.batch_update():
	# 		output.data = traces
	# 		output.layout = layout
	# 	return
	else:
		fig= go.Figure(data=traces, layout=layout)
		if 'display' in output:
			iplot(fig)
		if 'figure' in output:
			return fig




def scatter_graph_row(
		X,
		Y,
		S=None,
		C=None,
		*,
		df=None,
		title=None,
		showlegend=False,
		paper_bgcolor=None,
		plot_bgcolor=None,
		sizemin=None,
		sizemode='area',
		sizeref=None,
		cats=None,
		n_cats=None,
		cat_fmt=".2g",
		legend_title=None,
		legend_labels=None,
		axis_labels=True,
		output='widget',
		metadata=None,
		marker_opacity=1.0,
		layout=None,
		short_name_func=None,
		use_gl=True,
		**kwargs
):
	"""Generate a scatter plot.

	Parameters
	----------
	X : list of array-like, or list of str
		The x coordinates for each point. If given as a list
		of str, the values are taken from the matching named columns of `df`.
	Y : array-like, or str
		The y coordinates for each point. If given as a str, the values are
		taken from the matching named columns of `df`.
	S : array-like, str, list of array-like, or list of str, optional
		The marker size for each point. If given as str or a list
		of str, the values are taken from the matching named columns of `df`.
		Must either be one array or match length of X.
	df : pandas.DataFrame, optional
		The dataframe from which to draw the data. This must be given if
		any of {X,Y,S} are defined by str instead of passing array-like
		objects directly.
	cats : array-like, str, list of array-like, or list of str, optional
		The category for each point.  The dataframe is actually broken on categories.
	n_cats : int, optional
		If the categories column contains sortable (i.e. numeric) data, it gets broken into
		this number of roughly similar size categories.

	Other Parameters
	----------------
	y_title: str, optional
		A label to apply to the y axis. If omitted and the y arguments are
		given as strings, those strings are used.
	axis_labels : bool, default True
		Set to False to suppress labelling the axes, even if x,y arguments are
		given as strings.
	paper_bgcolor : color, optional
		Sets the color of paper where the graph is drawn. default: "#fff"
	plot_bgcolor : color, optional
		Sets the color of plotting area in-between x and y axes. default: "#fff"
	legend_title : str, optional
		A title for the legend.
	legend_labels : Iterable
		Labels to use in the legend.  Order should match that in X and Y.
	output : {'widget', 'display', 'figure'}
		How to return the outputs.
		For `widget`, a :ref:`FigureWidget` object is returned.
		For `display`, a figure is displayed using the iplot command from plotly,
		and nothing is returned.
		For `figure`, a :ref:`Figure` object is returned.
	metadata : dict, optional
		A dictionary of meta-data to attach to the FigureWidget.  Only attached
		if the return type is FigureWidget.
	short_name_func : callable, optional
		A function that converts names to short names, which might display better
		on the figure.

	Returns
	-------
	fig
	"""

	if short_name_func is None:
		short_name_func = lambda x: x

	x_names = [get_name(x, True) for x in X]
	y_name = get_name(Y, True)

	if y_name != '':
		kwargs['y_title'] = kwargs.get('y_title', short_name_func(y_name))

	if not isinstance(X, list):
		X = [X]
	if not isinstance(Y, list):
		Y = [Y]
	if not isinstance(S, list):
		S = [S]
	if not isinstance(C, list):
		C = [C]

	longer_XY = X if (len(X) >= len(Y)) else Y

	if cats is None and legend_labels is None:
		if any_names(longer_XY):
			legend_labels = [get_name(i) for i in longer_XY]

	if cats is not None:
		if n_cats is None and isinstance(df, pandas.DataFrame):
			uniq = numpy.unique(df[cats])
			df = {
				k: df[df[cats] == k]
				for k in uniq
			}
		elif n_cats is not None and isinstance(df, pandas.DataFrame):
			breaks = numpy.percentile(df[cats], numpy.linspace(0,100,n_cats+1))
			breaks[-1] += numpy.abs(breaks[-1])*0.001
			df = {
				f"{lowbound:{cat_fmt}}-{highbound:{cat_fmt}}": df[(df[cats] >= lowbound) & (df[cats] < highbound)]
				for lowbound, highbound in zip(breaks[:-1], breaks[1:])
			}


	if showlegend is None:
		showlegend = (len(X) > 1)

	max_S = 0
	
	def _get_val(i, df):
		if df is not None:
			if isinstance(i, str):
				if i in df.columns:
					return df[i]
				else:
					return [None] * len(df)
		return i

	X_data = [_get_val(i, df) for i in X]
	Y_data = [_get_val(i, df) for i in Y]
	S_data = [_get_val(i, df) for i in S]

	if sizeref is None:
		try:
			max_S = max(max_S, max(max(s) for s in S_data))
		except (TypeError, ValueError):
			pass

	if sizeref is None:
		sizeref = 2. * max_S / (40. ** 2)

	traces = []

	if legend_title is None and cats is not None and n_cats is not None:
		legend_title = cats

	Scatter = go.Scattergl if use_gl else go.Scatter
	if legend_title is not None:
		dummy_trace = Scatter(
			x=[None], y=[None],
			name=f'<b>{legend_title}</b>',
			# set opacity = 0
			line={'color': 'rgba(0, 0, 0, 0)'}
		)
		traces.append(dummy_trace)

	padding = 0.3
	X_data_1 = []
	X_data_ticks = {}

	from .perturbation import perturb_categorical
	for tracenum, x in enumerate(X_data):
		x, x_ticktext, x_tickvals, x_range, valid_scales = perturb_categorical(x, range_padding=padding)
		X_data_ticks[tracenum] = (
			x_range,
			x_tickvals,
			x_ticktext,
		)
		X_data_1.append(x)


	for x, y, s, tracenum, legend_label, tracecolor in zip(
			itertools.cycle(X_data_1),
			itertools.cycle(Y_data),
			itertools.cycle(S_data),
			range(max(len(X_data), len(Y_data), len(S_data))),
			itertools.cycle(legend_labels) if legend_labels is not None else itertools.cycle(['']),
			itertools.cycle(C),
	):
		if x is not None:
			i = Scatter(
				x = x,
				y = y,
				mode = 'markers',
				marker=dict(
					size=s,
					sizemode=sizemode,
					sizeref=sizeref,
					sizemin=sizemin,
					opacity=marker_opacity,
					color=tracecolor,
				),
				name=legend_label,
				xaxis=f'x{tracenum+1}',
			)
			traces.append(i)

	if isinstance(output, go.FigureWidget):
		with output.batch_update():
			for t_num, t in enumerate(traces):
				output.add_trace(t)
		return output

	n_traces = max(len(X_data), len(Y_data), len(S_data))
	domain_starts = (numpy.arange(n_traces)) / (n_traces - 0.1)
	domain_stops = (numpy.arange(n_traces) + 0.9) / (n_traces - 0.1)

	xaxis_dicts = {}

	for i, dom0, dom1, t in zip(range(n_traces), domain_starts, domain_stops, x_names):
		xaxis_dicts[f'xaxis{i+1}'] = dict(
			domain=[dom0,dom1],
			title=short_name_func(t),
			zeroline=False,
		)
		if i in X_data_ticks:
			xaxis_dicts[f'xaxis{i + 1}']['range'] = X_data_ticks[i][0]
			xaxis_dicts[f'xaxis{i + 1}']['tickvals'] = X_data_ticks[i][1]
			xaxis_dicts[f'xaxis{i + 1}']['ticktext'] = X_data_ticks[i][2]

	layout_kwds = {} if layout is None else layout
	layout= go.Layout(
		title= title,
		hovermode= 'closest',
		# xaxis= {i[2:]:j for i,j in kwargs.items() if i[:2]=='x_'},
		yaxis= {i[2:]:j for i,j in kwargs.items() if i[:2]=='y_'},
		showlegend= showlegend,
		paper_bgcolor=paper_bgcolor,
		plot_bgcolor=plot_bgcolor or DEFAULT_PLOT_BACKGROUND_COLOR,
		**xaxis_dicts,
		**layout_kwds,
	)

	# if not axis_labels:
	# 	layout.xaxis.title = None
	# 	layout.yaxis.title = None

	if output=='widget':
		figwid = FigureWidget(data=traces, layout=layout, metadata=metadata)
		return figwid
	# elif hasattr(output, 'batch_update'):
	# 	with output.batch_update():
	# 		output.data = traces
	# 		output.layout = layout
	# 	return
	else:
		fig= go.Figure(data=traces, layout=layout)
		if 'display' in output:
			iplot(fig)
		if 'figure' in output:
			return fig
