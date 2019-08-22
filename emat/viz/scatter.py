
from plotly.offline import iplot
import plotly.graph_objs as go
import itertools
import numpy
import pandas
from typing import Mapping

from .widget import FigureWidget
from .common import get_name, any_names

def simple_scatter_explicit(x,y, title=None, xtitle=None, ytitle=None):
	# Create a trace
	trace = go.Scattergl(
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


def simple_scatter_df(df,x,y, *, size=None, hovertext=None, title=None, xtitle=None, ytitle=None):
	# Create a trace

	marker = None
	if size is not None:
		marker=dict(
			size=df[size],
			# showscale=True,
		)

	trace = go.Scattergl(
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

	if legend_title is not None:
		dummy_trace = go.Scattergl(
			x=[None], y=[None],
			name=f'<b>{legend_title}</b>',
			# set opacity = 0
			line={'color': 'rgba(0, 0, 0, 0)'}
		)
		traces.append(dummy_trace)

	for n,df in DF.items():

		traces += [
			go.Scattergl(
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


	Returns
	-------
	fig
	"""

	x_names = [get_name(x, True) for x in X]
	y_name = get_name(Y, True)

	if y_name != '':
		kwargs['y_title'] = kwargs.get('y_title', y_name)

	if not isinstance(X, list):
		X = [X]
	if not isinstance(Y, list):
		Y = [Y]
	if not isinstance(S, list):
		S = [S]

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

	X_data = [(df[i] if isinstance(i, str) and df is not None else i) for i in X]
	Y_data = [(df[i] if isinstance(i, str) and df is not None else i) for i in Y]
	S_data = [(df[i] if isinstance(i, str) and df is not None else i) for i in S]

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

	if legend_title is not None:
		dummy_trace = go.Scattergl(
			x=[None], y=[None],
			name=f'<b>{legend_title}</b>',
			# set opacity = 0
			line={'color': 'rgba(0, 0, 0, 0)'}
		)
		traces.append(dummy_trace)

	padding = 0.3
	X_data_1 = []
	X_data_ticks = {}

	for tracenum, x in enumerate(X_data):
		if hasattr(x, 'dtype'):
			try:
				perturb_ = numpy.issubdtype(x.dtype, numpy.bool_)
			except:
				perturb_ = False
			if perturb_:
				s_ = x.size*0.01
				s_ = s_ / (1+s_)
				epsilon = 0.05 + 0.25 * s_
				x = numpy.asarray(x, dtype=float) + numpy.random.uniform(-epsilon, epsilon, size=x.shape)
				X_data_ticks[tracenum] = (
					[-epsilon-padding, 1+epsilon+padding],
					[-epsilon-padding, 0, 1, 1+epsilon+padding],
					["", "False", "True", ""],
				)
		X_data_1.append(x)


	for x, y, s, tracenum, legend_label in zip(
			itertools.cycle(X_data_1),
			itertools.cycle(Y_data),
			itertools.cycle(S_data),
			range(max(len(X_data), len(Y_data), len(S_data))),
			itertools.cycle(legend_labels) if legend_labels is not None else itertools.cycle(['']),
	):
		i = go.Scattergl(
			x = x,
			y = y,
			mode = 'markers',
			marker=dict(
				size=s,
				sizemode=sizemode,
				sizeref=sizeref,
				sizemin=sizemin,
				opacity=marker_opacity,
			),
			name=legend_label,
			xaxis=f'x{tracenum+1}',
		)
		traces.append(i)


	n_traces = max(len(X_data), len(Y_data), len(S_data))
	domain_starts = (numpy.arange(n_traces)) / (n_traces - 0.1)
	domain_stops = (numpy.arange(n_traces) + 0.9) / (n_traces - 0.1)

	xaxis_dicts = {}

	for i, dom0, dom1, t in zip(range(n_traces), domain_starts, domain_stops, x_names):
		xaxis_dicts[f'xaxis{i+1}'] = dict(
			domain=[dom0,dom1],
			title=t,
			zeroline=False,
		)
		if i in X_data_ticks:
			xaxis_dicts[f'xaxis{i + 1}']['range'] = X_data_ticks[i][0]
			xaxis_dicts[f'xaxis{i + 1}']['tickvals'] = X_data_ticks[i][1]
			xaxis_dicts[f'xaxis{i + 1}']['ticktext'] = X_data_ticks[i][2]

	layout= go.Layout(
		title= title,
		hovermode= 'closest',
		# xaxis= {i[2:]:j for i,j in kwargs.items() if i[:2]=='x_'},
		yaxis= {i[2:]:j for i,j in kwargs.items() if i[:2]=='y_'},
		showlegend= showlegend,
		paper_bgcolor=paper_bgcolor,
		plot_bgcolor=plot_bgcolor,
		**xaxis_dicts,
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
