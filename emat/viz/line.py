


from plotly.offline import iplot
import plotly.graph_objs as go
import itertools
import numpy
import pandas

from .widget import FigureWidget
from .common import get_name, any_names


def line_graph(
		X,
		Y=None,
		*,
		df=None,
		output='widget',
		title=None,
		showlegend=None,
		paper_bgcolor=None,
		plot_bgcolor=None,
		legend_title=None,
		legend_labels=None,
		axis_labels=True,
		interpolate='spline',
		fill='tonexty',
		mode='lines+markers',
		metadata=None,
		**kwargs
):
	"""
	Generate a plotly line graph figure.

	Parameters
	----------
	X, Y : array or str or list of array or list of str
		The x-positions and y-positions for each trace.
		If df is given, these can be the names of columns in
		that DataFrame.  If only X is given, then X is interpreted as
		y-positions and the x-positions are assumed sequential from zero.
	df : pandas.DataFrame, optional
		The dataframe from which to draw the data. If not given, X and Y
		must be actual data arrays, not names.
	output : {'widget', 'display', 'figure'}
		How to return the outputs.
		For `widget`, a :ref:`FigureWidget` object is returned.
		For `display`, a figure is displayed using the iplot command from plotly,
		and nothing is returned.
		For `figure`, a :ref:`Figure` object is returned.



	Other Parameters
	----------------
	title : str
		A chart figure title
	showlegend : bool, optional
		Explicitly declare whether a legend should be shown.  If not given, a
		legend is shown if there is more than one trace.
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
	metadata : dict, optional
		A dictionary of meta-data to attach to the FigureWidget.  Only attached
		if the return type is FigureWidget.
	interpolate : {'linear', 'spline', 'hv', 'vh', 'hvh', 'vhv' }, default 'spline'
		How to draw lines between points.  See	https://plot.ly/python/line-charts/
	fill : str
		The fill setting for traces.
	mode : str
		The mode setting for traces.

	Returns
	-------
	FigureWidget or Figure
	"""

	if Y is None and X is not None:
		X, Y = Y, X

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

	longer_XY = X if (len(X) >= len(Y)) else Y

	if legend_labels is None:
		if any_names(longer_XY):
			legend_labels = [get_name(i) for i in longer_XY]

	if isinstance(df, dict):
		DF = df
	else:
		DF = {'':df}

	if showlegend is None:
		showlegend = ((len(X) > 1) or (len(Y) > 1)) and legend_labels is not None

	X_data = {}
	Y_data = {}

	for n,df in DF.items():

		X_data[n] = [(df[i] if isinstance(i, str) and df is not None else i) for i in X]
		Y_data[n] = [(df[i] if isinstance(i, str) and df is not None else i) for i in Y]

	traces = []

	if legend_title is not None:
		dummy_trace = go.Scatter(
			x=[None], y=[None],
			name=f'<b>{legend_title}</b>',
			# set opacity = 0
			line={'color': 'rgba(0, 0, 0, 0)'}
		)
		traces.append(dummy_trace)

	if fill=='tonexty':
		fill_method = lambda tracenum: 'tonexty' if tracenum==0 else 'tonexty'
	elif fill == 'tozeroy':
		fill_method = lambda tracenum: 'tozeroy'
	else:
		fill_method = lambda tracenum: None

	for n,df in DF.items():

		traces += [
			go.Scatter(
				x = x,
				y = y,
				mode=mode,
				line=dict(
					shape=interpolate,
				) if interpolate is not None else None,
				name=n if legend_label=='' else legend_label,
				fill=fill_method(tracenum),
			)
			for x,y,tracenum,legend_label in zip(
				itertools.cycle(X_data[n]),
				itertools.cycle(Y_data[n]),
				range(max(len(X_data[n]), len(Y_data[n]), )),
				itertools.cycle(legend_labels) if legend_labels is not None else itertools.cycle(['']),
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

