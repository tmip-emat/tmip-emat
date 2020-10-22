
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ....viz import colors

def empty_two_dim_figure(
		use_gl=True,
		marker_opacity=(1.0,1.0),
):
	"""

	Args:
		use_gl (bool, default True):
			Use the WebGL versions of plots.
		marker_opacity (tuple, default (1.0,1.0)):
			The opacity of (unselected, selected) markers in
			the scatter plot.

	Returns:
		figure
	"""
	Scatter = go.Scattergl if use_gl else go.Scatter
	scattergraph = Scatter(
		x=None,
		y=None,
		mode='markers',
		marker=dict(
			opacity=marker_opacity[0],
			color=None,
			colorscale=[[0, colors.DEFAULT_BASE_COLOR], [1, colors.DEFAULT_HIGHLIGHT_COLOR]],
			cmin=0,
			cmax=1,
		),
		name='Cases',
	)

	x_hist = go.Histogram(
		x=None,
		# name='x density',
		marker=dict(
			color=colors.DEFAULT_BASE_COLOR,
			# opacity=0.7,
		),
		yaxis='y2',
		bingroup='xxx',
	)

	y_hist = go.Histogram(
		y=None,
		# name='y density',
		marker=dict(
			color=colors.DEFAULT_BASE_COLOR,
			# opacity=0.7,
		),
		xaxis='x2',
		bingroup='yyy',
	)

	x_hist_s = go.Histogram(
		x=None,
		marker=dict(
			color=colors.DEFAULT_HIGHLIGHT_COLOR,
			# opacity=0.7,
		),
		yaxis='y2',
		bingroup='xxx',
	)

	y_hist_s = go.Histogram(
		y=None,
		marker=dict(
			color=colors.DEFAULT_HIGHLIGHT_COLOR,
			# opacity=0.7,
		),
		xaxis='x2',
		bingroup='yyy',
	)

	fig = go.Figure()
	scattergraph = fig.add_trace(scattergraph).data[-1]
	x_hist = fig.add_trace(x_hist).data[-1]
	y_hist = fig.add_trace(y_hist).data[-1]
	x_hist_s = fig.add_trace(x_hist_s).data[-1]
	y_hist_s = fig.add_trace(y_hist_s).data[-1]

	fig.layout = dict(
		xaxis=dict(
			domain=[0, 0.85],
			showgrid=True,
			# title=self._df.data.columns[0],
		),
		yaxis=dict(
			domain=[0, 0.85],
			showgrid=True,
			# title=self._df.data.columns[-1],
		),

		xaxis2=dict(
			domain=[0.85, 1],
			showgrid=True,
			zeroline=True,
			zerolinecolor='#FFF',
			zerolinewidth=4,
		),
		yaxis2=dict(
			domain=[0.85, 1],
			showgrid=True,
			zeroline=True,
			zerolinecolor='#FFF',
			zerolinewidth=4,
		),

		barmode="overlay",
		showlegend=False,
		margin=dict(l=10, r=10, t=10, b=10),
		dragmode="lasso",
	)

	return fig