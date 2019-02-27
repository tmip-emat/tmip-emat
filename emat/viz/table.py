
import plotly.graph_objs as go
from .widget import FigureWidget


def table_figure(
		df,
		title=None,
		header_color='#C2D4FF',
		cell_color='#F5F8FF',
):

	trace = go.Table(
		header=dict(values=list(df.columns),
					fill = dict(color=header_color),
					align = ['left'] * len(df.columns)),
		cells=dict(values=[df[c] for c in df.columns],
				   fill = dict(color=cell_color),
				   align = ['left'] * len(df.columns)),
	)

	data = [trace]

	return FigureWidget(
		data=data,
		layout=dict(
			title=title,
		),
		metadata=df,
	)

