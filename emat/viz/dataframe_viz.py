

import numpy
import plotly.graph_objs as go
from ipywidgets import HBox, VBox, Dropdown, Label
from . import colors

class DataFrameViewer(HBox):

	def __init__(
			self,
			df,
			selection=None,
			box=None,
			target_marker_opacity=500,
			minimum_marker_opacity=0.25,
	):
		self.df = df

		self.selection = selection
		self.box = box

		self.x_axis_choose = Dropdown(
			options=self.df.columns,
			description='X Axis',
			value=self.df.columns[0],
		)
		self.y_axis_choose = Dropdown(
			options=self.df.columns,
			description='Y Axis',
			value=self.df.columns[-1],
		)

		self.axis_choose = VBox(
			[
				self.x_axis_choose,
				self.y_axis_choose,
			],
			layout=dict(
				overflow='hidden',
			)
		)

		self.minimum_marker_opacity = minimum_marker_opacity
		self.target_marker_opacity = target_marker_opacity
		marker_opacity = self._compute_marker_opacity()

		self._x_data_range = [0,1]
		self._y_data_range = [0,1]

		self.scattergraph = go.Scattergl(
			x=df.iloc[:,0],
			y=df.iloc[:,-1],
			mode = 'markers',
			marker=dict(
				opacity=marker_opacity[0],
				color=colors.DEFAULT_BASE_COLOR,
			),
		)

		self.x_hist = go.Histogram(
			x=df.iloc[:,0],
			name='x density',
			marker=dict(
				color=colors.DEFAULT_BASE_COLOR,
				#opacity=0.7,
			),
			yaxis='y2',
			bingroup='xxx',
		)

		self.y_hist = go.Histogram(
			y=df.iloc[:,-1],
			name='y density',
			marker=dict(
				color=colors.DEFAULT_BASE_COLOR,
				#opacity=0.7,
			),
			xaxis='x2',
			bingroup='yyy',
		)

		self.scattergraph_s = go.Scattergl(
			x=None,
			y=None,
			mode = 'markers',
			marker=dict(
				opacity=marker_opacity[1],
				color=colors.DEFAULT_HIGHLIGHT_COLOR,
			),
		)

		self.x_hist_s = go.Histogram(
			x=None,
			marker=dict(
				color=colors.DEFAULT_HIGHLIGHT_COLOR,
				#opacity=0.7,
			),
			yaxis='y2',
			bingroup='xxx',
		)

		self.y_hist_s = go.Histogram(
			y=None,
			marker=dict(
				color=colors.DEFAULT_HIGHLIGHT_COLOR,
				#opacity=0.7,
			),
			xaxis='x2',
			bingroup='yyy',
		)

		self.graph = go.FigureWidget([
			self.scattergraph,
			self.x_hist,
			self.y_hist,
			self.scattergraph_s,
			self.x_hist_s,
			self.y_hist_s,
		])


		self.graph.layout=dict(
			xaxis=dict(
				domain=[0, 0.85],
				showgrid=True,
				title=self.df.columns[0],
			),
			yaxis=dict(
				domain=[0, 0.85],
				showgrid=True,
				title=self.df.columns[-1],
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
		)

		self.x_axis_choose.observe(self._observe_change_column_x, names='value')
		self.y_axis_choose.observe(self._observe_change_column_y, names='value')

		super().__init__(
			[
				self.graph,
				self.axis_choose,
			],
			layout=dict(
				align_items='center',
			)
		)

	def _compute_marker_opacity(self):
		if self.selection is None:
			marker_opacity = 1.0
			if len(self.df) > self.target_marker_opacity:
				marker_opacity = self.target_marker_opacity / len(self.df)
			if marker_opacity < self.minimum_marker_opacity:
				marker_opacity = self.minimum_marker_opacity
			return marker_opacity, 1.0
		else:
			marker_opacity = [1.0, 1.0]
			n_selected = int(self.selection.sum())
			n_unselect = len(self.df) - n_selected
			if n_unselect > self.target_marker_opacity:
				marker_opacity[0] = self.target_marker_opacity / n_unselect
			if marker_opacity[0] < self.minimum_marker_opacity:
				marker_opacity[0] = self.minimum_marker_opacity

			if n_selected > self.target_marker_opacity:
				marker_opacity[1] = self.target_marker_opacity / n_selected
			if marker_opacity[1] < self.minimum_marker_opacity:
				marker_opacity[1] = self.minimum_marker_opacity
			return marker_opacity

	@property
	def _x_data_width(self):
		w = self._x_data_range[1] - self._x_data_range[0]
		if w > 0:
			return w
		return 1

	@property
	def _y_data_width(self):
		w = self._y_data_range[1] - self._y_data_range[0]
		if w > 0:
			return w
		return 1

	def _observe_change_column_x(self, payload):
		self.change_column_x(payload['new'])

	def change_column_x(self, col):
		with self.graph.batch_update():
			x = self.df[col]
			self.graph.layout.xaxis.title = col
			if self.selection is None:
				self.graph.data[0].x = x
				self.graph.data[3].x = None
				self.graph.data[1].x = x
				self.graph.data[4].x = None
			else:
				self.graph.data[0].x = x[~self.selection]
				self.graph.data[3].x = x[self.selection]
				self.graph.data[1].x = x
				self.graph.data[4].x = x[self.selection]
			self._x_data_range = [x.min(), x.max()]
			self.graph.layout.xaxis.range = (
				self._x_data_range[0] - self._x_data_width * 0.07,
				self._x_data_range[1] + self._x_data_width * 0.07,
			)
			self.draw_box()

	def _observe_change_column_y(self, payload):
		self.change_column_y(payload['new'])

	def change_column_y(self, col):
		with self.graph.batch_update():
			y = self.df[col]
			self.graph.layout.yaxis.title = col
			if self.selection is None:
				self.graph.data[0].y = y
				self.graph.data[3].y = None
				self.graph.data[2].y = y
				self.graph.data[5].y = None
			else:
				self.graph.data[0].y = y[~self.selection]
				self.graph.data[3].y = y[self.selection]
				self.graph.data[2].y = y
				self.graph.data[5].y = y[self.selection]
			self.draw_box()

	def change_selection(self, new_selection):
		self.selection = new_selection
		with self.graph.batch_update():
			# Update Selected Portion of Scatters
			x = self.df[self.x_axis_choose.value]
			y = self.df[self.y_axis_choose.value]
			self.graph.data[0].x = x[~self.selection]
			self.graph.data[0].y = y[~self.selection]
			self.graph.data[3].x = x[self.selection]
			self.graph.data[3].y = y[self.selection]
			marker_opacity = self._compute_marker_opacity()
			self.graph.data[0].marker.opacity = marker_opacity[0]
			self.graph.data[3].marker.opacity = marker_opacity[1]
			# Update Selected Portion of Histograms
			self.graph.data[4].x = x[self.selection]
			self.graph.data[5].y = y[self.selection]

	def draw_box(self, box=None):
		from ..scope.box import Bounds
		x_label = self.x_axis_choose.value
		y_label = self.y_axis_choose.value

		if box is None:
			box = self.box
		if box is None:
			self.graph.layout.shapes = []
		else:
			if x_label in box.thresholds or y_label in box.thresholds:
				x_lo, x_hi = None, None
				y_lo, y_hi = None, None
				if isinstance(box.thresholds.get(x_label), Bounds):
					x_lo, x_hi = box.thresholds[x_label]
				if isinstance(box.thresholds.get(y_label), Bounds):
					y_lo, y_hi = box.thresholds[y_label]
				if x_lo is None:
					x_lo = self.df[x_label].min()-self._x_data_width
				if x_hi is None:
					x_hi = self.df[x_label].max()+self._x_data_width
				if y_lo is None:
					y_lo = self.df[y_label].min()-self._y_data_width
				if y_hi is None:
					y_hi = self.df[y_label].max()+self._y_data_width

				self.graph.layout.shapes=[
					# Rectangle reference to the axes
					go.layout.Shape(
						type="rect",
						xref="x1",
						yref="y1",
						x0=x_lo,
						y0=y_lo,
						x1=x_hi,
						y1=y_hi,
						line=dict(
							width=0,
						),
						fillcolor="LightSalmon",
						opacity=0.5,
						layer="below",
					),
				]
			else:
				self.graph.layout.shapes=[]