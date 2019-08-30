


import plotly.graph_objs as go
from ipywidgets import HBox, VBox, Dropdown

class DataFrameViewer(HBox):

	def __init__(self, df):
		self.df = df
		self.x_axis_choose = Dropdown(options=self.df.columns)
		self.y_axis_choose = Dropdown(options=self.df.columns)

		self.axis_choose = VBox([
			self.x_axis_choose,
			self.y_axis_choose,
		])

		self.graph = go.FigureWidget(
			go.Scatter(
				x=df.iloc[:,0],
				y=df.iloc[:,0],
			)
		)

		super().__init__([
			self.graph,
			self.axis_choose,
		])
