
from .prototype_logging import logger
from .snippets import *

from ipywidgets import (
    VBox, HBox, Label, Layout, interactive_output, Dropdown, Output,
    Button, ToggleButtons, FloatRangeSlider, Accordion, Select, Text
)


class ExplorerComponent:

	def __init__(self, parent_widget=None):
		self._parent_widget = parent_widget
		for i in ['db','scope','box_universe', 'design_name']:
			if not hasattr(self._parent_widget, i):
				setattr(self._parent_widget, i, None)

	@property
	def db(self):
		return self._parent_widget.db

	@db.setter
	def db(self, value):
		self._parent_widget.db = value

	@property
	def scope(self):
		return self._parent_widget.scope

	@scope.setter
	def scope(self, value):
		self._parent_widget.scope = value

	@property
	def box_universe(self):
		return self._parent_widget.box_universe

	@box_universe.setter
	def box_universe(self, value):
		self._parent_widget.box_universe = value

	@property
	def design_name(self):
		return self._parent_widget.design_name

	@design_name.setter
	def design_name(self, value):
		self._parent_widget.design_name = value



class ExplorerPane(ExplorerComponent):

	def __init__(self, parent_widget=None):
		super().__init__(parent_widget)

		self.next_button = Button(
			description='Next',
			disabled=False,
			button_style='',  # 'success', 'info', 'warning', 'danger' or ''
			tooltip='Open the indicated selection in the Selectors editor',
			icon='chevron-circle-right',
			layout = Layout(flex='0 0 auto', width='auto'),
		)

		self.header = Output(
			layout=Layout(flex='1 1 0%', width='auto'),
		)

		self.header_area = HBox([
			self.header,
			self.next_button,
		])

		self.footer = Output(layout={'border': '1px solid red', })

	def make_stack(self, *body_widgets):

		self.stack = VBox([
			self.header_area,
			*body_widgets,
			self.footer,
		])

