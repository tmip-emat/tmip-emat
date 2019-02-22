
from .prototype_logging import logger
from .snippets import *

from ipywidgets import (
    VBox, HBox, Label, Layout, interactive_output, Dropdown, Output,
    Button, ToggleButtons, FloatRangeSlider, Accordion, Select, Text
)
from IPython.display import display, HTML


from .prototype_cfg import _prototype_universe
from .prototype_pane import ExplorerPane
from ..scope.box import ChainedBox, Box, pseudoname_boxes

_fancy_prototype_universe, _plain_prototype_universe = pseudoname_boxes(_prototype_universe)



class ExplorerSelectionLibrary(ExplorerPane):

	def __init__(self, parent_widget=None):
		super().__init__(parent_widget=parent_widget)

		self.box_universe = self.db.read_boxes(scope=self.scope)

		self.library_list = Select(
			options=self.box_universe.fancy_names(),
			#value=' ',
			rows=10,
			description='',
			disabled=False
		)

		self.new_child_button = Button(
			description='New Child',
			disabled=False,
			button_style='',  # 'success', 'info', 'warning', 'danger' or ''
			tooltip='Create a new child for the indicated selection',
			# icon='check'
		)
		self.new_child_button.on_click(self.open_new_child_interface)

		# self.load_box_button = Button(
		# 	description='Load '+CLUSTER,
		# 	disabled=False,
		# 	button_style='',  # 'success', 'info', 'warning', 'danger' or ''
		# 	tooltip='Open the indicated selection in the Selectors editor',
		# 	# icon='check'
		# )
		# self.load_box_button.on_click(self.load_box_action)
		self.next_button.on_click(self.load_box_action)

		self.buttons = VBox([
			self.new_child_button,
			# self.load_box_button,
		])

		self.popper = PseudoPopUpTextBox()
		self.main_window = HBox([
			self.library_list,
			self.buttons,
		])

		self.make_stack(
			self.popper,
			self.main_window,
		)

		self.library_list.observe(self.on_change_library_list)
		self.popper.disappear()

		self.on_change_library_list(None)

		with self.header:
			display(HTML("<h1>Box Selection</h1>"))



	def on_change_library_list(self, action_content):
		try:
			chained_ = self.get_current_selection_chained()
			self.footer.clear_output()
			with self.footer:
				if chained_ is None:
					print("Scenario Universe always contains all scenarios, there cannot be any thresholds.")
					self.next_button.tooltip = f"Cannot Load Feature Selection for Scenario Universe"
					self.next_button.disabled = True
				else:
					print(chained_.chain_repr())
					self.next_button.tooltip = f"Load Feature Selection for {chained_.chain_repr()}"
					self.next_button.disabled = False
		except:
			logger.exception("error on on_change_library_list")
			raise

	def show_main_window(self, z=None):
		# with self.footer:
		# 	print("show_main_window")
		self.main_window.layout.display = 'flex'
		self.main_window.layout.visibility = 'visible'

	def hide_main_window(self, z=None):
		# with self.footer:
		# 	print("hide_main_window")
		self.main_window.layout.visibility = 'hidden'
		self.main_window.layout.display = 'none'

	def open_new_child_interface(self, z):
		# with self.footer:
		# 	print("open_new_child_interface")
		# 	print(z)
		self.hide_main_window()
		self.popper.appear()
		self.popper.ok_button.on_click(self.close_new_child_interface)
		chained_ = self.get_current_selection_chained()
		self.footer.clear_output()
		with self.footer:
			print("Name a new child for:")
			print(chained_.names[0])
			for n,c in enumerate(chained_.names[1:]):
				print(" "*(n+1) + f"┗━ {c}")

	def close_new_child_interface(self, z):
		with self.footer:
			print("close_new_child_interface")
			print(z)
		self.popper.disappear()
		self.popper.ok_button.on_click(self.close_new_child_interface, remove=True)
		self.show_main_window()
		new_child_name = self.popper.text_entry.value
		plain_ = self.get_current_selection_plain()
		if new_child_name in self._parent_widget.box_universe:
			# ERROR
			logger.warn(f'A box with the name "{new_child_name}" already exists')
			self.footer.clear_output()
			with self.footer:
				print(f'ERROR: A box with the name "{new_child_name}" already exists')
		else:
			self._parent_widget.box_universe[new_child_name] = Box(plain_)
		logger.info("New Names = "+str(self._parent_widget.box_universe.fancy_names()))
		self.library_list.options = self._parent_widget.box_universe.fancy_names()



	def load_box_action(self, z=None):
		logger.info("load_box_action")
		plain_ = self.get_current_selection_plain()
		if plain_ is None:
			logger.warn(f"cannot load *Scenario Universe*, create a new child or choose another name")
			self.footer.clear_output()
			with self.footer:
				print("cannot load *Scenario Universe*, create a new child or choose another name")
		else:
			logger.info(f"  load_box_action:{plain_}")
			try:
				self._parent_widget.load_feature_selection(plain_)
			except Exception as err:
				logger.info(f"error {err}")
				with self.footer:
					print(err)
				raise

	def get_current_selection_plain(self):
		return self._parent_widget.box_universe.plain_names()[self.library_list.index]

	def get_current_selection_chained(self):
		plain_ = self.get_current_selection_plain()
		if plain_ is None:
			return None
		else:
			return ChainedBox(self._parent_widget.box_universe, plain_)


class PseudoPopUpTextBox(HBox):

	def __init__(self):
		self.text_entry = Text()
		self.ok_button = Button(
			description='OK',
			disabled=False,
			button_style='',  # 'success', 'info', 'warning', 'danger' or ''
			tooltip='OK',
			icon='check'
		)
		super().__init__([self.text_entry, self.ok_button])

	def appear(self):
		self.layout.visibility = 'visible'
		self.layout.display = 'flex'

	def disappear(self):
		self.layout.visibility = 'hidden'
		self.layout.display = 'none'