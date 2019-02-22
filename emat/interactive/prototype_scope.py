


import os
import sqlite3
from ipywidgets import (
    VBox, HBox, Label, Layout, interactive_output, Dropdown, Output,
    Button, ToggleButtons, FloatRangeSlider, Accordion, Text
)
from .prototype_pane import ExplorerPane
from IPython.display import display, HTML

from ..database import Database

import __main__

class ExplorerScopeManager(ExplorerPane):

	def __init__(self, parent_widget=None, proposed_db_name='db'):
		super().__init__(parent_widget=parent_widget)

		if isinstance(proposed_db_name, Database):
			self.filenamer = Text(
				value=proposed_db_name.get_db_info(),
				placeholder='db info',
				description='DB:',
				disabled=True
			)
			self.db = proposed_db_name
		else:
			self.filenamer = Text(
				value=proposed_db_name,
				placeholder='enter object name',
				description='DB:',
				disabled=False
			)

		self.scope_picker = Dropdown(
			options=[
				' ',
			],
			value=' ',
			description='Scope:',
			disabled=False,
		)

		self.design_picker = Dropdown(
			options=[
				' ',
			],
			value=' ',
			description='Design:',
			disabled=False,
		)

		self.filenamer.observe(self.check_db_status)
		self.scope_picker.observe(self.update_current_scope_from_db)
		self.design_picker.observe(self.update_current_design_from_db)

		self.make_stack(
			self.filenamer,
			self.scope_picker,
			self.design_picker,
		)

		self.next_button.on_click(lambda x: self._parent_widget.load_selection_library())
		with self.header:
			display(HTML("<h1>Scope Manager</h1>"))

		self.check_db_status(None)


	def check_db_status(self, action_content):
		main_object_name = self.filenamer.value
		self.footer.clear_output()
		with self.footer:
			if main_object_name in __main__.__dict__:
				db = getattr(__main__, main_object_name)
				if isinstance(db, Database):
					print(f"√ '{main_object_name}' is Database")
					self.db = db
				else:
					print(f"X '{main_object_name}' not Database")
					self.db = None
			elif self.db:
				print(f"√ {self.db.get_db_info()}")
			else:
				print(f"X '{main_object_name}' not a known object")
				self.db = None

			self._update_scope_name_choices()
			self._update_design_name_choices()
			self.update_current_scope_from_db(action_content)
			if self.scope is not None:
				print(f"√ Loaded Scope: {self.scope.name}")
			else:
				print(f"X No Scope")


	def _update_scope_name_choices(self):
		if self.db is None:
			self.scope_picker.options = [' ']
		else:
			scope_names = self.db.read_scope_names()
			if self.scope_picker.options != scope_names:
				self.scope_picker.options = scope_names
				if len(scope_names) == 1:
					self.update_current_scope_from_db(None)

	def _update_design_name_choices(self):
		if self.scope is None:
			self.design_picker.options = [' ']
		else:
			design_names = self.db.read_design_names(self.scope.name)
			if self.design_picker.options != design_names:
				self.design_picker.options = design_names
				if len(design_names)==1:
					self.update_current_design_from_db(None)

	def update_current_scope_from_db(self, action_content):
		scope_name = self.scope_picker.value
		if self.db is not None:
			if self.scope is not None:
				if self.scope.name == scope_name:
					return
			self.scope = self.db.read_scope(scope_name)
		else:
			self.scope = None

	def update_current_design_from_db(self, action_content):
		self.design_name = self.design_picker.value

	def check_file_status(self, action_content):
		f = self.filenamer.value
		self.footer.clear_output()
		with self.footer:
			if os.path.exists(f):
				print("√ File Exists")
				fa = os.path.abspath(f)
				checker = None
				try:
					checker = sqlite3.connect(
						f'file:{f}?mode=ro',
						uri=True
					)
				except Exception as err:
					print("X ERROR")
					print(str(err))
				else:

					try:
						checker.cursor().execute("SELECT strategyDesc FROM strategy LIMIT 1")
					except Exception as err:
						print("X ERROR")
						print(str(err))
					else:
						print("√ File has 'strategy' table")

					try:
						checker.cursor().execute("SELECT riskVarDesc FROM riskVar LIMIT 1")
					except Exception as err:
						print("X ERROR")
						print(str(err))
					else:
						print("√ File has 'riskVar' table")

					try:
						checker.cursor().execute("SELECT perfMeasDesc FROM perfMeas LIMIT 1")
					except Exception as err:
						print("X ERROR")
						print(str(err))
					else:
						print("√ File has 'perfMeas' table")

					try:
						checker.cursor().execute("SELECT perfMeasID FROM perfMeasRes LIMIT 1")
					except Exception as err:
						print("X ERROR")
						print(str(err))
					else:
						print("√ File has 'perfMeasRes' table")

					try:
						checker.cursor().execute("SELECT sampleID, varID, value FROM perfMeasInput LIMIT 1")
					except Exception as err:
						print("X ERROR")
						print(str(err))
					else:
						print("√ File has 'perfMeasInput' table")
				finally:
					if checker is not None:
						checker.close()

			else:
				print("X File Does Not Exist")
