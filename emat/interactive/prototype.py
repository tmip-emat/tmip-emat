from ipywidgets import (
    VBox, HBox, Label, Layout, interactive_output, Dropdown, Output,
    Button, ToggleButtons, FloatRangeSlider, Accordion, Tab
)

from .prototype_data import ExplorerData
from .prototype_thresholds import ExplorerThresholds
from .prototype_report import ExplorerReports
from .prototype_scope import ExplorerScopeManager
from .prototype_selection_library import ExplorerSelectionLibrary
from .prototype_cfg import _prototype_universe
from .prototype_logging import logger
from .prototype_feature_select import ExplorerFeatureSelect

from .snippets import *

from ..database import SQLiteDB
from ..scope.box import Boxes

class Explorer(Tab):

	def __init__(self, db="name_of_db_object"):

		super().__init__()

		self.db = None
		self.scope = None
		self.box_universe = Boxes()

		self.explorer_data = ExplorerData(parent=self)

		self.scope_mgr = ExplorerScopeManager(parent_widget=self, proposed_db_name=db)
		self.library = None
		# self.library = ExplorerSelectionLibrary(parent_widget=self)
		# self.selector = ExplorerSelector(self.explorer_data, cluster={})
		# self.reporter = ExplorerReports(self.selector)

		self.children = [
			self.scope_mgr.stack,
			# self.library.stack,
			# self.selector.accordion,
			# self.reporter.stack,
		]

		self.set_title(0, "Scope Manager")
		# self.set_title(1, f"{CLUSTER} Library")
		# self.set_title(2, "Selectors")
		# self.set_title(3, "Reports")

	def load_selection_library(self):
		logger.info("load_selection_library")
		try:

			if self.library is None:
				self.library = ExplorerSelectionLibrary(parent_widget=self)
				self.children += (self.library.stack,)
				self.set_title(len(self.children)-1, f"Box Selection")
			#self.relevant_attrib_pane.reset_sliders(self.box_universe, box_name=box_name)

			self.selected_index = 1

		except:
			logger.exception("error in load_selection_library")
			raise



	def load_feature_selection(self, box_name):

		logger.info("load_feature_selection")
		try:

			if len(self.children) < 3:
				self.relevant_attrib_pane = ExplorerFeatureSelect(
					self.explorer_data,
					self.box_universe,
					box_name=box_name,
					parent_widget=self,
				)
				self.children += (self.relevant_attrib_pane.stack,)
				self.set_title(2, RELEVANT)
			#self.relevant_attrib_pane.reset_sliders(self.box_universe, box_name=box_name)

			self.selected_index = 2

		except:
			logger.exception("error in load_cluster")
			raise

	def load_cluster(self, box_name, design_name=None):

		logger.info("load_cluster")
		try:
			if design_name is None:
				design_name = self.design_name

			chain = self.box_universe.get_chain(box_name)

			relevant_features = chain.relevant_features
			demanded_features = chain.demanded_features

			relevant_features |= demanded_features

			self.explorer_data.load_from_database(
				feature_names=relevant_features,
				design_name=design_name,
			)

			if len(self.children) < 4:
				self.thresholds_pane = ExplorerThresholds(
					self.explorer_data,
					self.box_universe,
					box_name=box_name,
					parent_widget=self,
				)
				self.children += (self.thresholds_pane.stack,)
				self.set_title(3, THRESHOLDS)
			self.thresholds_pane.reset_sliders(self.box_universe, box_name=box_name)

			self.selected_index = 3

		except:
			logger.exception("error in load_cluster")
			raise

	def load_reports(self):

		logger.debug("load_reports")
		try:

			if len(self.children) < 5:
				self.reporter = ExplorerReports(self.thresholds_pane, parent_widget=self)
				self.children += (self.reporter.stack,)
				self.set_title(4, REPORTS)

			self.selected_index = 4

		except:
			logger.exception("error in load_reports")
			raise

