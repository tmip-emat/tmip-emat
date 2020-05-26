# -*- coding: utf-8 -*-

from typing import List, Union, Mapping, Dict, Tuple, Callable
import yaml
import os, sys, time
from shutil import copyfile, copy
import glob
import numpy as np
import pandas as pd
from ...model.core_model import AbstractCoreModel
from ...scope.scope import Scope
from ...database.database import Database
from ...util.loggers import get_module_logger
from ...util.docstrings import copydoc
from ...exceptions import *
from .parsers import *

_logger = get_module_logger(__name__)



def copy_model_outputs_1(
		local_model,
		remote_repository,
		file
):
	copyfile(
		os.path.join(local_model, "Outputs", file),
		os.path.join(remote_repository, "Outputs", file)
	)

def copy_model_outputs_ext(
		local_model,
		remote_repository,
		basename,
		ext=('.bin', '.dcb')
):
	for x in ext:
		copy_model_outputs_1(
			local_model,
			remote_repository,
			os.path.splitext(basename)[0] + x
		)

ALL = slice(None)


class FilesCoreModel(AbstractCoreModel):
	"""
	Setup connections and paths to a file reading core model

	Args:
		configuration:
			The configuration for this
			core model. This can be passed as a dict, or as a str
			which gives the filename of a YAML file that will be
			loaded.
		scope:
			The exploration scope, as a Scope object or as
			a str which gives the filename of a YAML file that will be
			loaded.
		safe:
			Load the configuration YAML file in 'safe' mode.
			This can be disabled if the configuration requires
			custom Python types or is otherwise not compatible with
			safe mode. Loading configuration files with safe mode
			off is not secure and should not be done with files from
			untrusted sources.
		db:
			An optional Database to store experiments and results.
		name:
			A name for this model, given as an alphanumeric string.
			The name is required by ema_workbench operations.
			If not given, "FilesCoreModel" is used.

	"""

	def __init__(self,
				 configuration: Union[str, Mapping],
				 scope: Union[Scope, str],
				 safe: bool = True,
				 db: Database = None,
				 name: str = 'FilesCoreModel',
				 ):
		super().__init__(
			configuration=configuration,
			scope=scope,
			safe=safe,
			db=db,
			name=name,
			metamodel_id=0,
		)

		self.model_path = self.config['model_path']
		"""Path: The directory of the 'live' model instance."""

		self.rel_output_path = self.config.get('rel_output_path', 'Outputs')
		"""Path: The path to 'live' model outputs, relative to `model_path`."""

		self.archive_path = self.config['model_archive']
		"""Path: The directory where archived models are stored."""

		self.allow_short_circuit = self.config.get('allow_short_circuit', True)
		"""Bool: Allow model runs to be skipped if measures already appear in the database."""

		self._parsers = []

	def add_parser(self, parser):
		"""
		Add a FileParser to extract performance measures.

		Args:
			parser (FileParser): The parser to add.

		"""
		if not isinstance(parser, FileParser):
			raise TypeError("parser must be an instance of FileParser")
		self._parsers.append(parser)

	def model_init(self, policy):
		super().model_init(policy)

	def run_model(self, scenario, policy):
		"""
		Runs an experiment through core model.

		This method overloads the `run_model` method given in
		the EMA Workbench, and provides the correct execution
		of the GBNRTC model within that framework.

		For each experiment, the core model is called to:

			1. set experiment variables
			2. run the experiment
			3. run post-processors associated with specified
			   performance measures
			4. (optionally) archive model outputs
			5. record performance measures to database

		Note that this method does *not* return any outcomes.
		Outcomes are instead written into self.outcomes_output,
		and can be retrieved from there.

		Args:
			scenario (Scenario): A dict-like object that
				has key-value pairs for each uncertainty.
			policy (Policy): A dict-like object that
				has key-value pairs for each lever.

		Raises:
			UserWarning: If there are no experiments associated with
				this type.

		"""

		_logger.debug("run_core_model read_experiment_parameters")

		experiment_id = self.db.read_experiment_id(self.scope.name, None, scenario, policy)

		if experiment_id is not None and self.allow_short_circuit:
			# opportunity to short-circuit run by loading pre-computed values.
			precomputed = self.db.read_experiment_measures(
				self.scope.name,
				design=None,
				experiment_id=experiment_id,
			)
			if not precomputed.empty:
				self.outcomes_output = dict(precomputed.iloc[0])
				return

		if experiment_id is None:
			experiment_id = self.db.write_experiment_parameters_1(
				self.scope.name, 'ad hoc', scenario, policy
			)

		xl = {}
		xl.update(scenario)
		xl.update(policy)

		m_names = self.scope.get_measure_names()

		m_out = pd.DataFrame()

		_logger.debug(f"run_core_model setup {experiment_id}")
		self.setup(xl)

		_logger.debug(f"run_core_model run {experiment_id}")
		self.run()

		_logger.debug(f"run_core_model post_process {experiment_id}")
		self.post_process(xl, m_names)

		_logger.debug(f"run_core_model wrap up {experiment_id}")
		measures_dictionary = self.load_measures(m_names)
		m_df = pd.DataFrame(measures_dictionary, index=[experiment_id])

		# Assign to outcomes_output, for ema_workbench compatibility
		self.outcomes_output = measures_dictionary

		_logger.debug(f"run_core_model write db {experiment_id}")
		self.db.write_experiment_measures(self.scope.name, self.metamodel_id, m_df)

		try:
			archive_path = self.get_experiment_archive_path(experiment_id)
		except MissingArchivePathError:
			pass
		else:
			_logger.debug(f"run_core_model archive {experiment_id}")
			self.archive(xl, archive_path, experiment_id)

	@copydoc(AbstractCoreModel.get_experiment_archive_path)
	def get_experiment_archive_path(self, experiment_id: int, makedirs:bool=False) -> str:
		if self.archive_path is None:
			raise MissingArchivePathError('no archive set for this core model')
		mod_results_path = os.path.join(
			self.archive_path,
			f"scp_{self.scope.name}",
			f"exp_{experiment_id}"
		)
		if makedirs:
			os.makedirs(mod_results_path, exist_ok=True)
		return mod_results_path

	def setup(self, params: dict):
		# TODO: Make directory structure.  Subclass will fill it.
		raise NotImplementedError()

	@copydoc(AbstractCoreModel.load_measures)
	def load_measures(
			self,
			measure_names: List[str],
			*,
			rel_output_path=None,
			abs_output_path=None,
	):

		if rel_output_path is not None and abs_output_path is not None:
			raise ValueError("cannot give both `rel_output_path` and `abs_output_path`")
		elif rel_output_path is None and abs_output_path is None:
			output_path = os.path.join(self.model_path, self.rel_output_path)
		elif rel_output_path is not None:
			output_path = os.path.join(self.model_path, rel_output_path)
		else: # abs_output_path is not None
			output_path = abs_output_path

		if not os.path.isdir(output_path):
			raise NotADirectoryError(output_path)

		if measure_names is None:
			is_requested = lambda i: True
		else:
			requested_measure_names = set(measure_names)
			is_requested = lambda i: i in requested_measure_names

		results = {}

		for parser in self._parsers:
			if any(is_requested(name) for name in parser.measure_names):
				try:
					measures = parser.read(output_path)
				except FileNotFoundError as err:
					import warnings
					for name in parser.measure_names:
						if is_requested(name):
							warnings.warn(f'{name} unavailable, {err} not found')
				except Exception as err:
					import warnings
					for name in parser.measure_names:
						if is_requested(name):
							warnings.warn(f'{name} unavailable, {err!r}')
				else:
					for k, v in measures.items():
						if is_requested(k):
							results[k] = v

		return results

	def load_archived_measures(self, experiment_id, measure_names=None):
		"""
		Load performance measures from an archived model run.

		Args:
			experiment_id (int): The id for the experiment to load.
			measure_names (Collection, optional): A subset of
				performance measure names to load.  If not provided,
				all measures will be loaded.
		"""
		return self.load_measures(
			measure_names,
			abs_output_path=os.path.join(
				self.get_experiment_archive_path(experiment_id),
				self.rel_output_path,
			)
		)

	def archive(self, params, model_results_path, experiment_id:int=0):
		raise NotImplementedError

	def run(self):
		raise NotImplementedError

	def post_process(self, params, measure_names, output_path=None):
		raise NotImplementedError

