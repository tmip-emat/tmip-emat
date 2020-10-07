# -*- coding: utf-8 -*-

from typing import List, Union, Mapping, Dict, Tuple, Callable
import yaml
import os, sys, time
import shutil
import glob
import numpy as np
import pandas as pd
import subprocess
import warnings
import logging
from pathlib import Path
from ...model.core_model import AbstractCoreModel
from ...scope.scope import Scope
from ...database.database import Database
from ...util.docstrings import copydoc
from ...exceptions import *
from .parsers import *
from ...util.loggers import get_module_logger

_logger = get_module_logger(__name__)



def copy_model_outputs_1(
		local_model,
		remote_repository,
		file
):
	shutil.copyfile(
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
		local_directory:
			Optionally explicitly give this local_directory to use,
			overriding any directory set in the config file. If not
			given either here or in the config file, then Python's
			cwd is used.


	"""

	def __init__(self,
				 configuration: Union[str, Mapping],
				 scope: Union[Scope, str],
				 safe: bool = True,
				 db: Database = None,
				 name: str = 'FilesCoreModel',
				 local_directory: Path = None,
				 ):
		super().__init__(
			configuration=configuration,
			scope=scope,
			safe=safe,
			db=db,
			name=name,
			metamodel_id=0,
		)

		self.local_directory = local_directory or self.config.get("local_directory") or os.getcwd()
		"""Path: The current local working directory for this model."""

		self.model_path = os.path.expanduser(self.config['model_path'])
		"""Path: The directory of the 'live' model instance, relative to the local_directory."""

		self.rel_output_path = self.config.get('rel_output_path', 'Outputs')
		"""Path: The path to 'live' model outputs, relative to `model_path`."""

		self.archive_path = os.path.expanduser(self.config['model_archive'])
		"""Path: The directory where archived models are stored."""

		self.allow_short_circuit = self.config.get('allow_short_circuit', True)
		"""Bool: Allow model runs to be skipped if measures already appear in the database."""

		self.ignore_crash = self.config.get('ignore_crash', False)
		"""Bool: Allow model runs to continue to `post_process` and `archive` even after an apparent crash in `run`."""

		self.success_indicator = self.config.get('success_indicator', None)
		"""str: optional, The name of a file that indicates the model has run successfully.  
		
		This file is deleted automatically when the model `run` is initiated."""

		self.killed_indicator = self.config.get('killed_indicator', None)
		"""str: optional, The name of a file that indicates the model was killed due to an unrecoverable error.  

		This file is deleted automatically when the model `run` is initiated."""

		self._parsers = []

	def __getstate__(self):
		state = super().__getstate__()
		# The SQLite Database does not serialize for usage in other
		# threads or processes, so we will pass through the database path
		# to open another new connection on the other end, assuming it is
		# a file object that can be re-opened for other connections.
		db = getattr(self, 'db', None)
		from ...database import SQLiteDB
		if isinstance(db, SQLiteDB):
			if os.path.exists(db.database_path):
				state['_sqlitedb_path_'] = db.database_path
				state['_sqlitedb_readonly_'] = db.readonly
		return state

	def __setstate__(self, state):
		# When we are running on a dask worker, functions
		# are executed in a different thread from the worker
		# itself, even if there is only one thread.  To prevent
		# problems with SQLite, we check if this is a worker and
		# if there is only one thread, in which case we can
		# safely ignore the fact that the database is accessed
		# from a different thread than where it is created.
		from dask.distributed import get_worker
		try:
			worker = get_worker()
		except ValueError:
			n_threads = -1
		else:
			n_threads = worker.nthreads
		database_path = state.pop('_sqlitedb_path_', None)
		database_readonly = state.pop('_sqlitedb_readonly_', False)
		self.__dict__ = state
		if database_path and not database_readonly:
			from ...database import SQLiteDB
			if os.path.exists(database_path):
				self.db = SQLiteDB(
					database_path,
					initialize='skip',
					readonly=database_readonly,
					check_same_thread=(n_threads!=1),
				)

	@property
	def resolved_model_path(self):
		"""
		The model path to use.

		If `model_path` is set to an absolute path, then that path is returned,
		otherwise the `model_path` is joined onto the `local_directory`.

		Returns:
			str
		"""
		if self.model_path is None:
			raise MissingModelPathError('no archive set for this core model')
		if os.path.isabs(self.model_path):
			return self.model_path
		else:
			return os.path.join(self.local_directory, self.model_path)

	@property
	def resolved_archive_path(self):
		"""
		The archive path to use.

		If `archive_path` is set to an absolute path, then that path is returned,
		otherwise the `archive_path` is joined onto the `local_directory`.

		Returns:
			str
		"""
		if self.archive_path is None:
			raise MissingArchivePathError('no archive set for this core model')
		if os.path.isabs(self.archive_path):
			return self.archive_path
		else:
			return os.path.join(self.local_directory, self.archive_path)

	def add_parser(self, parser):
		"""
		Add a FileParser to extract performance measures.

		Args:
			parser (FileParser): The parser to add.

		"""
		if not isinstance(parser, FileParser):
			raise TypeError("parser must be an instance of FileParser")
		self._parsers.append(parser)

	def get_parser(self, idx):
		"""
		Access a FileParser, used to extract performance measures.

		Args:
			idx (int): The position of the parser to get.

		Returns:
			FileParser
		"""
		return self._parsers[idx]

	def model_init(self, policy):
		super().model_init(policy)

	def enter_run_model(self):
		"""A hook for actions at the very beginning of the run_model step."""
		print("enter_run_model")

	def exit_run_model(self):
		"""A hook for actions at the very end of the run_model step."""

	def run_model(self, scenario, policy):
		"""
		Runs an experiment through core model.

		This method overloads the `run_model` method given in
		the EMA Workbench, and provides the correct execution
		of a core model within the workbench framework.  This
		function assembles and executes the steps laid out in
		other methods of this class, adding some useful logic
		to optimize the process (e.g. optionally short-
		circuiting runs that already have results stored
		in the database).

		For each experiment, the core model is called to:

			1.  `setup` experiment variables, copy files
			    as needed, and otherwise prepare to run the
			    core model for a particular experiment,
			2.  `run` the experiment,
			3.  `post_process` the result if needed to
			    produce all relevant performance measures,
			4.  `archive` model outputs from this experiment
			    (optional), and
			5.  `load_measures` from the experiment and
			    store those measures in the associated database.

		Note that this method does *not* return any outcomes.
		Outcomes are instead written into self.outcomes_output,
		and can be retrieved from there, or from the database at
		a later time.

		In general, it should not be necessary to overload this
		method in derived classes built for particular core models.
		Instead, write overloaded methods for `setup`, `run`,
		`post_process` , `archive`, and `load_measures`.  Moreover,
		in typical usage a modeler will generally not want to rely
		on this method directly, but instead use `run_experiments`
		to automatically run multiple experiments with one command.

		Args:
			scenario (Scenario): A dict-like object that
				has key-value pairs for each uncertainty.
			policy (Policy): A dict-like object that
				has key-value pairs for each lever.

		Raises:
			UserWarning: If there are no experiments associated with
				this type.

		"""
		self.enter_run_model()
		try:
			self.comment_on_run = None

			_logger.debug("run_core_model read_experiment_parameters")

			experiment_id = policy.get("_experiment_id_", None)
			if experiment_id is None:
				experiment_id = scenario.get("_experiment_id_", None)

			if not hasattr(self, 'db') and hasattr(self, '_db'):
				self.db = self._db

			# If running a core files model using the DistributedEvaluator,
			# the workers won't have access to the DB directly, so we'll only
			# run the short-circuit test and the ad-hoc write-to-database
			# section of this code if the `db` attribute is available.
			if hasattr(self, 'db') and self.db is not None:

				assert isinstance(self.db, Database)

				if experiment_id is None:
					experiment_id = self.db.read_experiment_id(self.scope.name, scenario, policy)

				if experiment_id is not None and self.allow_short_circuit:
					# opportunity to short-circuit run by loading pre-computed values.
					precomputed = self.db.read_experiment_measures(
						self.scope.name,
						design_name=None,
						experiment_id=experiment_id,
					)
					if not precomputed.empty:
						self.outcomes_output = dict(precomputed.iloc[0])
						self.log(f"short circuit experiment_id {experiment_id} / {getattr(self, 'uid', 'no uid')}")
						return

				if experiment_id is None:
					experiment_id = self.db.write_experiment_parameters_1(
						self.scope.name, 'ad hoc', scenario, policy
					)
				self.log(f"YES DATABASE experiment_id {experiment_id}", level=logging.CRITICAL)

			else:
				_logger.critical(f"NO DATABASE experiment_id {experiment_id}")

			xl = {}
			xl.update(scenario)
			xl.update(policy)

			m_names = self.scope.get_measure_names()

			_logger.debug(f"run_core_model setup {experiment_id}")
			self.setup(xl)

			if self.success_indicator is not None:
				success_indicator = os.path.join(self.resolved_model_path, self.success_indicator)
				if os.path.exists(success_indicator):
					os.remove(success_indicator)
			else:
				success_indicator = None

			if self.killed_indicator is not None:
				killed_indicator = os.path.join(self.resolved_model_path, self.killed_indicator)
				if os.path.exists(killed_indicator):
					os.remove(killed_indicator)
			else:
				killed_indicator = None

			_logger.debug(f"run_core_model run {experiment_id}")
			try:
				self.run()
			except subprocess.CalledProcessError as err:
				_logger.error(f"ERROR in run_core_model run {experiment_id}: {str(err)}")
				try:
					ex_archive_path = self.get_experiment_archive_path(experiment_id, makedirs=True)
				except MissingArchivePathError:
					pass
				else:
					if err.stdout:
						with open(os.path.join(ex_archive_path, 'error.stdout.log'), 'ab') as stdout:
							stdout.write(err.stdout)
					if err.stderr:
						with open(os.path.join(ex_archive_path, 'error.stderr.log'), 'ab') as stderr:
							stderr.write(err.stderr)
					with open(os.path.join(ex_archive_path, 'error.log'), 'a') as errlog:
						errlog.write(str(err))
				measures_dictionary = {name:np.nan for name in m_names}
				# Assign to outcomes_output, for ema_workbench compatibility
				self.outcomes_output = measures_dictionary

				if not self.ignore_crash:
					# If 'ignore_crash' is False (the default), then abort now and skip
					# any post-processing and other archiving steps, which will
					# probably fail anyway.
					self.log(f"run_core_model ABORT {experiment_id}", level=logging.ERROR)
					self.comment_on_run = f"FAILED EXPERIMENT {experiment_id}: {str(err)}"
					return
				else:
					_logger.error(f"run_core_model CONTINUE AFTER ERROR {experiment_id}")

			try:
				if success_indicator and not os.path.exists(success_indicator):
					# The absence of the `success_indicator` file means that the model
					# did not actually terminate correctly, so we do not want to
					# post-process or store these results in the database.
					self.comment_on_run = f"NON-SUCCESSFUL EXPERIMENT {experiment_id}: success_indicator missing"
					raise ValueError(f"success_indicator missing: {success_indicator}")

				if killed_indicator and os.path.exists(killed_indicator):
					self.comment_on_run = f"KILLED EXPERIMENT {experiment_id}: killed_indicator present"
					raise ValueError(f"killed_indicator present: {killed_indicator}")

				_logger.debug(f"run_core_model post_process {experiment_id}")
				self.post_process(xl, m_names)

				_logger.debug(f"run_core_model wrap up {experiment_id}")
				measures_dictionary = self.load_measures(m_names)
				m_df = pd.DataFrame(measures_dictionary, index=[experiment_id])

				# Assign to outcomes_output instead of returning them, for ema_workbench compatibility
				self.outcomes_output = measures_dictionary
			except KeyboardInterrupt:
				_logger.exception(f"KeyboardInterrupt in post_process, load_measures or outcome processing {experiment_id}")
				raise
			except Exception as err:
				_logger.exception(f"error in post_process, load_measures or outcome processing {experiment_id}")
				_logger.error(f"proceeding directly to archive attempt {experiment_id}")
				if not self.comment_on_run:
					self.comment_on_run = f"PROBLEM IN EXPERIMENT {experiment_id}: {str(err)}"
			else:
				# only write to database if there was no error in post_process, load_measures or outcome processing
				_logger.debug(f"run_core_model write db {experiment_id}")
				if hasattr(self, 'db') and self.db is not None:
					run_id = getattr(self, 'run_id', None)
					try:
						self.db.write_experiment_measures(self.scope.name, self.metamodel_id, m_df, [run_id])
					except Exception as err:
						_logger.exception(f"error in writing results to database: {str(err)}")

			try:
				ex_archive_path = self.get_experiment_archive_path(experiment_id)
			except MissingArchivePathError:
				pass
			else:
				_logger.debug(f"run_core_model archive {experiment_id}")
				self.archive(xl, ex_archive_path, experiment_id)
		finally:
			self.exit_run_model()

	def get_experiment_archive_path(
			self,
			experiment_id=None,
			makedirs=False,
			parameters=None,
			run_id=None,
	):
		"""
		Returns a file system location to store model run outputs.

		For core models with long model run times, it is recommended
		to store the complete model run results in an archive.  This
		will facilitate adding additional performance measures to the
		scope at a later time.

		Both the scope name and experiment id can be used to create the
		folder path.

		Args:
		    experiment_id (int):
		        The experiment id, which is also the row id of the
		        experiment in the database. If this is omitted, an
		        experiment id is read or created using the parameters.
		    makedirs (bool, default False):
		        If this archive directory does not yet exist, create it.
		    parameters (dict, optional):
		        The parameters for this experiment, used to create or
		        lookup an experiment id. The parameters are ignored
		        if `experiment_id` is given.
		    run_id (UUID, optional):
		        The run_id of this model run.  If not given but a
		        run_id attribute is stored in this FilesCoreModel
		        instance, that value is used.

		Returns:
		    str: Experiment archive path (no trailing backslashes).
		"""
		if experiment_id is None:
			if parameters is None:
				raise ValueError("must give `experiment_id` or `parameters`")
			db = getattr(self, 'db', None)
			if db is not None:
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=MissingIdWarning)
					experiment_id = db.get_experiment_id(self.scope.name, parameters)

		if run_id is None:
			run_id = getattr(self, 'run_id', None)

		try:
			exp_dir_name = f"exp_{experiment_id:03d}"
		except ValueError:
			exp_dir_name = f"exp_{experiment_id}"
		if run_id is not None:
			exp_dir_name += f"_{run_id}"

		mod_results_path = os.path.join(
			self.resolved_archive_path,
			f"scp_{self.scope.name}",
			exp_dir_name,
		)
		if makedirs:
			os.makedirs(mod_results_path, exist_ok=True)
		return mod_results_path

	def setup(self, params):
		"""
		Configure the core model with the experiment variable values.

		This method is the place where the core model set up takes place,
		including creating or modifying files as necessary to prepare
		for a core model run.  When running experiments, this method
		is called once for each core model experiment, where each experiment
		is defined by a set of particular values for both the exogenous
		uncertainties and the policy levers.  These values are passed to
		the experiment only here, and not in the `run` method itself.
		This facilitates debugging, as the `setup` method can potentially
		be used without the `run` method, allowing the user to manually
		inspect the prepared files and ensure they are correct before
		actually running a potentially expensive model.

		Each input exogenous uncertainty or policy lever can potentially
		be used to manipulate multiple different aspects of the underlying
		core model.  For example, a policy lever that includes a number of
		discrete future network "build" options might trigger the replacement
		of multiple related network definition files.  Or, a single uncertainty
		relating to the cost of fuel might scale both a parameter linked to
		the modeled per-mile cost of operating an automobile, as well as the
		modeled total cost of fuel used by transit services.

		At the end of the `setup` method, a core model experiment should be
		ready to run using the `run` method.

		Classes derived from `FilesCoreModel` do not necessarily need to
		call `super().setup(params)`, but may find it convenient to do so,
		as this implementation provides some standard functionality,
		including validation of parameter names, managing existing
		archive directories, and logging the start time to the archive.

		Args:
			params (dict):
				experiment variables including both exogenous
				uncertainty and policy levers

		Raises:
			KeyError:
				if a defined experiment variable is not supported
				by the core model
		"""

		# Validate parameter names
		scope_param_names = set(self.scope.get_parameter_names())
		for key in params.keys():
			if key not in scope_param_names:
				self.log(
					f"SETUP ERROR: '{key}' not found in scope parameters",
					level=logging.ERROR,
				)
				raise KeyError(f"'{key}' not found in scope parameters")

		# Get the experiment_id if stored
		db = getattr(self, 'db', None)
		if db is not None:
			run_id, experiment_id = self.db.new_run_id(self.scope.name, params)
		else:
			import uuid
			experiment_id = getattr(self, 'experiment_id', None)
			run_id = uuid.uuid4()

		self.run_id = run_id
		self.experiment_id = experiment_id

		# Rename any existing archive directories
		if experiment_id is not None:
			orig_archive = self.get_experiment_archive_path(experiment_id)
			if os.path.exists(orig_archive):
				n = 1
				orig_archive = orig_archive.rstrip(os.path.sep)
				dirpath, basepath = os.path.split(orig_archive)
				new_archive = os.path.normpath(os.path.join(dirpath, f"{basepath}_OLD_{n}"))
				while os.path.exists(new_archive):
					n += 1
					new_archive = os.path.normpath(os.path.join(dirpath, f"{basepath}_OLD_{n}"))
				shutil.move(orig_archive, new_archive)
			os.makedirs(orig_archive, exist_ok=True)
			with open(os.path.join(orig_archive, "_emat_start_.log"), 'at') as f:
				f.write("Starting model run at {0}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
				if run_id is not None:
					f.write(f"run_id = {run_id}\n")

	@copydoc(AbstractCoreModel.load_measures)
	def load_measures(
			self,
			measure_names: List[str]=None,
			*,
			rel_output_path=None,
			abs_output_path=None,
	):

		if rel_output_path is not None and abs_output_path is not None:
			raise ValueError("cannot give both `rel_output_path` and `abs_output_path`")
		elif rel_output_path is None and abs_output_path is None:
			output_path = os.path.join(self.resolved_model_path, self.rel_output_path)
		elif rel_output_path is not None:
			output_path = os.path.join(self.resolved_model_path, rel_output_path)
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
					for name in parser.measure_names:
						if is_requested(name):
							warnings.warn(f'{name} unavailable, {err} not found')
				except Exception as err:
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
		experiment_archive_path = self.get_experiment_archive_path(experiment_id)
		experiment_archive_zip = experiment_archive_path.rstrip("/\\")+".zip"
		if os.path.exists(experiment_archive_zip):
			_logger.info(f"zipped archive found, loading from {experiment_archive_zip}")
			import tempfile, zipfile
			with tempfile.TemporaryDirectory() as tmpdir:
				zipfile.ZipFile(experiment_archive_zip).extractall(tmpdir)
				return self.load_measures(
					measure_names,
					abs_output_path=os.path.join(
						tmpdir,
						self.rel_output_path,
					)
				)
		else:
			_logger.info(f"loading from {experiment_archive_path}")
			return self.load_measures(
				measure_names,
				abs_output_path=os.path.join(
					experiment_archive_path,
					self.rel_output_path,
				)
			)

	def archive(self, params, model_results_path, experiment_id:int=0):
		raise NotImplementedError

	def run(self):
		raise NotImplementedError

	def post_process(self, params, measure_names, output_path=None):
		"""
		Runs post processors associated with particular performance measures.

		This method is the place to conduct automatic post-processing
		of core model run results, in particular any post-processing that
		is expensive or that will write new output files into the core model's
		output directory.  The core model run should already have
		been completed using `setup` and `run`.  If the relevant performance
		measures do not require any post-processing to create (i.e. they
		can all be read directly from output files created during the core
		model run itself) then this method does not need to be overloaded
		for a particular core model implementation.

		The default implementation of this method is a no-op, but it is
		available to be overloaded for particular implementations.

		Args:
			params (dict):
				Dictionary of experiment variables, with keys as variable names
				and values as the experiment settings. Most post-processing
				scripts will not need to know the particular values of the
				inputs (exogenous uncertainties and policy levers), but this
				method receives the experiment input parameters as an argument
				in case one or more of these parameter values needs to be known
				in order to complete the post-processing.
			measure_names (List[str]):
				List of measures to be processed.  Normally for the first pass
				of core model run experiments, post-processing will be completed
				for all performance measures.  However, it is possible to use
				this argument to give only a subset of performance measures to
				post-process, which may be desirable if the post-processing
				of some performance measures is expensive.  Additionally, this
				method may also be called on archived model results, allowing
				it to run to generate only a subset of (probably new) performance
				measures based on these archived runs.
			output_path (str, optional):
				Path to model outputs.  If this is not given (typical for the
				initial run of core model experiments) then the local/default
				model directory is used.  This argument is provided primarily
				to facilitate post-processing archived model runs to make new
				performance measures (i.e. measures that were not in-scope when
				the core model was actually run).

		Raises:
			KeyError:
				If post process is not available for specified measure
		"""
