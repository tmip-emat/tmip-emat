
import tempfile
import os
import re
import pandas as pd
import numpy as np
import shutil
from distutils.dir_util import copy_tree

from .core_files import FilesCoreModel, TableParser, MappingParser, loc, key
from ... import package_file
from ...util.loggers import get_module_logger

_logger = get_module_logger(__name__)

class ReplacementOfNumber:
	"""
	This class provides a mechanism to edit a text file, replacing
	a the numerical value of a particular parameter with a new value.
	"""
	numbr = r"([-+]?\d*\.?\d*[eE]?[-+]?\d*|\d+\/\d+)"  # matches any number representation
	def __init__(self, varname, assign_operator=":", logger=None):
		self.varname = varname
		self.regex = re.compile(f"({varname}\s*{assign_operator}\s*)({self.numbr})")
		self.logger = logger
	def sub(self, value, s):
		s, n = self.regex.subn(f"\g<1>{value}", s)
		if self.logger is not None:
			self.logger.info(f"For '{self.varname}': {n} substitutions made")
		return s

class ReplacementOfString:
	"""
	This class provides a mechanism to edit a yaml file, replacing
	the string value of a particular parameter with a new value.
	"""
	def __init__(self, varname, assign_operator=":", logger=None):
		self.varname = varname
		self.regex = re.compile(f"({varname}\s*{assign_operator}\s*)([^#\n]*)(#.*)?", flags=re.MULTILINE)
		self.logger = logger
	def sub(self, value, s):
		s, n = self.regex.subn(f"\g<1>{value}  \g<3>", s)
		if self.logger is not None:
			self.logger.info(f"For '{self.varname}': {n} substitutions made")
		return s

class RoadTestFileModel(FilesCoreModel):

	def __init__(self):

		# Make a temporary directory for this example
		# A 'real' core models application may want to use a
		# more permanent directory.
		self.master_directory = tempfile.TemporaryDirectory()
		os.chdir(self.master_directory.name)

		# Initialize the working directory of the files-based model.
		# Depending on how large your core model is, you may or may
		# not want to be copying the whole thing.
		copy_tree(
			package_file('examples','road-test-files'),
			os.path.join(self.master_directory.name, "road-test-files"),
		)

		# Housekeeping for this example:
		# move the CONFIG file out of the model files directory
		os.replace(
			os.path.join(self.master_directory.name, "road-test-files", "road-test-model-config.yml"),
			os.path.join(self.master_directory.name, "road-test-model-config.yml"),
		)

		# Initialize
		super().__init__(
			configuration=os.path.join(self.master_directory.name, "road-test-model-config.yml"),
			scope = package_file('model','tests','road_test.yaml'),
		)

		# Add parsers to instruct the load_measures function
		# how to parse the outputs and get the measure values.
		self.add_parser(
			TableParser(
				"output_1.csv.gz",
				{
					'value_of_time_savings': loc['plain', 'value_of_time_savings'],
					'present_cost_expansion': loc['plain', 'present_cost_expansion'],
					'cost_of_capacity_expansion': loc['plain', 'cost_of_capacity_expansion'],
					'net_benefits': loc['plain', 'net_benefits'],
				},
				index_col=0,
			)
		)
		self.add_parser(
			MappingParser(
				"output.yaml",
				{
					'build_travel_time': key['build_travel_time'],
					'no_build_travel_time': key['no_build_travel_time'],
					'time_savings': key['time_savings'],
				}
			)
		)


	def setup(self, params: dict):

		_logger.info("RoadTestFileModel SETUP...")

		numbers_to_levers_file = [
			'expand_capacity',
			'amortization_period',
			'interest_rate_lock',
			'lane_width',
			'mandatory_unused_lever',
		]

		strings_to_levers_file = [
			'debt_type',
		]

		numbers_to_uncs_file = [
			'alpha',
			'beta',
			'input_flow',
			'value_of_time',
			'unit_cost_expansion',
			'interest_rate',
			'yield_curve',
		]

		# load the text of the LEVERS yaml file
		with open(os.path.join(self.resolved_model_path, 'levers.yml'), 'rt') as f:
			y = f.read()

		# use regex to manipulate the content, inserting the defined
		# parameter values
		for n in numbers_to_levers_file:
			if n in params:
				y = ReplacementOfNumber(n).sub(params[n], y)
		for s in strings_to_levers_file:
			if s in params:
				y = ReplacementOfString(s).sub(params[s], y)

		# write the manipulated text back out to the LEVER file
		with open(os.path.join(self.resolved_model_path, 'levers.yml'), 'wt') as f:
			f.write(y)

		# load the text of the UNCERTAINTIES yaml file
		with open(os.path.join(self.resolved_model_path, 'uncertainties.yml'), 'rt') as f:
			y = f.read()

		# use regex to manipulate the content, inserting the defined
		# parameter values
		for n in numbers_to_uncs_file:
			if n in params:
				y = ReplacementOfNumber(n).sub(params[n], y)

		# write the manipulated text back out to the UNCERTAINTIES file
		with open(os.path.join(self.resolved_model_path, 'uncertainties.yml'), 'wt') as f:
			f.write(y)

		_logger.info("RoadTestFileModel SETUP complete")


	def run(self):
		_logger.info("RoadTestFileModel RUN ...")
		import subprocess
		subprocess.run(['emat-road-test-demo', '--uncs', 'uncertainties.yml'], cwd=self.resolved_model_path)
		_logger.info("RoadTestFileModel RUN complete")

	def post_process(self, params, measure_names, output_path=None):
		_logger.info("RoadTestFileModel POST-PROCESS ...")

		# Create Outputs directory as needed.
		os.makedirs(
			os.path.join(self.resolved_model_path, self.rel_output_path),
			exist_ok=True,
		)

		# Do some processing to recover values from output.csv.gz
		df = pd.read_csv(
			os.path.join(self.resolved_model_path, 'output.csv.gz'),
			index_col=0,
		)
		repair = pd.isna(df.loc['plain'])
		df.loc['plain', repair] = np.log(df.loc['exp', repair])*1000
		# Write edited output.csv.gz to Outputs directory.
		df.to_csv(
			os.path.join(self.resolved_model_path, self.rel_output_path, 'output_1.csv.gz')
		)

		# Copy output.yaml to Outputs directory, no editing needed.
		shutil.copy2(
			os.path.join(self.resolved_model_path, 'output.yaml'),
			os.path.join(self.resolved_model_path, self.rel_output_path, 'output.yaml'),
		)

		# Log the names of all the files in the master directory
		_logger.info(f"Files in {self.master_directory.name}")
		for i,j,k in os.walk(self.master_directory.name):
			for eachfile in k:
				_logger.info(os.path.join(i,eachfile).replace(self.master_directory.name, '.'))

		_logger.info("RoadTestFileModel POST-PROCESS complete")


	def archive(self, params, model_results_path, experiment_id: int = 0):
		"""
		Copies model outputs to archive location.

		Args:
			params (dict): Dictionary of experiment variables
			model_results_path (str): archive path
			experiment_id (int, optional): The id number for this experiment.

		"""
		if model_results_path is None:
			model_results_path = self.get_experiment_archive_path(experiment_id)
		copy_tree(
			os.path.join(self.master_directory.name, "road-test-files"),
			model_results_path,
		)

