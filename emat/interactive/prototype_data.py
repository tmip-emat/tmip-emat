import numpy, pandas
import ipywidgets
from ipywidgets import interactive
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .lazy import lazy
from .prototype_logging import logger

from ..database import SQLiteDB, Database
from .prototype_pane import ExplorerComponent

class ExplorerData(ExplorerComponent):

	def __init__(
			self,
			parent,
		):

		super().__init__(parent_widget=parent)

		self._all_performance_measures = None
		self._all_strategy_names = None
		self._all_risk_factor_names = None


	def load_from_database(
			self,
			performance_measures=None,
			risk_factor_names=None,
			strategy_names=None,
			feature_names=None,

			design_name = None,
	):

		if not isinstance(self.db, Database):
			raise ValueError('db not ready')

		if design_name is None:
			design_name = self.design_name

		self.X = self.db.read_experiment_parameters(self.scope.name, design_name)
		self.Y = self.db.read_experiment_measures(self.scope.name, design_name)

		if feature_names is not None:
			performance_measures = [i for i in feature_names if i in self.all_performance_measures_]
			risk_factor_names    = [i for i in feature_names if i in self.all_risk_factors_]
			strategy_names       = [i for i in feature_names if i in self.all_strategy_names_]

		if performance_measures is None:
			self.performance_measures = self.all_performance_measures
		else:
			self.performance_measures = list(performance_measures)

		if strategy_names is None:
			self.strategy_names = self.all_strategy_names
		else:
			self.strategy_names = list(strategy_names)

		if risk_factor_names is None:
			self.risk_factor_names = self.all_risk_factors
		else:
			self.risk_factor_names = list(risk_factor_names)

		self.X = self.X.loc[:, self.risk_factor_names+self.strategy_names]

		try:
			self.Y = self.Y.loc[:, performance_measures]
		except KeyError:
			logger.debug(":: The columns of Y include:")
			for col in self.Y.columns:
				logger.debug(f":: * {col}")
			raise


		Y_millions = self.Y.min(axis=0) > 1e6
		Y_thousands = (self.Y.min(axis=0) > 1e3) & (~Y_millions)

		self.Y.loc[:, Y_millions] /= 1e6
		self.Y.loc[:, Y_thousands] /= 1e3

		Y_millions.loc[:] = False
		Y_thousands.loc[:] = False

		self.joint_data = pandas.concat([self.X, self.Y], axis=1)

		# self.risk_factor_names = [i for i in self.X.columns if i not in self.strategy_names]
		# self.performance_measure_names = self.Y.columns

	# def workbench_format(self):
	# 	try:
	# 		self.X
	# 		self.Y
	# 	except AttributeError:
	# 		# must load from the database if not already in memory
	# 		self.load_from_database()
	#
	# 	experiments = self.X.copy()
	# 	for strategy in self.strategy_names:
	# 		# We presently assume all strategies are categorical (namely, binary)
	# 		# this could be relaxed in the future
	# 		# but it will always be important to explicitly transform datatypes here
	# 		# for categorical inputs (strategies OR risks)
	# 		experiments[strategy] = (experiments[strategy]>0).astype(str)
	# 	y = {
	# 		k:self.Y[k].values
	# 		for k in self.Y.columns
	# 	}
	# 	return experiments.to_records(), y

	# def singleton(self, qry):
	# 	try:
	# 		return self.engine.execute(qry).fetchone()[0]
	# 	except TypeError:
	# 		return
	#
	#
	# def data_X(self):
	# 	X = pandas.read_sql_query(f"""
	# 	SELECT
	# 	  sampleID, varID, value
	# 	FROM
	# 	  perfMeasInput
	# 	ORDER BY
	# 	  sampleID, varID
	# 	""", self.engine).pivot(index='sampleID', columns='varID', values='value')
	# 	X.columns = [
	# 		(self.singleton(f"SELECT riskVarDesc FROM riskVar WHERE riskVarID={j}")
	# 		 or self.singleton(f"SELECT strategyDesc FROM strategy WHERE strategyID={j}")
	# 		 or j)
	# 		for j in X.columns
	# 	]
	# 	return X
	#
	#
	# def data_Y(self):
	# 	Y = pandas.read_sql_query(f"""
	# 	SELECT
	# 	  sampleID, perfMeasID, estimate
	# 	FROM
	# 	  perfMeasRes
	# 	ORDER BY
	# 	  sampleID, perfMeasID
	# 	""", self.engine).pivot(index='sampleID', columns='perfMeasID', values='estimate')
	# 	Y.columns = [
	# 		(self.singleton(f"SELECT perfMeasDesc FROM perfMeas WHERE perfMeasID={j}")
	# 		 or j)
	# 		for j in Y.columns
	# 	]
	# 	Y = numpy.exp(Y)
	# 	return Y
	#

	@lazy
	def all_risk_factors(self):
		assert isinstance(self.db, Database)
		return self.db.read_uncertainties(self.scope.name)

	@lazy
	def all_strategy_names(self):
		assert isinstance(self.db, Database)
		return self.db.read_levers(self.scope.name)

	@lazy
	def all_performance_measures(self):
		assert isinstance(self.db, Database)
		return self.db.read_measures(self.scope.name)

	@lazy
	def all_risk_factors_(self):
		return set(self.all_risk_factors)

	@lazy
	def all_strategy_names_(self):
		return set(self.all_strategy_names)

	@lazy
	def all_performance_measures_(self):
		return set(self.all_performance_measures)

	# def query(self, query):
	# 	"An arbitrary database query.  Be careful!"
	# 	return pandas.read_sql_query(query, self.engine)
	#
	# def table_schema(self, tablename):
	# 	return self.singleton(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tablename}'")