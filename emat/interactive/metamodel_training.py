import sqlite3
import numpy as np
from sklearn import linear_model
from scipy import stats
import pandas
from sqlalchemy import create_engine
from .lazy import lazy



#
# def load_training_X(database_path, ema_id):
# 	"""Select experiment variable values from db and format for GPR model.
# 	"""
# 	result = []
# 	with sqlite3.connect(database_path) as con:
# 		con.row_factory = sqlite3.Row
# 		cur = con.cursor()
# 		id = 1
# 		expRow = [1]
# 		for row in cur.execute(
# 				"""
# 				SELECT expmntID, varID, value FROM expmntDef
# 				WHERE emaID = ?
# 				ORDER BY expmntID, varID
# 				""",
# 				[ema_id],
# 		).fetchall():
# 			rowId = row["expmntID"]
# 			if rowId != id:
# 				result.append(expRow)
# 				# start with 1 (0 is intercept)
# 				expRow = [1, row["value"]]
# 				id = rowId
# 			else:
# 				expRow.append(row["value"])
# 		result.append(expRow)
# 	return np.array(result)
#
# def load_training_X(self):
# 	"""Select experiment performance measure values from
# 	db and format for GPR model.
# 	"""
# 	result = []
# 	with sqlite3.connect(self.database_path) as con:
# 		con.row_factory = sqlite3.Row
# 		cur = con.cursor()
# 		return np.array(
# 			cur.execute(self._LOAD_PERF_MEASURE_SQL,
# 						[self.ema_id, self.perf_meas_id]).fetchall())
#
#
# def metamodel_training_data_y(perfMeasID=1,):
# 	y = pandas.read_sql_query(f"""
# 	SELECT
# 	  expmntID, value
# 	FROM
# 	  expmntPerfMeas
# 	WHERE
# 	  perfMeasID == {perfMeasID}
# 	ORDER BY
# 	  expmntID
# 	""", engine)
# 	y.index = y.expmntID
# 	name = singleton(f"SELECT perfMeasDesc FROM perfMeas WHERE perfMeasID={perfMeasID}")
# 	if name is None:
# 		name = 'value'
# 	y.columns = ['expmntID', name]
# 	return y.drop('expmntID', axis=1)
#
#
# def metamodel_training_data_Y(perfMeasIDs=(1,2,3),):
# 	return pandas.concat([metamodel_data_y(i) for i in perfMeasIDs], axis=1)
#


class EMA_Data():

	def __init__(
			self,
			prototype_data_file = 'ema_poc_fullrun2.db',
		):
		self.engine = create_engine(f'sqlite:///{prototype_data_file}')
		self._all_performance_measures = None
		self._all_strategy_names = None
		self._all_risk_factor_names = None



	def singleton(self, qry, *args, default=None):
		try:
			return self.engine.execute(qry, *args).fetchone()[0]
		except TypeError:
			return default


	def training_X(self):
		X = pandas.read_sql_query(f"""
		SELECT
		  expmntID, varID, value 
		FROM
		  expmntDef
		ORDER BY
		  expmntID, varID
		""", self.engine).pivot(index='expmntID', columns='varID', values='value')
		# WHERE
		#   emaID = ?
		#
		X.columns = [
			(self.singleton(f"SELECT riskVarDesc FROM riskVar WHERE riskVarID={j}")
			 or self.singleton(f"SELECT strategyDesc FROM strategy WHERE strategyID={j}")
			 or j)
			for j in X.columns
		]
		return X



	def training_y(self, perfMeasID=1, ):
		if isinstance(perfMeasID, str):
			perfMeasID = self.performance_measure_id_from_name(perfMeasID)

		y = pandas.read_sql_query(f"""
		SELECT 
		  expmntID, value 
		FROM 
		  expmntPerfMeas
		WHERE
		  perfMeasID == {perfMeasID}
		ORDER BY
		  expmntID
		""", self.engine)
		y.index = y.expmntID
		name = self.singleton(f"SELECT perfMeasDesc FROM perfMeas WHERE perfMeasID={perfMeasID}")
		if name is None:
			name = 'value'
		y.columns = ['expmntID', name]
		return y.drop('expmntID', axis=1)

	def training_Y(self, perfMeasIDs=(1, 2, 3), ):
		return pandas.concat([self.training_y(i) for i in perfMeasIDs], axis=1)

	# def training_Y(self):
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


	@lazy
	def all_risk_factors(self):
		_risk_factor_names = pandas.read_sql_query(f"""
				SELECT 
				  riskVarDesc
				FROM 
				  riskVar
				""", self.engine).values
		return tuple([j[0] for j in _risk_factor_names])

	@lazy
	def all_strategy_names(self):
		_strategy_names = pandas.read_sql_query(f"""
			SELECT 
			  strategyDesc
			FROM 
			  strategy
			""", self.engine).values
		return tuple([j[0] for j in _strategy_names])

	@lazy
	def all_performance_measures(self):

		# There are a lot of performance measure ID/Names that have no backing data.
		# Only load those with backing data available...
		_pm = pandas.read_sql_query(f"""
			SELECT 
			  perfMeasDesc
			FROM 
			  perfMeas
			WHERE
			  perfMeasID IN ( SELECT DISTINCT(perfMeasID) FROM perfMeasRes )
			""", self.engine).values
		return tuple([j[0] for j in _pm])

	def performance_measure_id_from_name(self, name):
		return self.singleton(f"SELECT perfMeasID FROM perfMeas WHERE perfMeasDesc=? LIMIT 1", (name,))

	@lazy
	def all_risk_factors_(self):
		return set(self.all_risk_factors)

	@lazy
	def all_strategy_names_(self):
		return set(self.all_strategy_names)

	@lazy
	def all_performance_measures_(self):
		return set(self.all_performance_measures)

	def query(self, query):
		"An arbitrary database query.  Be careful!"
		return pandas.read_sql_query(query, self.engine)

	def table_schema(self, tablename):
		return self.singleton(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{tablename}'")