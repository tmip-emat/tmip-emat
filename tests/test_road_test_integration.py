


import unittest
import pytest
from pytest import approx
import emat

import ema_workbench
import os, numpy, pandas, functools
from emat.experiment.experimental_design import design_experiments
from emat.model.core_python import PythonCoreModel
from emat.model.core_python import Road_Capacity_Investment
from ema_workbench import SequentialEvaluator

class TestRoadTest(unittest.TestCase):

	#
	# Tests
	#
	def test_road_test(self):
		road_test_scope_file = emat.package_file('model', 'tests', 'road_test.yaml')

		road_scope = emat.Scope(road_test_scope_file)

		# <emat.Scope with 2 constants, 7 uncertainties, 4 levers, 7 measures>
		assert len(road_scope.get_measures()) == 7
		assert len(road_scope.get_levers()) == 4
		assert len(road_scope.get_uncertainties()) == 7
		assert len(road_scope.get_constants()) == 2

		emat_db = emat.SQLiteDB()

		road_scope.store_scope(emat_db)

		with pytest.raises(KeyError):
			road_scope.store_scope(emat_db)

		assert emat_db.read_scope_names() == ['EMAT Road Test']

		design = design_experiments(road_scope, db=emat_db, n_samples_per_factor=10, sampler='lhs')
		design.head()

		large_design = design_experiments(road_scope, db=emat_db, n_samples=5000, sampler='lhs',
										  design_name='lhs_large')
		large_design.head()

		assert list(large_design.columns) == [
			'alpha',
			'amortization_period',
			'beta',
			'debt_type',
			'expand_capacity',
			'input_flow',
			'interest_rate',
			'interest_rate_lock',
			'unit_cost_expansion',
			'value_of_time',
			'yield_curve',
			'free_flow_time',
			'initial_capacity',
		]

		assert list(large_design.head().index) == [111, 112, 113, 114, 115]

		assert emat_db.read_design_names('EMAT Road Test') == ['lhs', 'lhs_large']

		m = PythonCoreModel(Road_Capacity_Investment, scope=road_scope, db=emat_db)

		with SequentialEvaluator(m) as eval_seq:
			lhs_results = m.run_experiments(design_name='lhs', evaluator=eval_seq)

		lhs_results.head()

		assert lhs_results.head()['present_cost_expansion'].values == approx(
			[2154.41598475, 12369.38053473, 4468.50683924, 6526.32517089, 2460.91070514])

		assert lhs_results.head()['net_benefits'].values == approx(
			[-79.51551505, -205.32148044, -151.94431822, -167.62487134, -3.97293985])

		with SequentialEvaluator(m) as eval_seq:
			lhs_large_results = m.run_experiments(design_name='lhs_large', evaluator=eval_seq)
		lhs_large_results.head()

		assert lhs_large_results.head()['net_benefits'].values == approx(
			[-584.36098322, -541.5458395, -185.16661464, -135.85689709, -357.36106457])

		lhs_outcomes = m.read_experiment_measures(design_name='lhs')
		assert lhs_outcomes.head()['time_savings'].values == approx(
			[13.4519273, 26.34172999, 12.48385198, 15.10165981, 15.48056139])

