

import platform
if 'Darwin' in platform.platform():
	import matplotlib
	matplotlib.use("TkAgg")


import unittest
import pytest
from pytest import approx
import emat

import emat.workbench
import os, numpy, pandas, functools, random
from emat.experiment.experimental_design import design_experiments
from emat.model.core_python import PythonCoreModel
from emat.model.core_python import Road_Capacity_Investment
from emat.workbench import SequentialEvaluator

def stable_df(filename, df, rtol=1e-3):
	if not os.path.exists(filename):
		df.to_pickle(filename)
	return pandas.testing.assert_frame_equal(df, pandas.read_pickle(filename), rtol=rtol)


class TestRoadTest(unittest.TestCase):

	#
	# Tests
	#
	def test_road_test(self):
		import os
		test_dir = os.path.dirname(__file__)
		os.chdir(test_dir)

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
			[ -22.29090499,  -16.84301382, -113.98841188,   11.53956058,        78.03661612])

		assert lhs_results.tail()['present_cost_expansion'].values == approx(
			[2720.51645703, 4000.91232689, 6887.83193063, 3739.47839941, 1582.52899124])

		assert lhs_results.tail()['net_benefits'].values == approx(
			[841.46278175, -146.71279267, -112.5681036, 25.48055303, 127.31154155])

		with SequentialEvaluator(m) as eval_seq:
			lhs_large_results = m.run_experiments(design_name='lhs_large', evaluator=eval_seq)
		lhs_large_results.head()

		assert lhs_large_results.head()['net_benefits'].values == approx(
			[-522.45283083, -355.1599307 , -178.6623215 ,   23.46263498,       -301.17700968])

		lhs_outcomes = m.read_experiment_measures(design_name='lhs')
		assert lhs_outcomes.head()['time_savings'].values == approx(
			[13.4519273, 26.34172999, 12.48385198, 15.10165981, 15.48056139])

		scores = m.get_feature_scores('lhs', random_state=123)
		stable_df("./road_test_feature_scores.pkl.gz", scores.data, rtol=0.2)

		from emat.workbench.analysis import prim

		x = m.read_experiment_parameters(design_name='lhs_large')

		prim_alg = prim.Prim(
			m.read_experiment_parameters(design_name='lhs_large'),
			m.read_experiment_measures(design_name='lhs_large')['net_benefits'] > 0,
			threshold=0.4,
		)

		box1 = prim_alg.find_box()

		stable_df("./road_test_box1_peeling_trajectory.pkl.gz", box1.peeling_trajectory)

		from emat.util.xmle import Show
		from emat.util.xmle.elem import Elem

		assert isinstance(Show(box1.show_tradeoff()), Elem)

		from emat.workbench.analysis import cart

		cart_alg = cart.CART(
			m.read_experiment_parameters(design_name='lhs_large'),
			m.read_experiment_measures(design_name='lhs_large')['net_benefits'] > 0,
		)
		cart_alg.build_tree()

		stable_df("./road_test_cart_box0.pkl.gz", cart_alg.boxes[0])

		cart_dict = dict(cart_alg.boxes[0].iloc[0])
		assert cart_dict['debt_type'] == {'GO Bond', 'Paygo', 'Rev Bond'}
		#assert cart_dict['interest_rate_lock'] == {False, True}

		assert isinstance(Show(cart_alg.show_tree(format='svg')), Elem)

		from emat import Measure

		MAXIMIZE = Measure.MAXIMIZE
		MINIMIZE = Measure.MINIMIZE

		robustness_functions = [
			Measure(
				'Expected Net Benefit',
				kind=Measure.INFO,
				variable_name='net_benefits',
				function=numpy.mean,
				#         min=-150,
				#         max=50,
			),

			Measure(
				'Probability of Net Loss',
				kind=MINIMIZE,
				variable_name='net_benefits',
				function=lambda x: numpy.mean(x < 0),
				min=0,
				max=1,
			),

			Measure(
				'95%ile Travel Time',
				kind=MINIMIZE,
				variable_name='build_travel_time',
				function=functools.partial(numpy.percentile, q=95),
				min=60,
				max=150,
			),

			Measure(
				'99%ile Present Cost',
				kind=Measure.INFO,
				variable_name='present_cost_expansion',
				function=functools.partial(numpy.percentile, q=99),
				#         min=0,
				#         max=10,
			),

			Measure(
				'Expected Present Cost',
				kind=Measure.INFO,
				variable_name='present_cost_expansion',
				function=numpy.mean,
				#         min=0,
				#         max=10,
			),

		]

		from emat import Constraint

		constraint_1 = Constraint(
			"Maximum Log Expected Present Cost",
			outcome_names="Expected Present Cost",
			function=Constraint.must_be_less_than(4000),
		)

		constraint_2 = Constraint(
			"Minimum Capacity Expansion",
			parameter_names="expand_capacity",
			function=Constraint.must_be_greater_than(10),
		)

		constraint_3 = Constraint(
			"Maximum Paygo",
			parameter_names='debt_type',
			outcome_names='99%ile Present Cost',
			function=lambda i, j: max(0, j - 1500) if i == 'Paygo' else 0,
		)

		from emat.optimization import HyperVolume, EpsilonProgress, SolutionViewer, ConvergenceMetrics

		convergence_metrics = ConvergenceMetrics(
			HyperVolume.from_outcomes(robustness_functions),
			EpsilonProgress(),
			SolutionViewer.from_model_and_outcomes(m, robustness_functions),
		)

		with SequentialEvaluator(m) as eval_seq:
			robust = m.robust_optimize(
				robustness_functions,
				scenarios=20,
				nfe=5,
				constraints=[
					constraint_1,
					constraint_2,
					constraint_3,
				],
				epsilons=[0.05, ] * len(robustness_functions),
				convergence=convergence_metrics,
				evaluator=eval_seq,
			)
		robust_results, convergence = robust.result, robust.convergence

		assert isinstance(robust_results, pandas.DataFrame)

		mm = m.create_metamodel_from_design('lhs')

		design2 = design_experiments(road_scope, db=emat_db, n_samples_per_factor=10, sampler='lhs', random_seed=2)

		design2_results = mm.run_experiments(design2)


	def test_robust_evaluation(self):
		# %%

		import os
		test_dir = os.path.dirname(__file__)

		from emat.workbench import ema_logging, MultiprocessingEvaluator, SequentialEvaluator
		from emat.examples import road_test
		import numpy, pandas, functools
		from emat import Measure
		s, db, m = road_test()

		MAXIMIZE = Measure.MAXIMIZE
		MINIMIZE = Measure.MINIMIZE

		robustness_functions = [
			Measure(
				'Expected Net Benefit',
				kind=Measure.INFO,
				variable_name='net_benefits',
				function=numpy.mean,
			),

			Measure(
				'Probability of Net Loss',
				kind=MINIMIZE,
				variable_name='net_benefits',
				function=lambda x: numpy.mean(x < 0),
				min=0,
				max=1,
			),

			Measure(
				'95%ile Travel Time',
				kind=MINIMIZE,
				variable_name='build_travel_time',
				function=functools.partial(numpy.percentile, q=95),
				min=60,
				max=150,
			),

			Measure(
				'99%ile Present Cost',
				kind=Measure.INFO,
				variable_name='present_cost_expansion',
				function=functools.partial(numpy.percentile, q=99),
			),

			Measure(
				'Expected Present Cost',
				kind=Measure.INFO,
				variable_name='present_cost_expansion',
				function=numpy.mean,
			),

		]
		# %%

		numpy.random.seed(42)
		os.chdir(test_dir)
		with SequentialEvaluator(m) as evaluator:
			r1 = m.robust_evaluate(
				robustness_functions,
				scenarios=20,
				policies=5,
				evaluator=evaluator,
			)

		stable_df('./road_test_robust_evaluate.pkl.gz', r1)

		numpy.random.seed(7)

		from emat.workbench.em_framework.samplers import sample_uncertainties
		scenes = sample_uncertainties(m, 20)

		scenes0 = pandas.DataFrame(scenes)
		stable_df('./test_robust_evaluation_scenarios.pkl.gz', scenes0)

		from emat import Constraint

		constraint_1 = Constraint(
			"Maximum Log Expected Present Cost",
			outcome_names="Expected Present Cost",
			function=Constraint.must_be_less_than(4000),
		)

		constraint_2 = Constraint(
			"Minimum Capacity Expansion",
			parameter_names="expand_capacity",
			function=Constraint.must_be_greater_than(10),
		)

		constraint_3 = Constraint(
			"Maximum Paygo",
			parameter_names='debt_type',
			outcome_names='99%ile Present Cost',
			function=lambda i, j: max(0, j - 1500) if i == 'Paygo' else 0,
		)

		from emat.optimization import HyperVolume, EpsilonProgress, SolutionViewer, ConvergenceMetrics

		convergence_metrics = ConvergenceMetrics(
			HyperVolume.from_outcomes(robustness_functions),
			EpsilonProgress(),
			SolutionViewer.from_model_and_outcomes(m, robustness_functions),
		)

		numpy.random.seed(8)
		random.seed(8)

		# Test robust optimize
		with SequentialEvaluator(m) as evaluator:
			robust = m.robust_optimize(
					robustness_functions,
					scenarios=scenes,
					nfe=25,
					constraints=[
						constraint_1,
						constraint_2,
						constraint_3,
					],
					epsilons=[0.05,]*len(robustness_functions),
					convergence=convergence_metrics,
					evaluator=evaluator,
			)
		robust_results, convergence = robust.result, robust.convergence

		stable_df('test_robust_results.pkl.gz', robust_results)

	def test_robust_optimization(self):

		import numpy.random
		import random
		numpy.random.seed(42)
		random.seed(42)
		import textwrap
		import pandas
		import numpy
		import emat.examples
		scope, db, model = emat.examples.road_test()

		import os
		test_dir = os.path.dirname(__file__)
		os.chdir(test_dir)

		result = model.optimize(
			nfe=10,
			searchover='levers',
			check_extremes=1,
		)

		stable_df('./test_robust_optimization.1.pkl.gz', result.result)

		from emat.workbench import Scenario, Policy
		assert result.scenario == Scenario(**{
			'alpha': 0.15, 'beta': 4.0, 'input_flow': 100,
			'value_of_time': 0.075, 'unit_cost_expansion': 100,
			'interest_rate': 0.03, 'yield_curve': 0.01
		})

		worst = model.optimize(
			nfe=10,
			searchover='uncertainties',
			reverse_targets = True,
			check_extremes=1,
			reference={
				'expand_capacity': 100.0,
				'amortization_period': 50,
				'debt_type': 'PayGo',
				'interest_rate_lock': False,
			}
		)

		stable_df('./test_robust_optimization.2.pkl.gz', worst.result)

		from emat import Measure

		minimum_net_benefit = Measure(
			name='Minimum Net Benefits',
			kind=Measure.MAXIMIZE,
			variable_name='net_benefits',
			function=min,
		)

		expected_net_benefit = Measure(
			name='Mean Net Benefits',
			kind=Measure.MAXIMIZE,
			variable_name='net_benefits',
			function=numpy.mean,
		)

		import functools

		pct5_net_benefit = Measure(
			'5%ile Net Benefits',
			kind = Measure.MAXIMIZE,
			variable_name = 'net_benefits',
			function = functools.partial(numpy.percentile, q=5),
		)

		from scipy.stats import percentileofscore

		neg_net_benefit = Measure(
			'Possibility of Negative Net Benefits',
			kind = Measure.MINIMIZE,
			variable_name = 'net_benefits',
			function = functools.partial(percentileofscore, score=0, kind='strict'),
		)

		pct95_cost = Measure(
			'95%ile Capacity Expansion Cost',
			kind = Measure.MINIMIZE,
			variable_name = 'cost_of_capacity_expansion',
			function = functools.partial(numpy.percentile, q = 95),
		)

		expected_time_savings = Measure(
			'Expected Time Savings',
			kind = Measure.MAXIMIZE,
			variable_name = 'time_savings',
			function = numpy.mean,
		)

		robust_result = model.robust_optimize(
			robustness_functions=[
				expected_net_benefit,
				pct5_net_benefit,
				neg_net_benefit,
				pct95_cost,
				expected_time_savings,
			],
			scenarios=50,
			nfe=10,
			check_extremes=1,
		)

		stable_df('./test_robust_optimization.3.pkl.gz', robust_result.result)

		from emat import Constraint

		c_min_expansion = Constraint(
			"Minimum Capacity Expansion",
			parameter_names="expand_capacity",
			function=Constraint.must_be_greater_than(10),
		)

		c_positive_mean_net_benefit = Constraint(
			"Minimum Net Benefit",
			outcome_names = "Mean Net Benefits",
			function = Constraint.must_be_greater_than(0),
		)

		constraint_bad = Constraint(
			"Maximum Interest Rate",
			parameter_names = "interest_rate",
			function = Constraint.must_be_less_than(0.03),
		)

		pct99_present_cost = Measure(
			'99%ile Present Cost',
			kind=Measure.INFO,
			variable_name='present_cost_expansion',
			function=functools.partial(numpy.percentile, q=99),
		)

		c_max_paygo = Constraint(
			"Maximum Paygo",
			parameter_names='debt_type',
			outcome_names='99%ile Present Cost',
			function=lambda i,j: max(0, j-3000) if i=='Paygo' else 0,
		)

		robust_constrained = model.robust_optimize(
			robustness_functions=[
				expected_net_benefit,
				pct5_net_benefit,
				neg_net_benefit,
				pct95_cost,
				expected_time_savings,
				pct99_present_cost,
			],
			constraints = [
				c_min_expansion,
				c_positive_mean_net_benefit,
				c_max_paygo,
			],
			scenarios=50,
			nfe=10,
			check_extremes=1,
		)

		stable_df('./test_robust_optimization.4.pkl.gz', robust_constrained.result)

		with pytest.raises(ValueError):
			model.robust_optimize(
				robustness_functions=[
					expected_net_benefit,
					pct5_net_benefit,
					neg_net_benefit,
					pct95_cost,
					expected_time_savings,
					pct99_present_cost,
				],
				constraints = [
					constraint_bad,
					c_min_expansion,
					c_positive_mean_net_benefit,
					c_max_paygo,
				],
				scenarios=50,
				nfe=10,
				check_extremes=1,
			)

