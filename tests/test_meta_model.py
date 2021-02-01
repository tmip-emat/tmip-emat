import unittest
import os
import pandas as pd
import numpy as np
import yaml

import pytest
from pytest import approx

import emat
from emat.scope.scope import Scope

def stable_df(filename, df):
    if not os.path.exists(filename):
        df.to_pickle(filename)
    return pd.testing.assert_frame_equal(df, pd.read_pickle(filename))


class TestMetaModelMethods(unittest.TestCase):
    ''' 
        tests model and meta-model methods     
    '''
    metam_scope_file = emat.package_file("model","tests","metam_test.yaml")
    metam_scp = Scope(metam_scope_file)

    
# =============================================================================
#     
#      Meta model tests
#     
# =============================================================================



    def test_derive_meta(self):
        import os
        test_dir = os.path.dirname(__file__)
        os.chdir(test_dir)
        from emat.examples import road_test

        s, db, m = road_test()

        db.get_db_info()

        m.design_experiments(n_samples=10, design_name='tiny')

        db.read_experiment_all(None, None)

        with pytest.raises(emat.PendingExperimentsError):
            m.create_metamodel_from_design('tiny', random_state=123)

        m.run_experiments(design_name='tiny')

        mm = emat.create_metamodel(
            m.scope,
            db.read_experiment_all(s.name, 'tiny'),
            random_state=123,
            metamodel_id=db.get_new_metamodel_id(None),
        )
        mm.db = db # add db after creation to prevent writing it into the db
        assert mm.scope == m.scope

        tiny2 = m.design_experiments(n_samples=10, design_name='tiny2', random_seed=456)

        assert tiny2.iloc[0]['debt_type'] == 'GO Bond'

        stable_df('./test_tiny2.pkl.gz',tiny2)

        result2 = mm.run_experiments('tiny2')

        tiny2out = mm.read_experiment_measures('tiny2')
        tiny2out.index = tiny2out.index.droplevel(1)
        stable_df('./test_tiny2out.pkl.gz', tiny2out)

        with pytest.raises(ValueError):
            # no metamodels stored
            mm3 = db.read_metamodel(None, None)

        db.write_metamodel(mm)
        mm2 = db.read_metamodel(None, 1)
        mm3 = db.read_metamodel(None, None)
        assert mm2 == mm == mm3
        assert mm2 is not mm

        print(mm2.function(**(tiny2.iloc[0])))

        assert mm2.function(**(tiny2.iloc[0])) == approx({
            'no_build_travel_time': 83.57502327972276,
            'build_travel_time': 62.221693766038015,
            'time_savings': 57.612063365257995,
            'value_of_time_savings': 3749.2913256457214,
            'net_benefits': 395.55020765212254,
            'cost_of_capacity_expansion': 1252.6916865286616,
            'present_cost_expansion': 23000.275573551233,
        })

        mm3.metamodel_id = db.get_new_metamodel_id(None)
        db.write_metamodel(mm3)

        with pytest.raises(ValueError):
            # now too many to get without giving an ID
            mm4 = db.read_metamodel(None, None)

    def test_derive_meta_w_transform(self):
        from emat.examples import road_test

        s, db, m = road_test(yamlfile='road_test2.yaml')

        db.get_db_info()

        m.design_experiments(n_samples=10, design_name='tiny')

        db.read_experiment_all(None, None)

        with pytest.raises(emat.PendingExperimentsError):
            m.create_metamodel_from_design('tiny', random_state=123)

        m.run_experiments(design_name='tiny')

        mm = emat.create_metamodel(
            m.scope,
            db.read_experiment_all(s.name, 'tiny'),
            random_state=123,
            metamodel_id=db.get_new_metamodel_id(None),
        )

        assert mm.scope != m.scope  # now not equal as road_test2 has transforms that are stripped.
        mm.db = db
        tiny2 = m.design_experiments(n_samples=10, design_name='tiny2', random_seed=456)

        assert tiny2.iloc[0]['debt_type'] == 'GO Bond'

        assert dict(tiny2.iloc[0].drop('debt_type')) == approx({
            'alpha': 0.10428005571929212,
            'amortization_period': 33,
            'beta': 4.8792451185772014,
            'expand_capacity': 61.4210886403998,
            'input_flow': 137,
            'interest_rate': 0.03099304322197216,
            'interest_rate_lock': 0,
            'unit_cost_expansion': 121.85520427974882,
            'value_of_time': 0.002953613029133872,
            'yield_curve': 0.016255990123028242,
            'free_flow_time': 60,
            'initial_capacity': 100})

        result2 = mm.run_experiments('tiny2')

        assert dict(mm.read_experiment_measures('tiny2').iloc[0]) == approx({
            'no_build_travel_time': 81.6839454971052,
            'build_travel_time': 61.91038371206646,
            'log_build_travel_time': 4.120826572003798,
            'time_savings': 44.94189289289446,
            'value_of_time_savings': 2904.081661408463,
            'net_benefits': -34.09931528157315,
            'cost_of_capacity_expansion': 1085.3565091745982,
            'present_cost_expansion': 19923.66625500023,
        })

        assert m.run_experiment(tiny2.iloc[0]) == approx({
            'no_build_travel_time': 89.07004237532217,
            'build_travel_time': 62.81032484779827,
            'log_build_travel_time': np.log(62.81032484779827),
            'time_savings': 26.259717527523904,
            'value_of_time_savings': 10.62586300480175,
            'present_cost_expansion': 7484.479303360477,
            'cost_of_capacity_expansion': 395.69034710662226,
            'net_benefits': -385.0644841018205,
        })

        with pytest.raises(ValueError):
            # no metamodels stored
            mm3 = db.read_metamodel(None, None)

        db.write_metamodel(mm)
        mm2 = db.read_metamodel(None, 1)
        mm3 = db.read_metamodel(None, None)
        assert mm2 == mm == mm3
        assert mm2 is not mm

        assert mm2.function(**(tiny2.iloc[0])) == approx({
            'no_build_travel_time': 81.6839454971052,
            'build_travel_time': 61.91038371206646,
            'log_build_travel_time': 4.120826572003798,
            'time_savings': 44.94189289289446,
            'value_of_time_savings': 2904.081661408463,
            'net_benefits': -34.09931528157315,
            'cost_of_capacity_expansion': 1085.3565091745982,
            'present_cost_expansion': 19923.66625500023,
        })

        mm3.metamodel_id = db.get_new_metamodel_id(None)
        db.write_metamodel(mm3)

        with pytest.raises(ValueError):
            # now too many to get without giving an ID
            mm4 = db.read_metamodel(None, None)


    def test_exogenously_stratified_k_fold(self):
        from emat.learn.splits import ExogenouslyStratifiedKFold
        X = np.arange(20)
        Y = np.asarray([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1])
        S = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        correct = [np.array([0, 1, 2, 14, 15]),
                   np.array([3, 4, 5, 16]),
                   np.array([6, 7, 8, 17]),
                   np.array([9, 10, 11, 18]),
                   np.array([12, 13, 19])]
        for j, (_, k) in zip(correct, ExogenouslyStratifiedKFold(n_splits=5, exo_data=S).split(X, Y)):
            assert np.array_equal(j, k)


if __name__ == '__main__':
    unittest.main()
    