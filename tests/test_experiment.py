
import unittest
import os
import pytest
from pytest import approx
import pandas as pd
import numpy as np
import emat
from emat.scope.scope import Scope
from emat.database.sqlite.sqlite_db import SQLiteDB




class TestExperimentMethods(unittest.TestCase):
    ''' 
        tests generating experiments      
    '''
    #  
    # one time test setup
    #
    scope_file = emat.package_file("scope","tests","scope_test.yaml")
    scp = Scope(scope_file)

    db_test = SQLiteDB(":memory:", initialize=True)
    scp.store_scope(db_test)

    def test_latin_hypercube(self):
        exp_def = self.scp.design_experiments(
            n_samples_per_factor=10,
            random_seed=1234,
            sampler='lhs',
            db=self.db_test,
        )
        assert len(exp_def) == self.scp.n_sample_factors()*10
        assert (exp_def['TestRiskVar'] == 1.0).all()
        assert (exp_def['Land Use - CBD Focus']).mean() == approx(1.0326, abs=1e-3)
        assert (exp_def['Freeway Capacity']).mean() == approx(1.5, abs=1e-3)

        exp_def2 = self.db_test.read_experiment_parameters(self.scp.name,'lhs')
        assert (exp_def[exp_def2.columns] == exp_def2).all().all()

    def test_latin_hypercube_not_joint(self):
        exp_def = self.scp.design_experiments(
            n_samples_per_factor=5,
            random_seed=1234,
            sampler='lhs',
            db=self.db_test,
            jointly=False,
            design_name='lhs_not_joint',
        )
        assert len(exp_def) == len(self.scp.get_uncertainties())*5 * len(self.scp.get_levers())*5
        assert (exp_def['TestRiskVar'] == 1.0).all()
        assert (exp_def['Land Use - CBD Focus']).mean() == approx(1.033, abs=1e-2)
        assert (exp_def['Freeway Capacity']).mean() == approx(1.5, abs=1e-2)

        exp_def2 = self.db_test.read_experiment_parameters(self.scp.name,'lhs_not_joint')
        assert (exp_def[exp_def2.columns] == exp_def2).all().all()

    def test_monte_carlo(self):
        exp_def = self.scp.design_experiments(
            n_samples_per_factor=10,
            random_seed=1234,
            sampler='mc',
            db=self.db_test,
        )
        assert len(exp_def) == self.scp.n_sample_factors()*10
        assert (exp_def['TestRiskVar'] == 1.0).all()
        assert (exp_def['Land Use - CBD Focus']).mean() == approx(1.04, abs=0.01)
        assert (exp_def['Freeway Capacity']).mean() == approx(1.5, abs=0.01)

        exp_def2 = self.db_test.read_experiment_parameters(self.scp.name,'mc')
        assert (exp_def[exp_def2.columns] == exp_def2).all().all()

    def test_sensitivity_tests(self):
        exp_def = self.scp.design_experiments(
            sampler='uni',
            db=self.db_test,
        )
        cols = ['TestRiskVar', 'Land Use - CBD Focus', 'Freeway Capacity',
           'Auto IVTT Sensitivity', 'Shared Mobility',
           'Kensington Decommissioning', 'LRT Extension']
        correct = '{"TestRiskVar":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0},' \
                  '"Land Use - CBD Focus":{"0":1.0,"1":0.82,"2":1.37,"3":1.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0},' \
                  '"Freeway Capacity":{"0":1.0,"1":1.0,"2":1.0,"3":2.0,"4":1.0,"5":1.0,"6":1.0,"7":1.0},' \
                  '"Auto IVTT Sensitivity":{"0":1.0,"1":1.0,"2":1.0,"3":1.0,"4":0.75,"5":1.0,"6":1.0,"7":1.0},' \
                  '"Shared Mobility":{"0":0.0,"1":0.0,"2":0.0,"3":0.0,"4":0.0,"5":1.0,"6":0.0,"7":0.0},' \
                  '"Kensington Decommissioning":{"0":false,"1":false,"2":false,"3":false,"4":false,' \
                  '"5":false,"6":true,"7":false},"LRT Extension":{"0":false,"1":false,"2":false,"3":false,' \
                  '"4":false,"5":false,"6":false,"7":true}}'
        correct = pd.read_json(correct)
        for k in cols:
            assert (exp_def[k].values == approx(correct[k].values))

        exp_def2 = self.db_test.read_experiment_parameters(self.scp.name,'uni')
        for k in cols:
            assert (exp_def2[k].values == approx(correct[k].values))




class TestCorrelatedExperimentMethods(unittest.TestCase):
    '''
        tests generating experiments
    '''
    #
    # one time test setup
    #

    def test_correlated_latin_hypercube(self):
        scope_file = emat.package_file("model", "tests", "road_test_corr.yaml")
        scp = Scope(scope_file)
        exp_def = scp.design_experiments(
            n_samples_per_factor=10,
            random_seed=1234,
            sampler='lhs',
        )
        assert len(exp_def) == scp.n_sample_factors() * 10
        assert (exp_def['free_flow_time'] == 60).all()
        assert (exp_def['initial_capacity'] == 100).all()
        assert np.corrcoef([exp_def.alpha, exp_def.beta])[0, 1] == approx(0.75, rel=0.05)
        assert np.corrcoef([exp_def.alpha, exp_def.expand_capacity])[0, 1] == approx(0.0, abs=0.01)
        assert np.corrcoef([exp_def.input_flow, exp_def.value_of_time])[0, 1] == approx(-0.5, rel=0.05)
        assert np.corrcoef([exp_def.unit_cost_expansion, exp_def.value_of_time])[0, 1] == approx(0.9, rel=0.05)

    def test_correlated_latin_hypercube_bad(self):
        scope_file = emat.package_file("model", "tests", "road_test_corr_bad.yaml")
        scp = Scope(scope_file)
        with pytest.raises(np.linalg.LinAlgError):
            scp.design_experiments(
                n_samples_per_factor=10,
                random_seed=1234,
                sampler='lhs',
            )



if __name__ == '__main__':
    unittest.main()
