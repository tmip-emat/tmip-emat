# -*- coding: utf-8 -*-

import unittest
import os
import pandas as pd
import numpy as np
import pytest

from emat.database.sqlite.sqlite_db import SQLiteDB
from emat._pkg_constants import *

import emat
from emat import config

class TestDatabaseMethods(unittest.TestCase):
   
    ''' 
        tests writing and reading experiments to database       
    '''
    #  
    # one time test setup
    #

    db_test = SQLiteDB(config.get("test_db_filename", ":memory:"), initialize=True)

    # load experiment variables and performance measures
    scp_xl = [('constant', 'constant'),
                ('exp_var1', 'risk'),
                ('exp_var2', 'strategy')]
    scp_m = [('pm_1','none'), ('pm_2','ln')]
    
    db_test.init_xlm(scp_xl, scp_m)
    
    # create emat scope 
    scope_name = 'test'
    sheet = 'emat_scope1.yaml'
    ex_xl = ['constant','exp_var1','exp_var2']
    ex_m = ['pm_1', 'pm_2']
    db_test.delete_scope(scope_name)
    
    def setUp(self):
         # create emat scope 
        self.db_test.write_scope(self.scope_name,
                                  self.sheet, 
                                  self.ex_xl, 
                                  self.ex_m)

    def tearDown(self):
        self.db_test.delete_scope(self.scope_name)
    
    #
    # Tests
    #
    
    def test_delete_experiment(self):
         # write experiment definition
        xl_df = pd.DataFrame({'constant' : [1,1], 
                                'exp_var1' : [1.1,1.2],
                                'exp_var2' : [2.1,2.2]})
        design = 'lhs'
        self.db_test.write_experiment_parameters(self.scope_name, design, xl_df)
        self.db_test.delete_experiments(self.scope_name, design)
    
        xl_readback = self.db_test.read_experiment_parameters(self.scope_name,design)
        #note - indexes may not match
        self.assertTrue(xl_readback.empty)
        
        
    def test_create_experiment(self):
         # write experiment definition
        xl_df = pd.DataFrame({'constant' : [1,1], 
                                'exp_var1' : [1.1,1.2],
                                'exp_var2' : [2.1,2.2]})
        design = 'lhs'
        self.db_test.write_experiment_parameters(self.scope_name, design, xl_df)
    
        xl_readback = self.db_test.read_experiment_parameters(self.scope_name,design)
        #note - indexes may not match
        self.assertTrue(np.array_equal(xl_readback.values, xl_df.values))

    def test_write_pm(self):
         # write experiment definition
        xl_df = pd.DataFrame({'constant' : [1,1], 
                                'exp_var1' : [1.1,1.2], 
                                'exp_var2' : [2.1,2.2]})
        design = 'lhs'
        self.db_test.write_experiment_parameters(self.scope_name, design, xl_df)
        
        # get experiment ids
        exp_with_ids = self.db_test.read_experiment_parameters(self.scope_name,design)
        exp_with_ids['pm_1'] = [4.0,5.0]
        exp_with_ids['pm_2'] = [6.0,7.0]
        
        # write performance measures
        self.db_test.write_experiment_measures(self.scope_name,SOURCE_IS_CORE_MODEL,exp_with_ids)
        xlm_readback = self.db_test.read_experiment_all(self.scope_name,design)
        pd.testing.assert_frame_equal(exp_with_ids, xlm_readback)

    def test_write_partial_pm(self):
         # write experiment definition
        xl_df = pd.DataFrame({'constant' : [1,1], 
                                'exp_var1' : [1.1,1.2], 
                                'exp_var2' : [2.1,2.2]})
        design = 'lhs'
        self.db_test.write_experiment_parameters(self.scope_name, design, xl_df)
        
        # get experiment ids
        exp_with_ids = self.db_test.read_experiment_parameters(self.scope_name,design)
        exp_with_ids['pm_1'] = [4.0,5.0]
        
        # write performance measures
        self.db_test.write_experiment_measures(self.scope_name,SOURCE_IS_CORE_MODEL,exp_with_ids)
        xlm_readback = self.db_test.read_experiment_all(self.scope_name,design)
        pd.testing.assert_frame_equal(exp_with_ids, xlm_readback)

    def test_write_experiment(self):
         # write experiment definition
        xlm_df = pd.DataFrame({'constant' : [1,1], 
                            'exp_var1' : [1.1,1.2], 
                            'exp_var2' : [2.1,2.2],
                            'pm_1'     : [4.0,5.0],
                            'pm_2'     : [6.0,7.0]})
        design = 'lhs'
        core_model = True
        self.db_test.write_experiment_all(self.scope_name, design, SOURCE_IS_CORE_MODEL, xlm_df)
        xlm_readback = self.db_test.read_experiment_all(self.scope_name,design)
        # index may not match
        self.assertTrue(np.array_equal(xlm_readback.values, xlm_df.values))   
    
    # set experiment without all variables defined
    def test_incomplete_experiment(self):
        xl_df = pd.DataFrame({'exp_var1' : [1]})
        design = 'lhs'
        with self.assertRaises(KeyError):
            self.db_test.write_experiment_parameters(self.scope_name, design, xl_df)

    # try to overwrite existing scope
    def test_scope_overwrite(self):
        with self.assertRaises(KeyError):
            self.db_test.write_scope(self.scope_name,
                                      self.sheet, 
                                      self.scp_xl, 
                                      self.scp_m) 

    # scope with invalid risk variables
    def test_scope_invalid_risk(self):
        with self.assertRaises(KeyError):
            self.db_test.write_scope('test2',
                                      self.sheet, 
                                      ['exp_var3'], 
                                      self.ex_m) 
        self.db_test.delete_scope('test2')
            
    # scope with invalid performance measures
    def test_scope_invalid_pm(self):
        with self.assertRaises(KeyError):
            self.db_test.write_scope('test2',
                                      self.sheet, 
                                      self.ex_xl,
                                      ['pm_3'])         
        self.db_test.delete_scope('test2')            

    # scope with invalid performance measures
    def test_scope_add_pm(self):
        
        # add new pm
        self.db_test.init_xlm([], [('pm_4','none')])
        self.db_test.add_scope_meas(self.scope_name,
                                   self.ex_m + ['pm_4'])
        
        
       
    
    # delete experiment
    

class TestDatabaseGZ():

    def test_read_db_gz(self):
        road_test_scope_file = emat.package_file('model', 'tests', 'road_test.yaml')
        with pytest.raises(FileNotFoundError):
            emat.Scope(emat.package_file('nope.yaml'))
        s = emat.Scope(road_test_scope_file)
        with pytest.raises(FileNotFoundError):
            emat.SQLiteDB(emat.package_file('nope.db.gz'))

        if not os.path.exists(emat.package_file("examples", "roadtest.db.gz")):
            db_w = emat.SQLiteDB(emat.package_file("examples", "roadtest.db.tmp"), initialize=True)
            s.store_scope(db_w)
            s.design_experiments(n_samples=110, random_seed=1234, db=db_w, design_name='lhs')
            from emat.model.core_python import Road_Capacity_Investment
            m_w = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db_w)
            m_w.run_experiments(design_name='lhs', db=db_w)
            db_w.conn.close()
            import gzip
            import shutil
            with open(emat.package_file("examples", "roadtest.db.tmp"), 'rb') as f_in:
                with gzip.open(emat.package_file("examples", "roadtest.db.gz"), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        db = emat.SQLiteDB(emat.package_file("examples", "roadtest.db.gz"))

        assert repr(db) == '<emat.SQLiteDB with scope "EMAT Road Test">'
        assert db.get_db_info()[:9] == 'SQLite @ '
        assert db.get_db_info()[-11:] == 'roadtest.db'

        assert db.read_scope_names() == ['EMAT Road Test']

        s1 = db.read_scope('EMAT Road Test')


        assert type(s1) == type(s)

        for k in ('_x_list', '_l_list', '_c_list', '_m_list', 'name', 'desc'):
            assert getattr(s,k) == getattr(s1,k), k

        assert s == s1

        experiments = db.read_experiment_all('EMAT Road Test', 'lhs')
        assert experiments.shape == (110, 20)
        assert list(experiments.columns) == [
            'free_flow_time',
            'initial_capacity',
            'alpha',
            'beta',
            'input_flow',
            'value_of_time',
            'unit_cost_expansion',
            'interest_rate',
            'yield_curve',
            'expand_capacity',
            'amortization_period',
            'debt_type',
            'interest_rate_lock',
            'no_build_travel_time',
            'build_travel_time',
            'time_savings',
            'value_of_time_savings',
            'net_benefits',
            'cost_of_capacity_expansion',
            'present_cost_expansion',
        ]

        from emat.model.core_python import Road_Capacity_Investment
        m = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db)
        assert m.metamodel_id == None

def test_multiple_connections():
    import tempfile
    with tempfile.TemporaryDirectory() as tempdir:
        tempdbfile = os.path.join(tempdir, "test_db_file.db")
        db_test = SQLiteDB(tempdbfile, initialize=True)

        road_test_scope_file = emat.package_file('model', 'tests', 'road_test.yaml')
        s = emat.Scope(road_test_scope_file)
        db_test.store_scope(s)

        assert db_test.read_scope_names() == ['EMAT Road Test']

        db_test2 = SQLiteDB(tempdbfile, initialize=False)
        with pytest.raises(KeyError):
            db_test2.store_scope(s)

        # Neither database is in a transaction
        assert not db_test.conn.in_transaction
        assert not db_test2.conn.in_transaction

        from emat.model.core_python import Road_Capacity_Investment
        m1 = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db_test)
        m2 = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db_test2)
        d1 = m1.design_experiments(n_samples=3, random_seed=1, design_name='d1')
        d2 = m2.design_experiments(n_samples=3, random_seed=2, design_name='d2')
        r1 = m1.run_experiments(design_name='d1')
        r2 = m2.run_experiments(design_name='d2')

        # Check each model can load the other's results
        pd.testing.assert_frame_equal(
            r1,
            m2.db.read_experiment_all(
                scope_name=s.name,
                design_name='d1',
                ensure_dtypes=True)[r1.columns],
        )
        pd.testing.assert_frame_equal(
            r2,
            m1.db.read_experiment_all(
                scope_name=s.name,
                design_name='d2',
                ensure_dtypes=True)[r2.columns],
        )

def test_duplicate_experiments():
    import emat.examples
    scope, db, model = emat.examples.road_test()
    design = model.design_experiments(n_samples=5)
    results = model.run_experiments(design)
    db.read_design_names(scope.name)
    design2 = model.design_experiments(n_samples=5)
    assert design2.design_name == 'lhs_2'
    assert design.design_name == 'lhs'
    from pandas.testing import assert_frame_equal
    assert_frame_equal(design, design2)
    assert db.read_experiment_all(scope.name, 'lhs').design_name == 'lhs'
    assert db.read_experiment_all(scope.name, 'lhs_2').design_name == 'lhs_2'
    assert_frame_equal(
        db.read_experiment_all(scope.name, 'lhs'),
        db.read_experiment_all(scope.name, 'lhs_2')
    )
    assert len(db.read_experiment_all(None, None)) == 5

def test_deduplicate_indexes():
    testing_df = pd.DataFrame(
        data=np.random.random([10, 5]),
        columns=['Aa', 'Bb', 'Cc', 'Dd', 'Ee'],
        index=np.arange(50, 60),
    )
    testing_df['Ee'] = (testing_df['Ee'] * 100).astype('int64')
    testing_df['Ff'] = testing_df['Ee'].astype(str)
    testing_df['Ff'] = 'str' + testing_df['Ff']
    testing_df.iloc[7:9, :] = testing_df.iloc[3:5, :].set_index(testing_df.index[7:9])
    x = testing_df.index.to_numpy()
    x[-5:] = -1
    testing_df.index = x
    from emat.util.deduplicate import reindex_duplicates
    r_df = reindex_duplicates(testing_df)
    assert all(r_df.index == [50, 51, 52, 53, 54, -1, -1, 53, 54, -1])
    np.testing.assert_array_equal(r_df.to_numpy(), testing_df.to_numpy())

def test_version_warning():
    from emat.exceptions import DatabaseVersionWarning
    with pytest.warns(DatabaseVersionWarning):
        db = emat.SQLiteDB("require_version_999.sqldb")


emat.package_file('model', 'tests', 'road_test.yaml')

if __name__ == '__main__':
    unittest.main()
