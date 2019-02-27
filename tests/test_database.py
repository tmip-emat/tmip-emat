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
        self.assertTrue(exp_with_ids.equals(xlm_readback))

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
        self.assertTrue(exp_with_ids.equals(xlm_readback))

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



emat.package_file('model', 'tests', 'road_test.yaml')

if __name__ == '__main__':
    unittest.main()
