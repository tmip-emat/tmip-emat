import unittest
import os
import pytest

import emat
from emat.scope.scope import Scope
from emat.model.core_python import PythonCoreModel
from emat import config
from emat.model.core_python.core_python_examples import Dummy


class TestCoreModelMethods(unittest.TestCase):
    ''' 
        tests model and meta-model methods     
    '''
    corem_scope_file = emat.package_file("model","tests","core_model_test.yaml")
    scp = Scope(corem_scope_file)

    corem = PythonCoreModel(
        function=Dummy(),
        configuration={
            'archive_path':config.get_subdir('test_directory', 'core_dummy_archive')
        },
        scope=scp,
    )

# =============================================================================
#     
#      Core model tests
#     
# =============================================================================

    def test_create_scenario(self):
        exp_def = {'Auto IVTT Sensitivity' : 0.9122442817924445}
        self.corem.setup(exp_def)

    @pytest.mark.skip(reason="TODO")
    def test_set_invalid_exp_variable(self):
        exp_def = {'unsupported' : 1}
        with self.assertRaises(KeyError):
            self.corem.setup(exp_def)
    
    @pytest.mark.skip(reason="TODO")
    def test_post_process(self):
        exp_def = {'Land Use - CBD Focus' : 1}
        pm = ['Region-wide VMT']
        self.corem.post_process(exp_def, pm)
    
    @pytest.mark.skip(reason="TODO")
    def test_archive_model(self):
        exp_id = 1
        archive_path = self.corem.get_exp_archive_path(self.scp.scp_name, exp_id)
        self.corem.archive(archive_path)
        
    @pytest.mark.skip(reason="TODO")
    def atest_hook_presence(self):
        ''' confirm that hooks are present for all performance measures, exp vars'''
        # TODO
        # set experiment variables
        
        # post process
        
        # import performance measure
 
    @pytest.mark.skip(reason="TODO")
    def test_pm_import(self):
        pm = ['Peak Walk-to-transit Boarding', 'Total LRT Boardings',
              "PM Trip Time (minutes)", "Daily Transit Share",
              "Households within 30 min of CBD",
              "Number of Home-based work tours taking <= 45 minutes via transit",
              "Downtown to Airport Travel Time", 'OD Volume District 1 to 1',
              '190 Daily VHT']
        pm_vals = self.corem.import_perf_meas(pm)
        
        expected_pm = {'Peak Walk-to-transit Boarding': 56247.88692999999, 
                       'Total LRT Boardings': 24784.475588, 
                       "PM Trip Time (minutes)": 15.652833,
                       "Daily Transit Share" : 0.019905000000000003,
                       "Households within 30 min of CBD" : 379894,
                       "Number of Home-based work tours taking <= 45 minutes via transit" : 322069.75,
                       "Downtown to Airport Travel Time": 14.734342999999999,
                       'OD Volume District 1 to 1':55642.74609400001, 
                       '190 Daily VHT':272612.499025}
        self.assertEqual(expected_pm, pm_vals)

        
if __name__ == '__main__':
    unittest.main()

