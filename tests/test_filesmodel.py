
import unittest
import os
from pytest import approx
import pandas as pd
import numpy as np
import emat
from emat.scope.scope import Scope
from emat.database.sqlite.sqlite_db import SQLiteDB




class TestCoreFilesMethods(unittest.TestCase):

    def test_getters(self):

        from emat.model.core_files.parsers import loc, loc_mean, loc_sum
        from emat.model.core_files.parsers import iloc, iloc_mean, iloc_sum

        zz = pd.DataFrame(
            np.arange(50).reshape(5, 10),
            index=[f'row{i}' for i in range(1, 6)],
            columns=[f'col{i}' for i in range(1, 11)],
        )

        a = loc['row2','col8']
        assert repr(a) == "loc['row2','col8']"

        h = iloc[3,3]
        assert repr(h) == 'iloc[3,3]'

        i = iloc_mean[:2,7:]
        assert repr(i) == 'iloc_mean[:2,7:]'

        k = iloc_sum[2,:]
        assert repr(k) == 'iloc_sum[2,:]'

        j = h + k
        assert repr(j) == 'iloc[3,3] + iloc_sum[2,:]'

        assert a(zz) == 17
        assert h(zz) == 33
        assert i(zz) == approx(13)
        assert k(zz) == 245
        assert j(zz) == 278


    def test_load_archived_gbnrtc(self):
        import emat.examples
        s, db, m = emat.examples.gbnrtc()
        m.archive_path = emat.examples.package_file("examples", "gbnrtc", "archive")
        assert os.path.exists(m.get_experiment_archive_path(1, run_id=False))
        measures = m.load_archived_measures(1)
        correct_1 = {
            'Peak Walk-to-transit Boarding': 34281.205786,
            'Off-Peak Walk-to-transit Boarding': 32321.752577999996,
            'Peak Drive-to-transit Boarding': 4650.044377,
            'Off-Peak Drive-to-transit Boarding': 3896.5493810000003,
            'Total Transit Boardings': 75149.55212200001,
            'Peak Walk-to-transit LRT Boarding': 9008.224461,
            'Off-Peak Walk-to-transit LRT Boarding': 10645.432359,
            'Peak Drive-to-transit LRT Boarding': 2761.200268,
            'Off-Peak Drive-to-transit LRT Boarding': 2406.542344,
            'Total LRT Boardings': 24821.399432,
            'Region-wide VMT': 25113613.736528996,
            'Total Auto VMT': 22511322.163062,
            'Total Truck VMT': 2602291.573469,
            'Interstate + Expressway + Ramp/Connector VMT': 10305109.628398,
            'Major and Minor Arterials VMT': 10475969.845537,
            'AM Trip Time (minutes)': 14.654542999999999,
            'AM Trip Length (miles)': 7.548014,
            'PM Trip Time (minutes)': 15.324133999999999,
            'PM Trip Length (miles)': 8.261152000000001,
            'Peak Transit Share': 0.014477,
            'Peak NonMotorized Share': 0.060296,
            'Off-Peak Transit Share': 0.011423,
            'Off-Peak NonMotorized Share': 0.056386,
            'Daily Transit Share': 0.012819999999999998,
            'Daily NonMotorized Share': 0.058175, 'Households within 30 min of CBD': 399597,
            'Number of Home-based work tours taking <= 45 minutes via transit': 340958.875,
            'Downtown to Airport Travel Time': 14.443295999999998,
            'OD Volume District 1 to 1': 2,
            'OD Volume District 1 to 2': 27850.572266000003,
            'OD Volume District 1 to 3': 93799.382813,
            'OD Volume District 1 to 4': 23470.341797,
            'OD Volume District 1 to 5': 20363.416016,
            'OD Volume District 1 to 6': 2140.624268,
            'OD Volume District 1 to 7': 21603.265625,
            'OD Volume District 1 to 8': 1890.3181149999998,
            'OD Volume District 1 to 9': 10427.630859,
            'OD Volume District 1 to 10': 4448.775879,
            'Kensington Daily VMT': 206937.015614,
            'Kensington Daily VHT': 239242.35552800001,
            'Kensington_OB PM VMT': 26562.351204,
            'Kensington_OB PM VHT': 31363.340938,
            'Kensington_IB AM VMT': 23796.174231999998,
            'Kensington_IB AM VHT': 30434.206062999998,
            '190 Daily VMT': 282469.874037,
            '190 Daily VHT': 300633.50829,
            '190_OB Daily VMT': 36483.463967,
            '190_OB Daily VHT': 45783.789093,
            '190_IB Daily VMT': 30282.776539,
            '190_IB Daily VHT': 33375.415786000005,
            '33_west Daily VMT': 45402.79583,
            '33_west Daily VHT': 57478.416767999995,
            'I90_south Daily VMT': 147224.53766099998,
            'I90_south Daily VHT': 153543.832728,
        }
        assert set(correct_1.keys()).issubset(measures.keys())
        assert {k: measures[k] for k in correct_1.keys()} == approx(correct_1)

def test_files_with_broken_scope():
    try:
        import core_files_demo
    except:
        import pytest
        pytest.skip("core_files_demo not installed")
    fx = core_files_demo.RoadTestFileModel(
        scope_file=emat.package_file('model', 'tests', 'road_test_corrupt2.yaml')
    )
    design = fx.design_experiments(n_samples=2)
    result = fx.run_experiments(design)
    assert result['bogus_measure'].isna().all()

if __name__ == '__main__':
    unittest.main()
