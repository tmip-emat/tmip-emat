
import numpy as np
import pandas as pd
import os

import emat
from emat.model import PythonCoreModel
from emat.analysis import feature_scores

def stable_df(filename, df):
	if not os.path.exists(filename):
		df.to_pickle(filename)
	return pd.testing.assert_frame_equal(df, pd.read_pickle(filename))

def _Road_Capacity_Investment_with_Bogus_Output(**kwargs):
	result = Road_Capacity_Investment(**kwargs)
	result['bogus_output'] = np.nan
	return result

from emat.model.core_python import Road_Capacity_Investment

def test_feature_scoring():
	road_scope = emat.Scope(emat.package_file('model','tests','road_test.yaml'))
	road_test = PythonCoreModel(Road_Capacity_Investment, scope=road_scope)
	road_test_design = road_test.design_experiments(n_samples=5000, sampler='lhs')
	road_test_results = road_test.run_experiments(design=road_test_design)
	fs = feature_scores(road_scope, road_test_results, random_state=123)
	assert isinstance(fs, pd.io.formats.style.Styler)
	stable_df("./road_test_feature_scores_1.pkl.gz", fs.data)

def test_feature_scoring_with_nan():
	road_scope = emat.Scope(emat.package_file('model','tests','road_test_bogus.yaml'))
	road_test = PythonCoreModel(_Road_Capacity_Investment_with_Bogus_Output, scope=road_scope)
	road_test_design = road_test.design_experiments(n_samples=5000, sampler='lhs')
	road_test_results = road_test.run_experiments(design=road_test_design)
	fs = feature_scores(road_scope, road_test_results, random_state=234)
	assert isinstance(fs, pd.io.formats.style.Styler)
	stable_df("./road_test_feature_scores_bogus_1.pkl.gz", fs.data)
