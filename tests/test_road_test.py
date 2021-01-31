import numpy as np
import pandas as pd
import pytest
from pytest import approx, raises
from sklearn.utils import Bunch
import emat.examples




def test_basic_road_test_example(dataframe_regression):
	scope, db, model = emat.examples.road_test()
	design = model.design_experiments()
	result = model.run_experiments(design)
	dataframe_regression.check(
		pd.DataFrame(result),
		basename='test_basic_road_test_example_first_result',
	)

	ref_result = model.run_reference_experiment()
	dataframe_regression.check(
		pd.DataFrame(ref_result),
		basename='test_basic_road_test_example_ref_result',
	)

	# re-run reference model, should get same experiment number
	ref_result2 = model.run_reference_experiment()
	pd.testing.assert_frame_equal(ref_result, ref_result2)


if __name__ == '__main__':

	class Noop:
		def check(self, *args, **kwargs):
			pass

	test_basic_road_test_example(Noop())