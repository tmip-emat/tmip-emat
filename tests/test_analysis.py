import os

import numpy as np
import pandas as pd
from pytest import approx

import emat
from emat.analysis import feature_scores
from emat.model import PythonCoreModel


def stable_df(filename, df):
    if not os.path.exists(filename):
        df.to_pickle(filename)
    return pd.testing.assert_frame_equal(df, pd.read_pickle(filename))


def _Road_Capacity_Investment_with_Bogus_Output(**kwargs):
    result = Road_Capacity_Investment(**kwargs)
    result["bogus_output"] = np.nan
    return result


from emat.model.core_python import Road_Capacity_Investment


def test_feature_scoring_and_prim():
    road_scope = emat.Scope(emat.package_file("model", "tests", "road_test.yaml"))
    road_test = PythonCoreModel(Road_Capacity_Investment, scope=road_scope)
    road_test_design = road_test.design_experiments(n_samples=5000, sampler="lhs")
    road_test_results = road_test.run_experiments(design=road_test_design)
    fs = feature_scores(road_scope, road_test_results, random_state=123)
    assert isinstance(fs, pd.io.formats.style.Styler)
    stable_df("./road_test_feature_scores_1.pkl.gz", fs.data)

    prim1 = road_test_results.prim(target="net_benefits >= 0")
    pbox1 = prim1.find_box()

    assert pbox1._cur_box == 64
    ts1 = prim1.tradeoff_selector()
    assert len(ts1.data) == 1
    assert ts1.data[0]["x"] == approx(
        np.asarray(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.99928315,
                0.99856631,
                0.99784946,
                0.99569892,
                0.99283154,
                0.98924731,
                0.98351254,
                0.97921147,
                0.97491039,
                0.96702509,
                0.95555556,
                0.94982079,
                0.94336918,
                0.92903226,
                0.91182796,
                0.89749104,
                0.87598566,
                0.85304659,
                0.83942652,
                0.83225806,
                0.82078853,
                0.79713262,
                0.77706093,
                0.76415771,
                0.75483871,
                0.74480287,
                0.73261649,
                0.71827957,
                0.70394265,
                0.68100358,
                0.65663082,
                0.63225806,
                0.61003584,
                0.59569892,
                0.57992832,
                0.55770609,
                0.54193548,
                0.52759857,
                0.51111111,
                0.49892473,
                0.48960573,
                0.4781362,
                0.45878136,
                0.44229391,
                0.42365591,
                0.409319,
                0.39498208,
                0.38064516,
                0.36487455,
                0.34767025,
                0.33261649,
                0.31756272,
                0.30322581,
                0.28888889,
                0.27741935,
                0.26379928,
                0.25089606,
                0.23942652,
                0.22795699,
                0.2172043,
            ]
        )
    )
    pbox1.select(40)
    assert pbox1._cur_box == 40
    assert ts1.data[0]["marker"]["symbol"] == approx(
        np.asarray(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
    )

    ebox1_40 = pbox1.to_emat_box()
    assert ebox1_40.coverage == approx(0.5577060931899641)
    assert ebox1_40.density == approx(0.8356605800214822)
    assert ebox1_40.mass == approx(0.1862)

    assert ebox1_40.thresholds["beta"].lowerbound == approx(3.597806324946271)
    assert ebox1_40.thresholds["beta"].upperbound is None
    assert ebox1_40.thresholds["input_flow"].lowerbound == 125
    assert ebox1_40.thresholds["input_flow"].upperbound is None
    assert ebox1_40.thresholds["value_of_time"].lowerbound == approx(0.07705746291056698)
    assert ebox1_40.thresholds["value_of_time"].upperbound is None
    assert ebox1_40.thresholds["expand_capacity"].lowerbound is None
    assert ebox1_40.thresholds["expand_capacity"].upperbound == approx(95.01870815358643)

    pbox1.splom()
    pbox1.hmm()


def test_feature_scoring_with_nan():
    road_scope = emat.Scope(emat.package_file("model", "tests", "road_test_bogus.yaml"))
    road_test = PythonCoreModel(
        _Road_Capacity_Investment_with_Bogus_Output, scope=road_scope
    )
    road_test_design = road_test.design_experiments(n_samples=5000, sampler="lhs")
    road_test_results = road_test.run_experiments(design=road_test_design)
    fs = feature_scores(road_scope, road_test_results, random_state=234)
    assert isinstance(fs, pd.io.formats.style.Styler)
    stable_df("./road_test_feature_scores_bogus_1.pkl.gz", fs.data)
