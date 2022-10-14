# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pytest
from sklearn.utils import Bunch

from emat.database.sqlite.sqlite_db import SQLiteDB
from emat._pkg_constants import *

import emat
from emat import config


scope_yaml = """
    scope:
        name: test-scope
    inputs:
        constant:
            ptype: constant
            dtype: float
            default: 1
        exp_var1:
            ptype: lever
            dtype: float
            default: 1
            min: 0
            max: 2
        exp_var2:
            ptype: uncertainty
            dtype: float
            default: 1
            min: 0
            max: 2
    outputs:
        pm_1:
            shortname: one
            kind: info
            tags:
                - odd
                - small
        pm_2:
            kind: info
            tags:
                - even
                - small
        pm_3:
            shortname: three
            desc: Third Measure
            kind: info
            formula: pm_1 + pm_2
            tags:
                - odd
        pm_4:
            shortname: four
            desc: Fourth Measure
            kind: info
            formula: pm_3 + pm_2
            tags:
                - even
    """


@pytest.fixture()
def db_setup():
    db_test = SQLiteDB(config.get("test_db_filename", ":memory:"), initialize=True)

    # load experiment variables and performance measures
    scp_xl = [("constant", "constant"), ("exp_var1", "risk"), ("exp_var2", "strategy")]
    scp_m = [("pm_1", "none"), ("pm_2", "ln")]

    db_test.init_xlm(scp_xl, scp_m)

    # create emat scope
    scope_name = "test"
    sheet = "emat_scope1.yaml"
    ex_xl = ["constant", "exp_var1", "exp_var2"]
    ex_m = ["pm_1", "pm_2"]
    db_test.delete_scope(scope_name)

    scope = emat.Scope(sheet, scope_yaml)

    db_test._write_scope(
        scope_name, sheet, ex_xl, ex_m, scope,
    )
    yield Bunch(
        db_test=db_test,
        scope_name=scope_name,
        sheet=sheet,
        scp_xl=scp_xl,
        scp_m=scp_m,
        ex_m=ex_m,
        ex_xl=ex_xl,
    )
    db_test.delete_scope(scope_name)


def test_delete_experiment(db_setup):

    # write experiment definition
    xl_df = pd.DataFrame(
        {"constant": [1, 1], "exp_var1": [1.1, 1.2], "exp_var2": [2.1, 2.2]}
    )
    design = "lhs"
    db_setup.db_test.write_experiment_parameters(db_setup.scope_name, design, xl_df)
    db_setup.db_test.delete_experiments(db_setup.scope_name, design)

    xl_readback = db_setup.db_test.read_experiment_parameters(
        db_setup.scope_name, design
    )
    # note - indexes may not match
    assert xl_readback.empty


def test_create_experiment(db_setup):
    # write experiment definition
    xl_df = pd.DataFrame(
        {"constant": [1, 1], "exp_var1": [1.1, 1.2], "exp_var2": [2.1, 2.2]}
    )
    design = "lhs"
    db_setup.db_test.write_experiment_parameters(db_setup.scope_name, design, xl_df)

    xl_readback = db_setup.db_test.read_experiment_parameters(
        db_setup.scope_name, design
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(xl_readback).reset_index(drop=True),
        xl_df.reset_index(drop=True),
        check_frame_type=False,
        check_column_type=False,
    )


def test_create_specific_experiments(db_setup):
    # write experiment definition
    xl_df = pd.DataFrame(
        {"constant": [1, 1], "exp_var1": [1.1, 1.2], "exp_var2": [2.1, 2.2]},
        index=pd.Index([12345, 54321], name="experiment"),
    )
    design = "lhs"
    db_setup.db_test.write_experiment_parameters(
        db_setup.scope_name, design, xl_df, force_ids=True
    )
    xl_readback = db_setup.db_test.read_experiment_parameters(
        db_setup.scope_name, design
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(xl_readback), xl_df, check_frame_type=False
    )

    # adding different experiment parameters with same id
    xl_df2 = pd.DataFrame(
        {"constant": [1, 1], "exp_var1": [3.3, 3.2], "exp_var2": [2.1, 2.2]},
        index=pd.Index([12345, 54321], name="experiment"),
    )
    from sqlite3 import IntegrityError

    with pytest.raises(IntegrityError):
        db_setup.db_test.write_experiment_parameters(
            db_setup.scope_name, design, xl_df2, force_ids=True,
        )

    # adding same experiment parameters with different id
    xl_df.index = xl_df.index + 1
    with pytest.raises(ValueError, match="cannot change experiment id .*"):
        db_setup.db_test.write_experiment_parameters(
            db_setup.scope_name, design, xl_df, force_ids=True
        )


def test_write_measures(db_setup):
    # write experiment definition
    xl_df = pd.DataFrame(
        {"constant": [1, 1], "exp_var1": [1.1, 1.2], "exp_var2": [2.1, 2.2]}
    )
    design = "lhs"
    db_setup.db_test.write_experiment_parameters(db_setup.scope_name, design, xl_df)

    # get experiment ids
    exp_with_ids = db_setup.db_test.read_experiment_parameters(
        db_setup.scope_name, design
    )
    exp_with_ids["pm_1"] = [4.4, 5.5]
    exp_with_ids["pm_2"] = [6.6, 7.7]

    # write performance measures
    db_setup.db_test.write_experiment_measures(
        db_setup.scope_name, SOURCE_IS_CORE_MODEL, exp_with_ids
    )
    xlm_readback = db_setup.db_test.read_experiment_all(db_setup.scope_name, design)
    pd.testing.assert_frame_equal(exp_with_ids, xlm_readback)


def test_write_measure_runs(db_setup):
    # write experiment definition
    xl_df = pd.DataFrame(
        {"constant": [1, 1], "exp_var1": [1.1, 1.2], "exp_var2": [2.1, 2.2]}
    )
    design = "lhs"
    db_setup.db_test.write_experiment_parameters(db_setup.scope_name, design, xl_df)

    # get experiment ids
    exp_with_ids = db_setup.db_test.read_experiment_parameters(
        db_setup.scope_name, design
    )
    exp_with_ids["pm_1"] = [4.4, 5.5]
    exp_with_ids["pm_2"] = [6.6, 7.7]

    # write performance measures
    db_setup.db_test.write_experiment_measures(
        db_setup.scope_name, SOURCE_IS_CORE_MODEL, exp_with_ids
    )

    import time
    time.sleep(1)

    # second run, different results
    exp_with_ids2 = exp_with_ids.copy()
    exp_with_ids2["pm_1"] = [14.4, 15.5]
    exp_with_ids2["pm_2"] = [16.6, 17.7]
    # write performance measures
    db_setup.db_test.write_experiment_measures(
        db_setup.scope_name, SOURCE_IS_CORE_MODEL, exp_with_ids2
    )

    xlm_readback2 = db_setup.db_test.read_experiment_all(
        db_setup.scope_name, design,
    )
    pd.testing.assert_frame_equal(exp_with_ids2, xlm_readback2)

    xlm_readback = db_setup.db_test.read_experiment_all(
        db_setup.scope_name, design, runs='all', with_run_ids=False,
    )
    pd.testing.assert_frame_equal(
        pd.concat([exp_with_ids, exp_with_ids2]).sort_values('pm_1'),
        xlm_readback.sort_values('pm_1'),
    )

    xlm_readback_mean = db_setup.db_test.read_experiment_all(
        db_setup.scope_name, design, runs='valid_mean',
    )
    pd.testing.assert_frame_equal(
        ((exp_with_ids+exp_with_ids2)/2).sort_values('pm_1'),
        xlm_readback_mean.sort_values('pm_1'),
        check_dtype=False,
    )


def test_write_partial_measures(db_setup):
    # assert self.db_test.read_scope(self.scope_name) is not None

    # write experiment definition
    xl_df = pd.DataFrame(
        {"constant": [1, 1], "exp_var1": [1.1, 1.2], "exp_var2": [2.1, 2.2]}
    )
    design = "lhs"
    db_setup.db_test.write_experiment_parameters(db_setup.scope_name, design, xl_df)

    # get experiment ids
    exp_with_ids = db_setup.db_test.read_experiment_parameters(
        db_setup.scope_name, design
    )
    exp_with_ids["pm_1"] = [4.4, 5.5]

    # write performance measures
    db_setup.db_test.write_experiment_measures(
        db_setup.scope_name, SOURCE_IS_CORE_MODEL, exp_with_ids
    )
    xlm_readback = db_setup.db_test.read_experiment_all(
        db_setup.scope_name,
        design,
        formulas=False, # must be false because partial measures only
    )
    pd.testing.assert_frame_equal(exp_with_ids, xlm_readback)
    with pytest.raises(Exception):
        db_setup.db_test.read_experiment_all(
            db_setup.scope_name,
            design,
            formulas=True,
        )

def test_write_experiment(db_setup):
    # write experiment definition
    xlm_df = pd.DataFrame(
        {
            "constant": [1, 1],
            "exp_var1": [1.1, 1.2],
            "exp_var2": [2.1, 2.2],
            "pm_1": [4.0, 5.0],
            "pm_2": [6.0, 7.0],
        }
    )
    design = "lhs"
    core_model = True
    db_setup.db_test.write_experiment_all(
        db_setup.scope_name, design, SOURCE_IS_CORE_MODEL, xlm_df
    )
    xlm_readback = db_setup.db_test.read_experiment_all(db_setup.scope_name, design)
    # index may not match
    assert np.array_equal(xlm_readback.values, xlm_df.values)


# set experiment without all variables defined
def test_incomplete_experiment(db_setup):
    xl_df = pd.DataFrame({"exp_var1": [1]})
    design = "lhs"
    with pytest.raises(KeyError):
        db_setup.db_test.write_experiment_parameters(db_setup.scope_name, design, xl_df)


# try to overwrite existing scope
def test_scope_overwrite(db_setup):
    with pytest.raises(KeyError):
        db_setup.db_test._write_scope(
            db_setup.scope_name, db_setup.sheet, db_setup.scp_xl, db_setup.scp_m,
        )

    # scope with invalid risk variables


def test_scope_invalid_risk(db_setup):
    with pytest.raises(KeyError):
        db_setup.db_test._write_scope(
            "test2", db_setup.sheet, ["exp_var3"], db_setup.ex_m,
        )
    db_setup.db_test.delete_scope("test2")


# scope with invalid performance measures
def test_scope_invalid_pm(db_setup):
    with pytest.raises(KeyError):
        db_setup.db_test._write_scope(
            "test2", db_setup.sheet, db_setup.ex_xl, ["pm_3"],
        )
    db_setup.db_test.delete_scope("test2")


def test_database_scope_updating():
    scope = emat.Scope("fake_filename.yaml", scope_yaml)
    db = emat.SQLiteDB()
    db.store_scope(scope)
    assert db.read_scope(scope.name) == scope
    scope.add_measure("plus1")
    db.update_scope(scope)
    assert db.read_scope(scope.name) == scope
    assert len(scope.get_measures()) == 5
    scope.add_measure("plus2", db=db)
    assert db.read_scope(scope.name) == scope
    assert len(scope.get_measures()) == 6


def test_read_db_gz():
    road_test_scope_file = emat.package_file("model", "tests", "road_test.yaml")
    with pytest.raises(FileNotFoundError):
        emat.Scope(emat.package_file("nope.yaml"))
    s = emat.Scope(road_test_scope_file)
    with pytest.raises(FileNotFoundError):
        emat.SQLiteDB(emat.package_file("nope.db.gz"))

    if not os.path.exists(emat.package_file("examples", "roadtest.db.gz")):
        db_w = emat.SQLiteDB(
            emat.package_file("examples", "roadtest.db.tmp"), initialize=True
        )
        s.store_scope(db_w)
        s.design_experiments(
            n_samples=110, random_seed=1234, db=db_w, design_name="lhs"
        )
        from emat.model.core_python import Road_Capacity_Investment

        m_w = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db_w)
        m_w.run_experiments(design_name="lhs", db=db_w)
        db_w.conn.close()
        import gzip
        import shutil

        with open(emat.package_file("examples", "roadtest.db.tmp"), "rb") as f_in:
            with gzip.open(
                emat.package_file("examples", "roadtest.db.gz"), "wb"
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)

    db = emat.SQLiteDB(emat.package_file("examples", "roadtest.db.gz"))

    assert repr(db) == '<emat.SQLiteDB with scope "EMAT Road Test">'
    assert db.get_db_info()[:9] == "SQLite @ "
    assert db.get_db_info()[-11:] == "roadtest.db"

    assert db.read_scope_names() == ["EMAT Road Test"]

    s1 = db.read_scope("EMAT Road Test")

    assert type(s1) == type(s)

    for k in ("_x_list", "_l_list", "_c_list", "_m_list", "name", "desc"):
        left = getattr(s, k)
        right = getattr(s1, k)
        if isinstance(left, (list, tuple)):
            for l,r in zip(left,right):
                try:
                    explain = l.explain_neq(r)
                except AttributeError:
                    explain = "no explanation"
                assert l == r, f"{k}: {explain}"
        else:
            assert getattr(s, k) == getattr(s1, k), k

    assert s == s1

    experiments = db.read_experiment_all("EMAT Road Test", "lhs")
    assert experiments.shape == (110, 20)
    assert list(experiments.columns) == [
        "free_flow_time",
        "initial_capacity",
        "alpha",
        "beta",
        "input_flow",
        "value_of_time",
        "unit_cost_expansion",
        "interest_rate",
        "yield_curve",
        "expand_capacity",
        "amortization_period",
        "debt_type",
        "interest_rate_lock",
        "no_build_travel_time",
        "build_travel_time",
        "time_savings",
        "value_of_time_savings",
        "net_benefits",
        "cost_of_capacity_expansion",
        "present_cost_expansion",
    ]

    from emat.model.core_python import Road_Capacity_Investment

    m = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db)
    assert m.metamodel_id == None


def test_multiple_connections():
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        tempdbfile = os.path.join(tempdir, "test_db_file.db")
        db_test = SQLiteDB(tempdbfile, initialize=True)

        road_test_scope_file = emat.package_file("model", "tests", "road_test.yaml")
        s = emat.Scope(road_test_scope_file)
        db_test.store_scope(s)

        assert db_test.read_scope_names() == ["EMAT Road Test"]

        db_test2 = SQLiteDB(tempdbfile, initialize=False)
        with pytest.raises(KeyError):
            db_test2.store_scope(s)

        # Neither database is in a transaction
        assert not db_test.conn.in_transaction
        assert not db_test2.conn.in_transaction

        from emat.model.core_python import Road_Capacity_Investment

        m1 = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db_test)
        m2 = emat.PythonCoreModel(Road_Capacity_Investment, scope=s, db=db_test2)
        d1 = m1.design_experiments(n_samples=3, random_seed=1, design_name="d1")
        d2 = m2.design_experiments(n_samples=3, random_seed=2, design_name="d2")
        r1 = m1.run_experiments(design_name="d1")
        r2 = m2.run_experiments(design_name="d2")

        # Check each model can load the other's results
        pd.testing.assert_frame_equal(
            r1,
            m2.db.read_experiment_all(
                scope_name=s.name, design_name="d1", ensure_dtypes=True
            )[r1.columns],
        )
        pd.testing.assert_frame_equal(
            r2,
            m1.db.read_experiment_all(
                scope_name=s.name, design_name="d2", ensure_dtypes=True
            )[r2.columns],
        )


def test_duplicate_experiments():
    import emat.examples

    scope, db, model = emat.examples.road_test()
    design = model.design_experiments(n_samples=5)
    results = model.run_experiments(design)
    db.read_design_names(scope.name)
    design2 = model.design_experiments(n_samples=5)
    assert design2.design_name == "lhs_2"
    assert design.design_name == "lhs"
    from pandas.testing import assert_frame_equal

    assert_frame_equal(design, design2)
    assert db.read_experiment_all(scope.name, "lhs").design_name == "lhs"
    assert db.read_experiment_all(scope.name, "lhs_2").design_name == "lhs_2"
    assert_frame_equal(
        db.read_experiment_all(scope.name, "lhs"),
        db.read_experiment_all(scope.name, "lhs_2"),
    )
    assert len(db.read_experiment_all(None, None)) == 5


def test_deduplicate_indexes():
    testing_df = pd.DataFrame(
        data=np.random.random([10, 5]),
        columns=["Aa", "Bb", "Cc", "Dd", "Ee"],
        index=np.arange(50, 60),
    )
    testing_df["Ee"] = (testing_df["Ee"] * 100).astype("int64")
    testing_df["Ff"] = testing_df["Ee"].astype(str)
    testing_df["Ff"] = "str" + testing_df["Ff"]
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

    print(os.getcwd())
    test_dir = os.path.dirname(__file__)
    db_file = os.path.join(test_dir, "require_version_999.sqldb")
    assert os.path.exists(db_file)
    with pytest.warns(DatabaseVersionWarning):
        db = emat.SQLiteDB(db_file)


@pytest.mark.skip
def test_database_merging():
    import emat

    road_test_scope_file = emat.package_file("model", "tests", "road_test.yaml")

    road_scope = emat.Scope(road_test_scope_file)
    emat_db = emat.SQLiteDB()
    road_scope.store_scope(emat_db)
    assert emat_db.read_scope_names() == ["EMAT Road Test"]

    from emat.experiment.experimental_design import design_experiments

    design = design_experiments(
        road_scope, db=emat_db, n_samples_per_factor=10, sampler="lhs"
    )
    large_design = design_experiments(
        road_scope, db=emat_db, n_samples=500, sampler="lhs", design_name="lhs_large"
    )

    assert emat_db.read_design_names("EMAT Road Test") == ["lhs", "lhs_large"]

    from emat.model.core_python import PythonCoreModel, Road_Capacity_Investment

    m = PythonCoreModel(Road_Capacity_Investment, scope=road_scope, db=emat_db)

    lhs_results = m.run_experiments(design_name="lhs")

    lhs_large_results = m.run_experiments(design_name="lhs_large")

    reload_results = m.read_experiments(design_name="lhs")

    pd.testing.assert_frame_equal(
        reload_results, lhs_results, check_like=True,
    )

    lhs_params = m.read_experiment_parameters(design_name="lhs")
    assert len(lhs_params) == 110
    assert len(lhs_params.columns) == 13

    lhs_outcomes = m.read_experiment_measures(design_name="lhs")
    assert len(lhs_outcomes) == 110
    assert len(lhs_outcomes.columns) == 7

    mm = m.create_metamodel_from_design("lhs")

    assert mm.metamodel_id == 1

    assert isinstance(mm.function, emat.MetaModel)

    design2 = design_experiments(
        road_scope, db=emat_db, n_samples_per_factor=10, sampler="lhs", random_seed=2
    )

    design2_results = mm.run_experiments(design2)

    assert len(design2_results) == 110

    assert len(design2_results.columns) == 20

    assert emat_db.read_design_names(None) == ["lhs", "lhs_2", "lhs_large"]

    check = emat_db.read_experiment_measures(None, "lhs_2")
    assert len(check) == 110
    assert len(check.columns) == 7

    assert emat_db.read_experiment_measure_sources(None, "lhs_2") == [1]

    m.allow_short_circuit = False
    design2_results0 = m.run_experiments(design2.iloc[:5])

    assert len(design2_results0) == 5
    assert len(design2_results0.columns) == 20

    with pytest.raises(ValueError):
        # now there are two sources of some measures
        emat_db.read_experiment_measures(None, "lhs_2")

    assert set(emat_db.read_experiment_measure_sources(None, "lhs_2")) == {0, 1}

    check = emat_db.read_experiment_measures(None, "lhs_2", source=0)
    assert len(check) == 5

    check = emat_db.read_experiment_measures(None, "lhs_2", source=1)
    assert len(check) == 110

    import emat.examples

    s2, db2, m2 = emat.examples.road_test()

    # write the design for lhs_2 into a different database.
    # it ends up giving different experient id's to these, which is fine.
    db2.write_experiment_parameters(
        None, "lhs_2", emat_db.read_experiment_parameters(None, "lhs_2")
    )

    check = db2.read_experiment_parameters(None, "lhs_2",)
    assert len(check) == 110
    assert len(check.columns) == 13

    pd.testing.assert_frame_equal(
        design2.reset_index(drop=True), check.reset_index(drop=True), check_like=True,
    )

    design2_results2 = m2.run_experiments("lhs_2")

    check = emat_db.read_experiment_measures(None, "lhs_2", source=0)
    assert len(check) == 5
    assert len(check.columns) == 7

    check = emat_db.read_experiment_measures(None, "lhs_2", runs="valid")
    assert len(check) == 115

    emat_db.merge_database(db2)

    check = emat_db.read_experiment_measures(None, "lhs_2", source=0)
    assert len(check) == 110
    assert len(check.columns) == 7

    check = emat_db.read_experiment_measures(None, "lhs_2", runs="valid")
    assert len(check) == 225


def test_update_old_database():
    import shutil

    test_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(test_dir, "old-format-database.sqlitedb"),
        os.path.join(test_dir, "old-format-database-copy.sqlitedb"),
    )
    old = emat.SQLiteDB(os.path.join(test_dir, "old-format-database-copy.sqlitedb"))
    assert old.read_experiment_parameters(None, "lhs_1").shape == (100, 13)
    assert old.read_experiment_measures(None, "lhs_1").shape == (50, 7)
    old.conn.close()
    os.remove(os.path.join(test_dir, "old-format-database-copy.sqlitedb"))

import tempfile
import yaml

def test_database_walkthrough(data_regression, dataframe_regression):

    # import os
    # import numpy as np
    # import pandas as pd
    # import seaborn;
    # seaborn.set_theme()
    # import plotly.io;
    # plotly.io.templates.default = "seaborn"
    # import emat
    # import yaml
    # from emat.util.show_dir import show_dir
    # from emat.analysis import display_experiments
    # emat.versions()

    # For this walkthrough of database features, we'll work in a temporary directory.
    # (In real projects you'll likely want to save your data somewhere less ephemeral,
    # so don't just copy this tempfile code into your work.)

    tempdir = tempfile.TemporaryDirectory()
    os.chdir(tempdir.name)

    # We begin our example by populating a database with some experimental data, by creating and
    # running a single design of experiments for the Road Test model.

    import emat.examples
    scope, db, model = emat.examples.road_test()
    design = model.design_experiments()
    model.run_experiments(design)

    # ## Single-Design Datasets

    # ### Writing Out Raw Data
    #
    # When the database has only a single design of experiments, or if we
    # don't care about any differentiation between multiple designs that we
    # may have created and ran, we can dump the entire set of model runs,
    # including uncertainties, policy levers, and performance measures, all
    # consolidated into a single pandas DataFrame using the
    # `read_experiment_all` function.  The constants even appear in this DataFrame
    # too, for good measure.

    df = db.read_experiment_all(scope.name)
    dataframe_regression.check(pd.DataFrame(df), basename='test_database__df')

    # Exporting this data is simply a matter of using the usual pandas
    # methods to save the dataframe to a format of your choosing.  We'll
    # save our data into a gzipped CSV file, which is somewhat compressed
    # (we're not monsters here) but still widely compatible for a variety of uses.

    df.to_csv("road_test_1.csv.gz")

    # This table contains most of the information we want to export from
    # our database, but not everything.  We also probably want to have access
    # to all of the information in the exploratory scope as well.  Our example
    # generator gives us a `Scope` reference directly, but if we didn't have that
    # we can still extract it from the database, using the `read_scope` method.

    s = db.read_scope()
    s.dump(filename="road_test_scope.yaml")

    # ### Reading In Raw Data
    #
    # Now, we're ready to begin anew, constructing a fresh database from scratch,
    # using only the raw formatted files.
    #
    # First, let's load our scope from the yaml file, and initialize a clean database
    # using that scope.

    s2 = emat.Scope("road_test_scope.yaml")
    db2 = emat.SQLiteDB("road_test_2.sqldb")
    db2.store_scope(s2)

    # Just as we used pandas to save out our consolidated DataFrame of experimental results,
    # we can use it to read in a consolidated table of experiments.

    df2 = pd.read_csv("road_test_1.csv.gz", index_col='experiment')
    # dataframe_regression.check(df2, basename='test_database__df2')

    # Writing experiments to a database is not quite as simple as reading them.  There
    # is a parallel `write_experiment_all` method for the `Database` class, but to use
    # it we need to provide not only the DataFrame of actual results, but also a name for
    # the design of experiments we are writing (all experiments exist within designs) and
    # the source of the performance measure results (zero means actual results from a
    # core model run, and non-zero values are ID numbers for metamodels). This allows many
    # different possible sets of performance measures to be stored for the same set
    # of input parameters.

    db2.write_experiment_all(
        scope_name=s2.name,
        design_name='general',
        source=0,
        xlm_df=df2,
    )
    df2b = db.read_experiment_all(scope.name)
    dataframe_regression.check(pd.DataFrame(df2b), basename='test_database__df2b')

    # ## Multiple-Design Datasets
    #
    # The EMAT database is not limited to storing a single design of experiments.  Multiple designs
    # can be stored for the same scope.  We'll add a set of univariate sensitivity test to our
    # database, and a "ref" design that contains a single experiment with all inputs set to their
    # default values.

    design_uni = model.design_experiments(sampler='uni')
    model.run_experiments(design_uni)
    model.run_reference_experiment()

    # We now have three designs stored in our database. We can confirm this
    # by reading out the set of design names.

    assert sorted(db.read_design_names(s.name)) == sorted(['lhs', 'ref', 'uni'])

    # Note that there
    # can be some experiments that are in more than one design.  This is
    # not merely duplicating the experiment and results, but actually
    # assigning the same experiment to both designs.  We can see this
    # for the 'uni' and 'ref' designs -- both contain the all-default
    # parameters experiment, and when we read these designs out of the
    # database, the same experiment number is reported out in both
    # designs.

    uni = db.read_experiment_all(scope.name, design_name='uni')
    ref = db.read_experiment_all(scope.name, design_name='ref')
    dataframe_regression.check(pd.DataFrame(uni), basename='test_database__uni')
    dataframe_regression.check(pd.DataFrame(ref), basename='test_database__ref')

    # ### Writing Out Raw Data
    #
    # We can read a single dataframe containing all the experiments associated with
    # this scope by omitting the `design_name` argument, just as if there was only
    # one design.

    df = db.read_experiment_all(scope.name)
    df.to_csv("road_test_2.csv.gz")

    # If we want to be able to reconstruct the various designs of experiments later,
    # we'll also need to write out instructions for that.  The `read_all_experiment_ids`
    # method can give us a dictionary of all the relevant information.

    design_experiments = db.read_all_experiment_ids(scope.name, design_name='*', grouped=True)
    data_regression.check(design_experiments)


    # We can write this dictionary to a file in 'yaml' format.

    with open("road_test_design_experiments.yaml", 'wt') as f:
        yaml.dump(design_experiments, f)

    ### Reading In Raw Data

    # To construct a new emat Database with multiple designs of experients,...

    db3 = emat.SQLiteDB("road_test_3.sqldb")
    db3.store_scope(s2)
    df3 = pd.read_csv("road_test_2.csv.gz", index_col='experiment')

    with open("road_test_design_experiments.yaml", 'rt') as f:
        design_experiments2 = yaml.safe_load(f)
    data_regression.check(design_experiments2)

    db3.write_experiment_all(
        scope_name=s2.name,
        design_name=design_experiments2,
        source=0,
        xlm_df=df3,
    )

    assert sorted(db3.read_design_names(s.name)) == sorted(['lhs', 'ref', 'uni'])

    dx = db3.read_all_experiment_ids(scope.name, design_name='*', grouped=True)
    assert dx == {'lhs': '1-110', 'ref': '111', 'uni': '111-132'}

    uni3 = db3.read_experiment_all(scope.name, design_name='uni')
    dataframe_regression.check(pd.DataFrame(uni3), basename='test_database__uni')

    ## Re-running Experiments

    # This section provides a short walkthrough of how to handle mistakes
    # in an EMAT database.  By "mistakes" we are referring to incorrect
    # values that have been written into the database by accident, generally
    # arising from core model runs that were misconfigured or suffered
    # non-fatal errors that caused the results to be invalid.
    #
    # One approach to handling such problems is to simply start over with a
    # brand new clean database file.  However, this may be inconvenient if
    # the database already includes a number of valid results, especially if
    # those valid results were expensive to generate.  It may also be desirable
    # to keep prior invalid results on hand, so as to easily recognized when
    # errors recur.
    #
    # We begin this example by populating our database with some more experimental data, by creating and
    # running a single design of experiments for the Road Test model, except these experiments will be
    # created with a misconfigured model (lane_width = 11, it should be 10), so the results will be bad.

    model.lane_width = 10.3
    oops = model.design_experiments(design_name='oops', random_seed=12345)
    model.run_experiments(oops)

    # We can review a dataframe of results as before, using the `read_experiment_all`
    # method. This time we will add `with_run_ids=True`, which will add an extra
    # column to the index, showing a universally unique id attached to each row
    # of results.

    oops_result1 = db.read_experiment_all(scope.name, 'oops', with_run_ids=True)
    dataframe_regression.check(pd.DataFrame(oops_result1).reset_index(drop=True), basename='test_database__oops_result1')

    # Some of these results are obviously problematic.  Increasing capacity cannot possibly
    # result in a negative travel time savings. (Braess paradox doesn't apply here because
    # it's just one link, not a network.)  So those negative values are clearly wrong.  We
    # can fix the model so they won't be wrong, but by default the `run_experiments` method
    # won't actually re-run models when the results are already available in the database.
    # To solve this conundrum, we can mark the incorrect results as invalid, using a query
    # to pull out the rows that can be flagged as wrong.

    db.invalidate_experiment_runs(
        queries=['time_savings < 0']
    )

    # The `[73]` returned here indicates that 73 sets of results were invalidated by this command.
    # Now we can fix our model, and then use the `run_experiments` method to get new model runs for
    # the invalidated results.

    model.lane_width = 10
    oops_result2 = model.run_experiments(oops)
    dataframe_regression.check(pd.DataFrame(oops_result2).reset_index(drop=True), basename='test_database__oops_result2')

    # The re-run fixed the negative values, although it left in place the other
    # experimental runs in the database. By the way we constructed this example,
    # we know those are wrong too, and it's evident in the apparent discontinuity
    # in the input flow graph, which we can zoom in on.

    # ax = oops_result2.plot.scatter(x='input_flow', y='time_savings', color='r')
    # ax.plot([109, 135], [0, 35], '--', color='y');

    # Those original results are bad too, and we want to invalidate them as well.
    # In addition to giving conditional queries to the `invalidate_experiment_runs`
    # method, we can also give a dataframe of results that have run ids attached,
    # and those unique ids will be used to to find and invalidate results in the
    # database.  Here, we pass in the dataframe of all the results, which contains
    # all 110 runs, but only 37 runs are newly invalidated (77 were invalidated
    # previously).

    db.invalidate_experiment_runs(oops_result1)

    # Now when we run the experiments again, those 37 experiments are re-run.

    oops_result3 = model.run_experiments(oops)
    dataframe_regression.check(pd.DataFrame(oops_result3).reset_index(drop=True), basename='test_database__oops_result3')

    ### Writing Out All Runs
    #
    # By default, the `read_experiment_all` method returns the most recent valid set of
    # performance measures for each experiment, but we can override this behavior to
    # ask for all run results, or all valid or invalid results.  This allows us to easily
    # write out data files containing all the results stored in the database.

    oops_all = db.read_experiment_all(scope.name, with_run_ids=True, runs='all')
    dataframe_regression.check(pd.DataFrame(oops_all).reset_index(drop=True), basename='test_database__oops_all')

    # If we want to mark the valid and invalid runs, we can read them
    # seperately and attach a tag to the two dataframes.

    runs_1 = db.read_experiment_all(scope.name, with_run_ids=True, runs='valid')
    runs_1['is_valid'] = True
    runs_0 = db.read_experiment_all(scope.name, with_run_ids=True, runs='invalid')
    runs_0['is_valid'] = False
    all_runs = pd.concat([runs_1, runs_0])
    dataframe_regression.check(pd.DataFrame(all_runs).reset_index(drop=True), basename='test_database__all_runs')


    # These mechanisms can be use to write out results of multiple runs,
    # and to repopulate a database with both valid
    # and invalid raw results. This can be done multiple ways (seperate
    # files, one combined file, keeping track of invalidation queries, etc.).
    # The particular implementations of each are left as an exercise for
    # the reader.
