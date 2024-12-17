import os
import subprocess
import tempfile
import uuid

import numpy as np
import pandas as pd
import pytest

import emat.examples
from emat.database.dynamo.storage import DynamoDB


@pytest.fixture(scope="module")
def dynamo_local():
    if os.environ.get("GITHUB_ACTIONS") == "true":
        yield None
    else:
        print("initializing local DynamoDB")
        dynamo_jar_dir = os.path.expanduser("~/Applications/dynamodb_local_latest")
        if not os.path.isdir(dynamo_jar_dir):
            raise NotADirectoryError(dynamo_jar_dir)
        tempdir = tempfile.TemporaryDirectory()
        java = subprocess.Popen(
            [
                "java",
                "-Djava.library.path=./DynamoDBLocal_lib",
                "-jar",
                "DynamoDBLocal.jar",
                "-port",
                "8123",
                "-dbPath",
                tempdir.name,
            ],
            cwd=dynamo_jar_dir,
        )
        print("local DynamoDB ready")
        yield None
        # teardown
        print("terminate local DynamoDB")
        java.terminate()


def test_simple_experiments(dynamo_local):
    ddb = DynamoDB(local_port=8123)
    s, db, m = emat.examples.road_test(db=ddb)
    ddb.store_scope(s)
    assert s.domain == "EMAT Road Test"
    assert db.read_design_names(s.domain) == []

    design = s.design_experiments(db=ddb, n_samples=5, design_name="lhs5")
    readback = ddb.read_experiment_parameters(s.name, "lhs5")
    pd.testing.assert_frame_equal(design, readback[design.columns], check_names=False)

    out1 = m.run_experiments(design)
    out2 = (
        m.read_experiment_measures("lhs5")
        .reset_index()
        .set_index("experiment_id")
        .drop(columns="run_id")
    )

    print("#" * 33)
    print(out1)
    print("$" * 33)
    print(out2)
    print("^" * 33)

    pd.testing.assert_frame_equal(out2, out1[out2.columns], check_names=False)

    from addicty import Dict

    ss = Dict.load(s.dump())
    ss.scope.name = "EMAT Road Test v2"
    ss.inputs.expand_capacity.default = 50.0
    ss.inputs.expand_capacity.min = 50.0
    s2 = emat.Scope(None, scope_def=ss.dump())
    db.store_scope(s2)

    assert s2.domain == "EMAT Road Test"
    assert s2.name == "EMAT Road Test v2"
    design6 = s2.design_experiments(db=db, n_samples=6, design_name="lhs6")
    pd.testing.assert_index_equal(
        design6.index, pd.Index([6, 7, 8, 9, 10, 11], name="experiment")
    )

    assert len(db.read_experiment_parameters(s2.domain)) == 11

    del ss.inputs.alpha.max
    del ss.inputs.alpha.min
    del ss.inputs.alpha.dist
    ss.inputs.alpha.ptype = "constant"
    ss.scope.name = "EMAT Road Test v3"
    s3 = emat.Scope(None, scope_def=ss.dump())
    db.store_scope(s3)
    assert db.read_scope_names() == [
        "EMAT Road Test",
        "EMAT Road Test v2",
        "EMAT Road Test v3",
    ]

    design4 = s3.design_experiments(db=db, n_samples=4, design_name="lhs4")

    assert len(db.read_experiment_parameters(s2.domain)) == 15
    assert db.read_scope_names(domain="EMAT Road Test") == [
        "EMAT Road Test",
        "EMAT Road Test v2",
        "EMAT Road Test v3",
    ]


def test_giant_write():
    ddb = DynamoDB(local_port=8123)
    runid = uuid.uuid1()
    big_result = {f"LongItemName{n}": np.sqrt(n) for n in range(1, 500)}

    ddb._put_experiment_result(
        scope_name="NotNamed",
        experiment_id=1,
        run_id=runid,
        results=big_result,
        overwrite=False,
    )

    x = ddb._get_experiment_result(scope_name="NotNamed", experiment_id=1, run_id=runid)

    assert x == pytest.approx(big_result)
