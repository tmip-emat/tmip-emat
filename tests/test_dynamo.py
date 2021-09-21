import os
import tempfile
import uuid
import numpy as np
import pandas as pd
import pytest
import subprocess
import emat.examples
from emat.database.dynamo.storage import DynamoDB


@pytest.fixture(scope="module")
def dynamo_local():
    print("initializing local DynamoDB")
    dynamo_jar_dir = os.path.expanduser("~/Applications/dynamodb_local_latest")
    if not os.path.isdir(dynamo_jar_dir):
        raise NotADirectoryError(dynamo_jar_dir)
    tempdir = tempfile.TemporaryDirectory()
    java = subprocess.Popen(
        [
            "java", "-Djava.library.path=./DynamoDBLocal_lib",
            "-jar", "DynamoDBLocal.jar", "-port", "8123", "-dbPath", tempdir.name
        ],
        cwd=dynamo_jar_dir,
    )
    print("local DynamoDB ready")
    yield None
    # teardown
    print("terminate local DynamoDB")
    java.terminate()


def test_simple_experiments(dynamo_local):
    s, db, m = emat.examples.road_test()
    ddb = DynamoDB(bucket="trial-addicty-bucket", local_port=8123)
    ddb.store_scope(s)
    design = s.design_experiments(db=ddb, n_samples=5, design_name="lhs5")
    readback = ddb.read_experiment_parameters(s.name, "lhs5")
    pd.testing.assert_frame_equal(design, readback[design.columns], check_names=False)


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
