import functools
import hashlib
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile

import appdirs
import requests
from tqdm.auto import tqdm


def download(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path


def download_jar(
    url="https://s3.us-west-2.amazonaws.com/dynamodb-local/dynamodb_local_latest.tar.gz",
    directory=None,
    checksum_url="https://s3.us-west-2.amazonaws.com/dynamodb-local/dynamodb_local_latest.tar.gz.sha256",
    known_sha256="10d31bb846c4879fcb0f147304bca8274b2a01c140867533e52af390134f5986",
):

    if directory is None:
        directory = appdirs.user_data_dir(appname="emat", appauthor="camsys")
    tar_gz = os.path.join(directory, "dynamodb_local_latest.tar.gz")
    try:
        r = requests.get(checksum_url, allow_redirects=True, timeout=10)
    except:
        expected_sha256 = known_sha256
    else:
        expected_sha256 = r.content[:64].decode()
    if os.path.exists(tar_gz):
        block_size = 65536
        sha256 = hashlib.sha256()
        with open(tar_gz, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                sha256.update(block)
        if expected_sha256 != sha256.hexdigest():
            print(f"existing file fails checksum: {tar_gz}")
            os.remove(tar_gz)
        else:
            print(f"using existing file: {tar_gz}")
    if not os.path.exists(tar_gz):
        print(f"downloading jar file to: {tar_gz}")
        download(url, tar_gz)
    with tarfile.open(tar_gz) as tar:
        tar.extractall(directory)


class DynamoServer:
    def __init__(self, directory=None, data_directory=".", port=8123):
        if directory is None:
            directory = appdirs.user_data_dir(appname="emat", appauthor="camsys")
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)
        download_jar(directory=directory)
        print("initializing local DynamoDB")
        if data_directory is None:
            self.tempdir = tempfile.TemporaryDirectory()
            data_directory = self.tempdir.name
        self.port = str(port)
        self.java = subprocess.Popen(
            [
                "java",
                "-Djava.library.path=./DynamoDBLocal_lib",
                "-jar",
                "DynamoDBLocal.jar",
                "-port",
                self.port,
                "-dbPath",
                data_directory,
            ],
            cwd=directory,
        )
        print("local DynamoDB ready")

    def __del__(self):
        print("terminate local DynamoDB")
        self.java.terminate()
