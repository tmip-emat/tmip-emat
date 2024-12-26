from __future__ import annotations

import os
import sys

import emat


def get_version():
    version = emat.__version__
    os.environ["EMAT_VERSION"] = version
    if len(sys.argv) >= 2:
        v_file = sys.argv[1]
    else:
        v_file = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "EMAT_VERSION.txt")
        )
    with open(v_file, "w") as f:
        f.write(f"EMAT_VERSION={version}\n")
    print(f"EMAT_VERSION={version}")
    return version


if __name__ == "__main__":
    get_version()
