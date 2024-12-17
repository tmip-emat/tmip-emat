from yaml import SafeDumper
import numpy as np

SafeDumper.add_representer(
    np.float64,
    lambda dumper, data: dumper.represent_float(float(data))
)
SafeDumper.add_representer(
    np.float32,
    lambda dumper, data: dumper.represent_float(float(data))
)
SafeDumper.add_representer(
    np.int32,
    lambda dumper, data: dumper.represent_int(int(data))
)
SafeDumper.add_representer(
    np.int64,
    lambda dumper, data: dumper.represent_int(int(data))
)
SafeDumper.add_representer(
    np.bool_,
    lambda dumper, data: dumper.represent_bool(bool(data))
)
SafeDumper.add_representer(
    np.ndarray,
    lambda dumper, data: dumper.represent_list(list(data))
)
