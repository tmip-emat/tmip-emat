

import appdirs
import os
import yaml
from collections.abc import MutableMapping

config_dir = appdirs.user_config_dir(appname='emat', appauthor='tmip')
config_file = os.path.join(config_dir, 'config.yaml')

class Config(MutableMapping):
    """
    A configuration dictionary-like object.

    Args:
        filename (str): Initial values for this dictionary are
            loaded from this file using `yaml.safe_load`. Changes
            to this dictionary are immediately written to disk in
            the same file.
        makedirs (bool, default True): If true, any intermediate
            directories are created as needed.

    """

    def __init__(self, filename, makedirs=True):
        self._filename = filename
        if makedirs:
            os.makedirs(os.path.dirname(self._filename), exist_ok=True)
        if os.path.exists(self._filename):
            with open(self._filename) as s:
                self._data = yaml.safe_load(s)
            if self._data is None:
                self._data = {}
        else:
            self._data = {}

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value
        with open(self._filename, 'wt') as s:
            yaml.safe_dump(self._data, s)

    def __delitem__(self, key):
        del self._data[key]
        with open(self._filename, 'wt') as s:
            yaml.safe_dump(self._data, s)

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self._data.__len__()

    def __repr__(self):
        return f"<emat.Config {self._filename}>\n"+self._data.__repr__()

    def get_subdir(self, key, *subdirs, makedirs=True, normpath=True):
        """Get a sub-directory of a configuration directory, it it exists.

        Args:
            key (str): The key for a directory setting
            *subdirs: Subdirectory components to add to the path
            makedirs (bool): Whether to make the sub-directories if
                they do not exist in the file system, default True.
            normpath (bool): Whether to normalize the resulting path,
                default True.

        """
        directory = self.get(key, None)
        if directory is None:
            return None
        directory = os.path.join(directory, *subdirs)
        if normpath:
            directory = os.path.normpath(directory)
        if makedirs:
            os.makedirs(directory, exist_ok=True)
        return directory

config = Config(config_file)
