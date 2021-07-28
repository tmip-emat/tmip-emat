from abc import abstractmethod
from addicty import Dict
from typing import Mapping
from botocore.exceptions import ClientError

from ...scope.scope import Scope
from ...util.loggers import get_module_logger

_logger = get_module_logger(__name__)

_NO_DEFAULT = 'yeah no'


class _GeneralStore:

    _valid_types = ()

    def __init__(self, parent, keydir):
        self._data = {}
        self.parent = parent
        self.keydir = keydir

    @property
    def bucket(self):
        if self.parent is not None:
            return self.parent.bucket
        raise ValueError("no bucket assigned")

    @property
    def readonly(self):
        if self.parent is not None:
            return self.parent.readonly
        raise ValueError("no parent storage attached")

    def _s3_uri(self, *keys):
        key = "/".join(str(i) for i in keys)
        if self.keydir:
            return f"s3://{self.bucket}/{self.keydir}/{key}"
        else:
            return f"s3://{self.bucket}/{key}"

    def __getitem__(self, key, default=_NO_DEFAULT):
        if key not in self._data:
            # This will raise a KeyError if it's not there either
            try:
                return self._download_item(key)
            except KeyError:
                if default != _NO_DEFAULT:
                    return default
                else:
                    raise
        elif default != _NO_DEFAULT:
            return self._data.get(key, default)
        else:
            return self._data[key]

    def __setitem__(self, key, value):
        if self.readonly:
            raise ValueError("this store is read-only")
        if not isinstance(value, self._valid_types):
            raise TypeError(f"{self.__class__.__name__} can only "
                            f"hold {' or '.join(i.__name__ for i in self._valid_types)} "
                            f"not {type(value).__name__}")
        value = self._upload_item(key, value)
        print(f"cached [{key}] = {value}")
        self._data[key] = value

    def _upload_item(self, key, value):
        print(value)
        print(f"fake writing to {self._s3_uri(key)}")
        return value

    @abstractmethod
    def _download_item(self, key):
        """
        Load and return an item from S3 into the local cache.

        Raises
        ------
        KeyError
            If the key is not available from S3.
        """

    def __repr__(self):
        return f"{self.__class__.__name__} @ s3://{self.bucket}/{self.keydir}\n{self._data!r}"

class DictStore(_GeneralStore):

    _valid_types = (Mapping, )

    def _upload_item(self, key, value):
        value = Dict(value)
        print(value)
        print(f"fake writing to {self._s3_uri(key)}")
        return value

    def _download_item(self, key):
        """
        Load an item from S3
        """
        value = Dict(TODO='make this work')
        self._data[key] = value
        return value


class ScopeStore(_GeneralStore):

    _valid_types = (Scope, )

    def _upload_item(self, key, value):
        uri = self._s3_uri(key)
        print(f"writing to {uri}")
        Dict.load(value.dump()).dump(uri)
        return value

    def _download_item(self, key):
        """
        Load an item from S3
        """
        uri = self._s3_uri(key)
        print(f"loading from {uri}")
        try:
            value = Scope(None, Dict.load(uri).dump())
        except ClientError:
            raise KeyError(key)
        except:
            _logger.exception("failed to load from S3")
            raise
        # from ...examples import road_test
        # value, _, _ = road_test()
        self._data[key] = value
        _logger.debug(f"downloaded [{key}] {value}")
        return value

