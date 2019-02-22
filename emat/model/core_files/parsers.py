
import os
import abc
import numpy as np
import pandas as pd
from typing import Mapping

from ...util.loggers import get_module_logger
_logger = get_module_logger(__name__)

class FileParser(abc.ABC):
	"""
	A tool to parse performance measure(s) from an arbitrary file format.

	Args:
		filename (str):
			The name of the file in which the measure(s) are stored.
			The filename is a relative path to the file, and will be
			evaluated relative to the `from_dir` argument in the `read`
			method.
	"""

	def __init__(
			self,
			filename,
	):
		self.filename = filename

	@abc.abstractmethod
	def read(self, from_dir):
		"""
		Read the performance measures.

		Args:
			from_dir (Path-like): The base directory from which to read the data.

		Returns:
			Dict: The measures read from this file.
		"""
		pass

	@property
	@abc.abstractmethod
	def measure_names(self):
		"""
		List: the measure names contained in this TableParser.
		"""
		pass


class TableParser(FileParser):
	"""
	A tool to parse performance measure from an arbitrary table format.

	Args:
		filename (str):
			The name of the file in which the tabular data is stored.
			The filename is a relative path to the file, and will be
			evaluated relative to the `from_dir` argument in the `read`
			method.
		measure_getters (Mapping[str, Getter]): A mapping that
			relates scalar performance measure values to Getters that
			extract values from the tabular data.
		reader_method (Callable, default pandas.read_csv): A function that
			accepts one positional argument (the filename to be read) and
			optionally some keyword arguments, and returns a pandas.DataFrame.
		handle_errors (str, default 'raise'): How to handle errors when reading a table, one
			of {'raise', 'nan'}
		**kwargs (Mapping, optional): A set of fixed keyword arguments
			that will be passed to `reader_method` each time it is called.

	"""

	def __init__(
			self,
			filename,
			measure_getters,
			reader_method = pd.read_csv,
			handle_errors = 'raise',
			**kwargs,
	):
		super().__init__(filename)
		if not isinstance(measure_getters, Mapping):
			raise TypeError('measure_getters must be a mapping')
		self.measure_getters = measure_getters
		self.reader_method = reader_method
		self.reader_kwargs = kwargs
		if handle_errors not in {'raise','nan'}:
			raise ValueError("handle_errors not in {'raise', 'nan'}")
		self.handle_errors = handle_errors

	def raw(self, from_dir):
		"""
		Read the raw tabular data.

		Args:
			from_dir (Path-like): The base directory from which to read the data.

		Returns:
			pandas.DataFrame
		"""
		f = os.path.join(from_dir, self.filename)
		if not os.path.exists(f):
			raise FileNotFoundError(f)
		return self.reader_method( f, **self.reader_kwargs, )

	def read(self, from_dir):
		"""
		Read the performance measures.

		Args:
			from_dir (Path-like): The base directory from which to read the data.

		Returns:
			Dict: The measures read from this file.
		"""
		data = self.raw(from_dir)
		result = {}

		for measure_name, getter in self.measure_getters.items():
			try:
				result[measure_name] = getter(data)
			except:
				if self.handle_errors == 'nan':
					_logger.exception(f"Error in reading {os.path.join(from_dir, self.filename)}")
					result[measure_name] = np.nan
				else:
					_logger.error(f"Error in reading {os.path.join(from_dir, self.filename)}")
					_logger.error(f"  table shape {data.shape}")
					_logger.error(f"  index {data.index}")
					_logger.error(f"  columns  {data.columns}")
					raise

		return result

	@property
	def measure_names(self):
		"""
		List: the measure names contained in this TableParser.
		"""
		return sorted(self.measure_getters.keys())





###


def slice_repr(x):
	if isinstance(x, slice):
		if x.start is None and x.stop is not None:
			r = f":{x.stop}"
		elif x.start is not None and x.stop is None:
			r = f"{x.start}:"
		elif x.start is not None and x.stop is not None:
			r = f"{x.start}:{x.stop}"
		else:
			r = ":"
		if x.step is not None:
			r += f":{x.step}"
		return r
	else:
		return repr(x)

def tuple_repr_with_slice(xx):
	return ",".join(slice_repr(x) for x in xx)


class Getter:
	"""
	A tool to get defined value[s] from a pandas.DataFrame.

	Use a getter by calling it with the DataFrame as the sole argument.
	"""
	def __call__(self, x):
		raise NotImplementedError

class SingleGetter(Getter):
	def __init__(self, *item):
		self._item = item
	def __repr__(self):
		return f"{self.__class__.__name__[1:].lower()}[{tuple_repr_with_slice(self._item)}]"
	def __add__(self, other):
		return SumOfGetter(self, other)

class SumOfGetter(Getter):
	def __init__(self, *parts):
		self._parts = list(parts)
	def __call__(self, x):
		return sum(p(x) for p in self._parts)
	def __repr__(self):
		return "+".join(repr(x) for x in self._parts)
	def __add__(self, other):
		return SumOfGetter(*self._parts, other)


class _Loc(SingleGetter):
	def __call__(self, x):
		return x.loc[self._item]

class _Loc_Sum(SingleGetter):
	def __call__(self, x):
		return np.sum(x.loc[self._item])

class _Loc_Mean(SingleGetter):
	def __call__(self, x):
		return np.nanmean(x.loc[self._item])


class __LocMaker:
	def __getitem__(self, item):
		return _Loc(*item)

class __LocSumMaker:
	def __getitem__(self, item):
		return _Loc_Sum(*item)

class __LocMeanMaker:
	def __getitem__(self, item):
		return _Loc_Mean(*item)


loc = __LocMaker()
loc_sum = __LocSumMaker()
loc_mean = __LocMeanMaker()


class _Iloc(SingleGetter):
	def __call__(self, x):
		return x.iloc[self._item]

class _Iloc_Sum(SingleGetter):
	def __call__(self, x):
		return np.sum(x.iloc[self._item])

class _Iloc_Mean(SingleGetter):
	def __call__(self, x):
		return np.nanmean(x.iloc[self._item])


class __IlocMaker:
	def __getitem__(self, item):
		return _Iloc(*item)

class __IlocSumMaker:
	def __getitem__(self, item):
		return _Iloc_Sum(*item)

class __IlocMeanMaker:
	def __getitem__(self, item):
		return _Iloc_Mean(*item)


iloc = __IlocMaker()
iloc_sum = __IlocSumMaker()
iloc_mean = __IlocMeanMaker()


