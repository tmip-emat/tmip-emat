
import hashlib
from typing import Collection
from ..workbench.em_framework.samplers import DefaultDesigns
from ..workbench.em_framework.util import NamedDict
import pandas
from pandas.util import hash_pandas_object

def hash_it(*args, ha=None):
	if ha is None:
		ha = hashlib.sha1()
	for a in args:
		try:
			ha.update(a)
		except:
			if isinstance(a, str):
				ha.update(a.encode())
			elif isinstance(a, pandas.DataFrame):
				ha.update(hash_pandas_object(a, index=True).values)
				hash_it(a.columns, ha=ha)
			elif isinstance(a, Collection):
				hash_it(*a, ha=ha)
			elif hasattr(a, '_hash_it'):
				hash_it(a._hash_it(ha=ha))
			elif isinstance(a, (int, float)):
				ha.update(str(a).encode())
			elif a is None:
				ha.update(b"None")
			elif isinstance(a, DefaultDesigns):
				hash_it(
					a.designs,
					a.parameters,
					a.params,
					ha=ha,
				)
			elif isinstance(a, NamedDict):
				hash_it(
					a.name,
					tuple(a.items()),
					ha=ha,
				)
			else:
				print("cant hashit",(type(a)))
				print(a)
				raise
	return ha.hexdigest()
