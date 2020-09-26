
import pandas as pd
import numpy as np
from typing import Iterable, cast
from pandas import Index
from pandas.core import algorithms as _algorithms

def reindex_duplicates(df, subset=None):
	"""
	Replace index for all duplicate rows with the first row's index.

	Args:
		df (pandas.DataFrame): The frame to reindex
		subset: Only these columns are used to determine duplicate entries.

	Returns:
		pandas.DataFrame
	"""
	from pandas._libs.hashtable import _SIZE_HINT_LIMIT

	from pandas.core.sorting import get_group_index

	if df.empty:
		return df._constructor_sliced(dtype=bool)

	def f(vals):
		labels, shape = _algorithms.factorize(
			vals, size_hint=min(len(df), _SIZE_HINT_LIMIT)
		)
		return labels.astype("i8", copy=False), len(shape)

	if subset is None:
		subset = df.columns
	elif (
			not np.iterable(subset)
			or isinstance(subset, str)
			or isinstance(subset, tuple)
			and subset in df.columns
	):
		subset = (subset,)

	#  needed for mypy since can't narrow types using np.iterable
	subset = cast(Iterable, subset)

	# Verify all columns in subset exist in the queried dataframe
	# Otherwise, raise a KeyError, same as if you try to __getitem__ with a
	# key that doesn't exist.
	diff = Index(subset).difference(df.columns)
	if not diff.empty:
		raise KeyError(diff)

	vals = (col.values for name, col in df.items() if name in subset)
	labels, shape = map(list, zip(*map(f, vals)))

	ids = get_group_index(labels, shape, sort=False, xnull=False)
	uids = np.unique(ids, return_index=True, return_inverse=True)

	first_ids = df.index[uids[1][uids[2]]]
	return df._constructor(data=df.to_numpy(), index=first_ids, columns=df.columns)


def count_diff_rows(df_a, df_b):
	df_a_ = df_a.reindex(df_b.index)
	df_b_ = df_b.reindex(df_a_.index)
	return (~((df_a_.eq(df_b_) | (df_a_.isna() & df_b_.isna())).all(axis=1))).sum()