
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.validation import _num_samples, column_or_1d
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.externals.six import with_metaclass
from sklearn.utils.fixes import signature, comb
from sklearn.base import _pprint


class ExogenouslyStratifiedKFold(StratifiedKFold):
	"""Exogenously Stratified K-Folds cross-validator

	Provides train/test indices to split data in train/test sets.

	This cross-validation object is a variation of KFold that returns
	stratified folds. The folds are made by preserving the percentage of
	an exogenously defined factor for each class.

	Parameters
	----------
	n_splits : int, default=3
		Number of folds. Must be at least 2.

	shuffle : boolean, optional
		Whether to shuffle each stratification of the data before splitting
		into batches.

	random_state : int, RandomState instance or None, optional, default=None
		If int, random_state is the seed used by the random number generator;
		If RandomState instance, random_state is the random number generator;
		If None, the random number generator is the RandomState instance used
		by `np.random`. Used when ``shuffle`` == True.

	Examples
	--------
	>>> from sklearn.model_selection import ExogenouslyStratifiedKFold
	>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
	>>> y = np.array([0, 0, 1, 1])
	>>> skf = StratifiedKFold(n_splits=2)
	>>> skf.get_n_splits(X, y)
	2
	>>> print(skf)  # doctest: +NORMALIZE_WHITESPACE
	StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
	>>> for train_index, test_index in skf.split(X, y):
	...    print("TRAIN:", train_index, "TEST:", test_index)
	...    X_train, X_test = X[train_index], X[test_index]
	...    y_train, y_test = y[train_index], y[test_index]
	TRAIN: [1 3] TEST: [0 2]
	TRAIN: [0 2] TEST: [1 3]

	Notes
	-----
	All the folds have size ``trunc(n_samples / n_splits)``, the last one has
	the complementary.

	See also
	--------
	RepeatedExogenouslyStratifiedKFold: Repeats Exogenously Stratified K-Fold n times.
	"""

	def __init__(self, exo_data=None, n_splits=3, shuffle=False, random_state=None):
		super().__init__(n_splits, shuffle, random_state)
		self.exo_data = exo_data

	def _make_test_folds(self, X, y=None):
		rng = self.random_state

		if self.exo_data is not None:
			if y.shape[0] != self.exo_data.shape[0]:
				raise ValueError(f"bad shape of exo_data, y.shape={y.shape}, self.exo_data.shape={self.exo_data.shape}")
			y = self.exo_data

		y = np.asarray(y)
		type_of_target_y = type_of_target(y)
		allowed_target_types = ('binary', 'multiclass')
		if type_of_target_y not in allowed_target_types:
			raise ValueError(
				'Supported target types are: {}. Got {!r} instead.'.format(
					allowed_target_types, type_of_target_y))

		y = column_or_1d(y)
		n_samples = y.shape[0]
		unique_y, y_inversed = np.unique(y, return_inverse=True)
		y_counts = np.bincount(y_inversed)
		min_groups = np.min(y_counts)
		if np.all(self.n_splits > y_counts):
			raise ValueError("n_splits=%d cannot be greater than the"
							 " number of members in each class."
							 % (self.n_splits))
		if self.n_splits > min_groups:
			warnings.warn(("The least populated class in y has only %d"
						   " members, which is too few. The minimum"
						   " number of members in any class cannot"
						   " be less than n_splits=%d."
						   % (min_groups, self.n_splits)), Warning)

		# pre-assign each sample to a test fold index using individual KFold
		# splitting strategies for each class so as to respect the balance of
		# classes
		# NOTE: Passing the data corresponding to ith class say X[y==class_i]
		# will break when the data is not 100% stratifiable for all classes.
		# So we pass np.zeroes(max(c, n_splits)) as data to the KFold
		per_cls_cvs = [
			KFold(self.n_splits, shuffle=self.shuffle,
				  random_state=rng).split(np.zeros(max(count, self.n_splits)))
			for count in y_counts]

		test_folds = np.zeros(n_samples, dtype=np.int)
		for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
			for cls, (_, test_split) in zip(unique_y, per_cls_splits):
				cls_test_folds = test_folds[y == cls]
				# the test split can be too big because we used
				# KFold(...).split(X[:max(c, n_splits)]) when data is not 100%
				# stratifiable for all the classes
				# (we use a warning instead of raising an exception)
				# If this is the case, let's trim it:
				test_split = test_split[test_split < len(cls_test_folds)]
				cls_test_folds[test_split] = test_fold_indices
				test_folds[y == cls] = cls_test_folds

		return test_folds

	def _iter_test_masks(self, X, y=None, groups=None):
		test_folds = self._make_test_folds(X, y)
		for i in range(self.n_splits):
			yield test_folds == i

	def split(self, X, y, groups=None):
		"""Generate indices to split data into training and test set.

		Parameters
		----------
		X : array-like, shape (n_samples, n_features)
			Training data, where n_samples is the number of samples
			and n_features is the number of features.

			Note that providing ``y`` is sufficient to generate the splits and
			hence ``np.zeros(n_samples)`` may be used as a placeholder for
			``X`` instead of actual training data.

		y : array-like, shape (n_samples,)
			The target variable for supervised learning problems.
			Stratification is done based on the y labels.

		groups : object
			Always ignored, exists for compatibility.

		Returns
		-------
		train : ndarray
			The training set indices for that split.

		test : ndarray
			The testing set indices for that split.

		Notes
		-----
		Randomized CV splitters may return different results for each call of
		split. You can make the results identical by setting ``random_state``
		to an integer.
		"""
		y = check_array(y, ensure_2d=False, dtype=None)
		return super().split(X, y, groups)


class RepeatedExogenouslyStratifiedKFold(_RepeatedSplits):
	"""Repeated Stratified K-Fold cross validator.

	Repeats Stratified K-Fold n times with different randomization in each
	repetition.

	Read more in the :ref:`User Guide <cross_validation>`.

	Parameters
	----------
	n_splits : int, default=5
		Number of folds. Must be at least 2.

	n_repeats : int, default=10
		Number of times cross-validator needs to be repeated.

	random_state : None, int or RandomState, default=None
		Random state to be used to generate random state for each
		repetition.

	Examples
	--------
	>>> from sklearn.model_selection import RepeatedStratifiedKFold
	>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
	>>> y = np.array([0, 0, 1, 1])
	>>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
	...     random_state=36851234)
	>>> for train_index, test_index in rskf.split(X, y):
	...     print("TRAIN:", train_index, "TEST:", test_index)
	...     X_train, X_test = X[train_index], X[test_index]
	...     y_train, y_test = y[train_index], y[test_index]
	...
	TRAIN: [1 2] TEST: [0 3]
	TRAIN: [0 3] TEST: [1 2]
	TRAIN: [1 3] TEST: [0 2]
	TRAIN: [0 2] TEST: [1 3]


	See also
	--------
	RepeatedKFold: Repeats K-Fold n times.
	"""
	def __init__(self, exo_data=None, n_splits=5, n_repeats=10, random_state=None):
		super().__init__(
			ExogenouslyStratifiedKFold,
			n_repeats,
			random_state,
			n_splits=n_splits,
			exo_data=exo_data,
		)
