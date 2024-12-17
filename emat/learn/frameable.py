
import pandas

class FrameableMixin:
	"""
	Methods to store and apply DataFrame formatting for fit-predict estimators.
	"""

	def _pre_fit(self, X, Y):
		"""Store column labels from fitted data."""

		if isinstance(Y, pandas.DataFrame):
			self._Y_columns = list(Y.columns)
		elif isinstance(Y, pandas.Series):
			self._Y_columns = str(Y.name)
		else:
			self._Y_columns = None


	def _post_predict(self, X, Yhat, on_error='raise'):
		"""Attach index and/or column labels to predictions."""

		if isinstance(X, pandas.DataFrame):
			idx = X.index
		else:
			idx = None

		if Yhat.ndim == 3 and Yhat.shape[0] == 1:
			Yhat = Yhat.squeeze(0)

		cols = None
		if self._Y_columns is not None:
			if isinstance(self._Y_columns, str):
				if len(Yhat.shape) > 1 and Yhat.shape[1] == 1:
					cols = [self._Y_columns]
				elif len(Yhat.shape) == 1:
					cols = self._Y_columns
			else:
				if len(Yhat.shape) > 1 and Yhat.shape[1] == len(self._Y_columns):
					cols = self._Y_columns

		if idx is not None or cols is not None:
			if isinstance(cols, str):
				try:
					Yhat = pandas.Series(
						Yhat,
						index=idx,
						name=self._Y_columns,
					)
				except ValueError:
					if on_error == 'raise':
						raise
			else:
				try:
					Yhat = pandas.DataFrame(
						Yhat,
						index=idx,
						columns=self._Y_columns,
					)
				except ValueError:
					if on_error == 'raise':
						raise
		return Yhat
