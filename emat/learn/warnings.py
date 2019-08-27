import warnings, contextlib

@contextlib.contextmanager
def ignore_warnings(category=Warning):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=category)
		yield
