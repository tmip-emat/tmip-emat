
__version__ = "1.0.3"

def require_version(required_version):
	if required_version.split('.') > __version__.split('.'):
		raise ValueError("this multitarget is version {}".format(__version__))

import warnings, contextlib


@contextlib.contextmanager
def ignore_warnings(category=Warning):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=category)
		yield



from .simple import MultipleTargetRegression, DetrendedMultipleTargetRegression

from . import multiout_patch
