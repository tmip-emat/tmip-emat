
__version__ = "1.0.4"

def require_version(required_version):
	if required_version.split('.') > __version__.split('.'):
		raise ValueError("this filez is version {}".format(__version__))

from .opening import *
from .finding import *
from .spooling import *
from .timing import *
from .saving import *

