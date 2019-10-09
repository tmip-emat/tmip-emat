
import os
from .hasher import hash_it
from . import filez

def load_cache_if_available(cache_dir=None, cache_file=None, **kwargs):
	"""
	Load an object from a disk cache file if possible.

	Parameters
	----------
	*args
	cache_dir
	cache_file

	Returns
	-------
	obj : object, or None
		The loaded object, or None if not available
	cache_file : path-like, or None
		The path to where a future cache of this object
		should go.
	"""

	if cache_file is not None:
		if os.path.exists(cache_file):
			return filez.load(cache_file), None
		else:
			os.makedirs(os.path.dirname(cache_file), exist_ok=True)
			with open(cache_file+'.info.txt', 'wt') as notes:
				for k,v in kwargs.items():
					print(k, "=", v, file=notes)
					print("\n",file=notes)
			return None, cache_file

	if cache_dir is None:
		return None, None

	# If cache_dir is used
	try:
		hh = hash_it(*kwargs.items())
		subdir = os.path.join(cache_dir, hh[2:4], hh[4:6])
		os.makedirs(subdir, exist_ok=True)
		cache_file = os.path.join(subdir, hh[6:] + ".gz")
		if os.path.exists(cache_file):
			return filez.load(cache_file), None
		else:
			with open(cache_file+'.info.txt', 'wt') as notes:
				for k,v in kwargs.items():
					print(k, "=", v, file=notes)
					print("\n",file=notes)
			return None, cache_file

	except:
		import warnings, traceback
		warnings.warn('unable to manage cache')
		traceback.print_exc()
		return None, None


def save_cache(obj, cache_file):
	if obj is None or cache_file is None:
		return
	filez.save(obj, cache_file, overwrite=True)

