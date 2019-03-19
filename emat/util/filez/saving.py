
from .opening import open_file_writer

import os
import gzip
try:
	import cloudpickle as pickle
except ImportError:
	import pickle




def save(
		obj,
		filename,
		overwrite=False,
		archive_dir='./archive/',
		compress=True,
):
	"""
	Save an object to a file.

	Parameters
	----------

	obj : Any
		An object to be saved.  Must be pickle-able (using cloudpickle if
		available, otherwise regular pickle).

	filename : Path-like
		This is either a text or byte string giving the name (and the path
		if the file isn't in the current working directory) of the file to
		be written to.

	overwrite : {True, False, 'spool', 'archive'}, default False
		Indicates what to do with an existing file at the same location.
		True will simply overwrite the existing file.
		False will raise a `FileExistsError`.
		'archive' will rename and/or move the existing file so that it
		will not be overwritten.
		'spool' will add a number to the filename of the file to be
		created, so that it will not overwrite the existing file.

	archive_dir : Path-like
		Gives the location to move existing files when overwrite is set to
		'archive'. If given as a relative path, this is relative to the
		dirname of `file`, not relative to the current working directory.
		Has no effect for other overwrite settings.

	"""

	with open_file_writer(filename, overwrite=overwrite, archive_dir=archive_dir, binary=True) as f:
		if compress:
			f.write(gzip.compress(pickle.dumps(obj)))
		else:
			f.write(pickle.dumps(obj))

def load(filename):
	"""
	Load an object from a file.

	Parameters
	----------

	filename : Path-like
		This is either a text or byte string giving the name (and the path
		if the file isn't in the current working directory) of the file to
		be written to.

	Returns
	-------
	object
	"""
	if not os.path.isfile(filename):
		raise FileNotFoundError(filename)
	content = None
	with open(filename, 'rb') as f:
		content = f.read()

	try:
		content = gzip.decompress(content)
	except:
		pass

	return pickle.loads(content)


def cache(filename, function, *args, **kwargs):
	"""
	Cache a function's return value to a file.

	Parameters
	----------
	filename : Path-like
		This is either a text or byte string giving the name (and the path
		if the file isn't in the current working directory) of the file to
		be used.
	function : Callable
		A function
	*args, **kwargs : Any
		Arguments to the function if the result is not cached.

	Returns
	-------
	object
	"""
	if os.path.isfile(filename):
		return load(filename)
	else:
		obj = function(*args, **kwargs)
		save(obj, filename, overwrite=False)
		return obj