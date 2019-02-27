

def get_name(i, from_first_item=False):
	"""
	Get a name from an object.

	Parameters
	----------
	i
		The object from which to get a name.
		If the object is a string, it is its own name.
		Otherwise, check if it has a name and use that.
	from_first_item : bool, default False
		If no name is available from the object and this
		is True, try to get a name from the first member
		of the object, assuming the object is a collection.
		If there is any problem (e.g., the object is not
		a collection, or the first item has no name)
		then silently ignore this.

	Returns
	-------
	str
		The name, or empty string if there is no name.
	"""
	if isinstance(i, str):
		return i
	if hasattr(i, 'name') and isinstance(i.name, str):
		return i.name
	if from_first_item:
		try:
			i0 = i[0]
		except:
			pass
		else:
			return get_name(i0)
	return ''

def any_names(I):
	"""
	Check if any member of a collection has a name.

	Parameters
	----------
	I : Iterable

	Returns
	-------
	bool
	"""
	for i in I:
		if get_name(i):
			return True
	return False