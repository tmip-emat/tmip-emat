
import hashlib
from typing import Collection

def hash_it(*args, ha=None):
	if ha is None:
		ha = hashlib.sha1()
	for a in args:
		try:
			ha.update(a)
		except:
			if isinstance(a, str):
				ha.update(a.encode())
			elif isinstance(a, Collection):
				hash_it(*a, ha=ha)
			elif hasattr(a, '_hash_it'):
				hash_it(a._hash_it(ha=ha))
			elif isinstance(a, (int, float)):
				ha.update(str(a).encode())
			elif a is None:
				ha.update(b"None")
			else:
				print("cant hashit",(type(a)))
				print(a)
				raise
	return ha.hexdigest()
