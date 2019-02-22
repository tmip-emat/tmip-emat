
class dicta(dict):
	'''Dictionary with attribute access.'''
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			if '_helper' in self:
				return self['_helper'](name)
			raise AttributeError(name)
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	def __repr__(self):
		if self.keys():
			m = max(map(len, list(self.keys()))) + 1
			return '\n'.join([k.rjust(m) + ': ' + repr(v) for k, v in self.items()])
		else:
			return self.__class__.__name__ + "()"

class Counter(dicta):

	def one(self, key):
		if key in self:
			self[key] += 1
		else:
			self[key] = 1

	def add(self, other_counter):
		for key, val in other_counter.items():
			if key in self:
				self[key] += val
			else:
				self[key] = val


import contextlib

@contextlib.contextmanager
def noop_wrapper():
	yield