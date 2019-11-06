

class ShortnameMixin:
	"""
	Adds a shortname attribute, which falls back to name if not set.
	"""

	@property
	def shortname(self):
		"""Str: An abbreviated name, or the full name if not otherwise defined."""
		if not hasattr(self, '_shortname') or self._shortname is None:
			return self.name
		return self._shortname

	@shortname.setter
	def shortname(self, value):
		if value is None:
			self._shortname = None
		else:
			self._shortname = str(value)
			if self._shortname == self.name:
				self._shortname = None

	@shortname.deleter
	def shortname(self):
		self._shortname = None

	@property
	def shortname_if_any(self):
		"""Str: The abbreviated name, or None."""
		if not hasattr(self, '_shortname') or self._shortname is None:
			return None
		return self._shortname

