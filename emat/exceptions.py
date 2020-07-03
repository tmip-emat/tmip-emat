
class ScopeError(Exception):
	pass


class ScopeFormatError(ScopeError):
	pass


class PendingExperimentsError(ValueError):
	pass


class MissingModelPathError(FileNotFoundError):
	pass


class MissingArchivePathError(FileNotFoundError):
	pass


class AsymmetricCorrelationError(ValueError):
	"""Two conflicting values are given for correlation of two parameters."""


class DistributionTypeError(TypeError):
	"""The distribution is expected to be continuous but it is actually discrete, or vice versa."""


class DistributionFreezeError(Exception):
	"""An error is thrown when creating an rv_frozen object."""


class MissingIdWarning(Warning):
	"""An experiment id is not found in the database."""