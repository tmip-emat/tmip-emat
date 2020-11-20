
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


class DatabaseVersionWarning(Warning):
	"""The database requires a more recent version of emat."""


class DatabaseError(Exception):
	"""A generic database error."""


class DatabaseVersionError(DatabaseError):
	"""The database requires a more recent version of emat."""


class ReadOnlyDatabaseError(DatabaseError):
	"""Writing to a readonly database is prohibited."""


class MissingMeasuresWarning(Warning):
	"""Some experiments have performance measures that are missing."""


class MissingMeasuresError(ValueError):
	"""Some experiments have performance measures that are missing."""


class DesignExistsError(DatabaseError):
	"""Attempting to create a design that already exists."""
