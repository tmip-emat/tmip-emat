
class ScopeError(Exception):
	pass

class ScopeFormatError(ScopeError):
	pass

class PendingExperimentsError(ValueError):
	pass

class MissingArchivePathError(FileNotFoundError):
	pass