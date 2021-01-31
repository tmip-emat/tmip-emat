

def versions():
	from . import __version__
	v = {'emat':__version__}
	from . import workbench
	import numpy, pandas, plotly
	v['plotly'] = plotly.__version__
	print(", ".join(f"{key} {value}" for key, value in v.items()))

def require_version(n, plotly=None):
	from . import __version__
	from packaging import version
	if version.parse(n) > version.parse(__version__):
		raise ValueError("the installed emat is version {}".format(__version__))
	if plotly is not None:
		import plotly as _plotly
		if version.parse(plotly) > version.parse(_plotly.__version__):
			raise ValueError("the installed plotly is version {}".format(_plotly.__version__))
	versions()
