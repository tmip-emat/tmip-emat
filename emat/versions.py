

def versions():
	from . import __version__
	v = {'emat':__version__}
	from . import workbench
	import numpy, pandas, plotly
	v['workbench'] = workbench.__version__
	v['plotly'] = plotly.__version__
	print(", ".join(f"{key} {value}" for key, value in v.items()))
