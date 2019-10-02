

def versions():
	from . import __version__
	v = {'emat':__version__}
	import ema_workbench, numpy, pandas, plotly
	v['ema_workbench'] = ema_workbench.__version__
	v['plotly'] = plotly.__version__
	print(", ".join(f"{key} {value}" for key, value in v.items()))
