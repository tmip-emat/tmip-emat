import numpy as np
import re

tiers = [
	'y', # yocto
	'z', # zepto
	'a', # atto
	'f', # femto
	'p', # pico
	'n', # nano
	'µ', # micro
	'm', # milli
	'',  #
	'K', # Kilo
	'M', # Mega
	'G', # Giga
	'T', # Tera
	'P', # Peta
	'E', # Exa
	'Z', # Zeta
	'Y', # Yotta
]

def si_units(x, kind='', f="{:.3g}{}{}"):
	tier = 8
	shift = 1024 if kind=='B' else 1000
	while np.absolute(x) > shift and tier < len(tiers):
		x /= shift
		tier += 1
	while np.absolute(x) < 1 and tier >= 0:
		x *= shift
		tier -= 1
	# Convert modest fractions back to simple decimals
	if tiers[tier] == 'm' and x > 10:
		x /= shift
		tier += 1
	return f.format(x,tiers[tier],kind)

_si_float = re.compile("^\s*([0-9]+\.?[0-9]*)\s?([yzafpnµmKMGTPEZY])\s*$")
_si_int = re.compile("^\s*([0-9]+)\s?([yzafpnµmKMGTPEZY])\s*$")

def get_float(x):
	try:
		return float(x)
	except ValueError:
		match = _si_float.match(x)
		if match is None:
			raise
		else:
			y = float(match.group(1))
			power = (tiers.index(match.group(2))-8)*3
			return y * 10**power

def get_int(x):
	try:
		return int(x)
	except ValueError:
		match = _si_int.match(x)
		if match is None:
			raise
		else:
			y = int(match.group(1))
			power = (tiers.index(match.group(2))-8)*3
			return y * 10**power