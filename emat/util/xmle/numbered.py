
from .elem import Elem
from .uid import uid

class NumberedCaption:
	def __init__(self, kind, level=2):
		self._kind = kind
		self._level = level
	def __call__(self, caption, anchor=None, level=None, attrib=None, **extra):
		n = level if level is not None else self._level
		result = Elem("h{}".format(n))
		if anchor:
			result.put("a", {
				'name': uid(),
				'reftxt': anchor if isinstance(anchor, str) else caption,
				'class': 'toc',
				'toclevel': '{}'.format(n)
			}, )

		result.put("span", {
				'class': f'larch_{self._kind.lower().replace(" ","_")}_caption larch_caption',
				'larch_caption':self._kind,
		}, tail=caption, text=f"{self._kind}: ")
		return result
