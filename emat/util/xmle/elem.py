

import os
import re
import inspect
import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, TreeBuilder, XMLParser
from contextlib import contextmanager
from .uid import uid as _uid
import base64
import cloudpickle, pickle
import pandas
from io import BytesIO, StringIO, BufferedIOBase, TextIOBase


xml.etree.ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
xml.etree.ElementTree.register_namespace("xlink", "http://www.w3.org/1999/xlink")

def Show(arg, **kwargs):
	return Elem.from_any(arg, **kwargs)

class Elem(Element):
	"""Extends :class:`xml.etree.ElementTree.Element`"""

	def __init__(self, tag, attrib=None, text=None, tail=None, html_repr=1, **extra):
		if isinstance(tag, Element):
			extra.update(attrib)
			Element.__init__(self, tag.tag, tag.attrib, **extra)
			for s in list(tag):
				self.append(s)
		else:
			if attrib is None:
				attrib = {}
			if 'cls' in extra:
				extra['class'] = extra.pop('cls')
			if isinstance(text, Element):
				Element.__init__(self, tag, attrib, **extra)
				for k, v in text.attrib.items():
					if k not in attrib and k not in extra:
						self.set(k, v)
				self.text = text.text
				if tail:
					self.tail = text.tail + tail
				else:
					self.tail = text.tail
			else:
				Element.__init__(self, tag, attrib, **extra)
				if text: self.text = str(text)
				if tail: self.tail = str(tail)

	@classmethod
	def from_any(cls, arg, **kwargs):
		if isinstance(arg, bytes) and arg[:5] == b'<svg ':
			return cls.from_string(arg)
		if isinstance(arg, str) and arg[:5] == '<svg ':
			return cls.from_string(arg)
		if isinstance(arg, bytes) and arg[:6] == b'<?xml ':
			return cls.from_string(arg)
		if isinstance(arg, str) and arg[:6] == '<?xml ':
			return cls.from_string(arg)
		if isinstance(arg, bytes) and arg[:4] == b'\x89PNG':
			return cls.from_png_raw(arg)
		if isinstance(arg, bytes) and arg[:5] == b'iVBOR':
			return cls.from_png_b64(arg.decode())
		if isinstance(arg, str) and arg[:5] == 'iVBOR':
			return cls.from_png_b64(arg)
		if hasattr(arg, '__xml__'):
			try:
				return cls(arg.__xml__())
			except:
				pass
		if hasattr(arg, 'get_png'):
			try:
				return cls.from_any(arg.get_png())
			except:
				pass
		if isinstance(arg, str) and arg[:2] == '# ':
			try:
				return cls.from_heading(1, arg[2:])
			except:
				pass
		if isinstance(arg, str) and arg[:3] == '## ':
			try:
				return cls.from_heading(2, arg[3:])
			except:
				pass
		if isinstance(arg, str) and arg[:4] == '### ':
			try:
				return cls.from_heading(3, arg[4:])
			except:
				pass
		if isinstance(arg, str) and arg[:5] == '#### ':
			try:
				return cls.from_heading(4, arg[5:])
			except:
				pass
		if hasattr(arg, '_repr_html_'):
			try:
				return cls.from_string(arg._repr_html_())
			except:
				pass
		if isinstance(arg, str):
			try:
				return cls.from_rst(arg)
			except:
				pass
		if isinstance(arg, bytes):
			try:
				return cls.from_bytes(arg)
			except:
				raise ValueError(f"cannot create Elem from {arg}")
		if 'matplotlib' in str(type(arg)):
			import matplotlib.figure
			if isinstance(arg, matplotlib.figure.Figure):
				try:
					return cls.from_figure(arg)
				except:
					pass
		if isinstance(arg, pandas.DataFrame):
			return cls.from_dataframe(arg)
		if 'plotly' in str(type(arg)) or any('plotly' in str(i) for i in inspect.getmro(type(arg))):
			import plotly.io as pio
			# import platform
			# if platform.system() == 'Windows':
			# 	import sys
			# 	possible_orca = sys.executable.replace("python.exe","orca.cmd")
			# 	if os.path.exists(possible_orca):
			# 		import plotly.io.orca as orca
			# 		orca.config.executable = possible_orca
			try:
				img_bytes = pio.to_image(arg, **kwargs)
				return cls.from_any(img_bytes)
			except:
				pass
		if hasattr(arg, 'figure'):
			try:
				return cls.from_any(arg.figure)
			except:
				pass
		raise ValueError(f"cannot create Elem from {arg}")

	@classmethod
	def from_heading(cls, heading_level, heading_text, attrib=None, **extra):
		if attrib is None:
			attrib = {}
		anchor = True
		if "|" in heading_text:
			heading_text, anchor = heading_text.split("|", 1)
			anchor = anchor.strip()
		heading_text = heading_text.strip()
		self = cls(tag=f'h{heading_level}', attrib=attrib, **extra)
		self.put("a",
				 {
					 'name': _uid(),
					 'reftxt': anchor if isinstance(anchor, str) else heading_text,
					 'class': 'toc',
					 'toclevel': '{}'.format(heading_level)
				 },
				 tail=heading_text,
				 )
		return self

	@classmethod
	def from_string(cls, xml_as_string):
		if isinstance(xml_as_string, bytes):
			xml_as_string = xml_as_string.decode()
		try:
			return xml.etree.ElementTree.fromstring(xml_as_string, parser=XMLParser(target=TreeBuilder(element_factory=cls)))
		except xml.etree.ElementTree.ParseError:
			return cls.from_string(xml_as_string.replace("<style scoped>","<style scoped='1'>"))

	@classmethod
	def from_bytes(cls, xml_as_bytes):
		xml_as_string = xml_as_bytes.decode()
		try:
			return xml.etree.ElementTree.fromstring(xml_as_string, parser=XMLParser(target=TreeBuilder(element_factory=cls)))
		except xml.etree.ElementTree.ParseError:
			return cls.from_string(xml_as_string.replace("<style scoped>","<style scoped='1'>"))

	@classmethod
	def from_rst(cls, rst_as_string):
		if isinstance(rst_as_string, bytes):
			rst_as_string = rst_as_string.decode()
		from .restructuredtext import rst_to_xhtml
		return rst_to_xhtml(rst_as_string, factory=cls)

	@classmethod
	def from_dataframe(cls, df, **kwargs):
		if isinstance(df, pandas.DataFrame):
			return xml.etree.ElementTree.fromstring(df.to_html(**kwargs), parser=XMLParser(target=TreeBuilder(element_factory=cls)))
		elif isinstance(df, pandas.io.formats.style.Styler):
			render = df.render()
			render = re.sub(
				r"colspan=([1234567890]*)>",
				"colspan=\"\g<1>\">",
				render, 0)
			try:
				return xml.etree.ElementTree.fromstring(f"<div>{render}</div>", parser=XMLParser(target=TreeBuilder(element_factory=cls)))
			except xml.etree.ElementTree.ParseError as parse_err:
				x = Elem('div')
				x << xml.etree.ElementTree.fromstring(df.data.to_html(**kwargs), parser=XMLParser(target=TreeBuilder(element_factory=cls)))
				x << Elem('pre', text=render)
				x << Elem('pre', text=str(parse_err))
				return x

	@classmethod
	def from_figure(cls, fig, format='svg', transparent=True, tooltip=None, bbox_inches='tight', classname='figure', close_after=True, **kwargs):
		"""
		Constructor from matplotlib Figure.

		Parameters
		----------
		fig : matplotlib.figure.Figure or obj with get_figure

		"""
		from matplotlib import pyplot as plt
		import matplotlib.figure


		if not isinstance(fig, matplotlib.figure.Figure):
			try:
				fig = fig.get_figure()
			except AttributeError:
				if not hasattr(fig, 'savefig'):
					raise TypeError('fig must be a figure or provide `get_figure` or `savefig` method.')

		try:
			fig_number = fig.number
		except AttributeError:
			fig_number = None
		else:
			this_fig = plt.figure(fig_number) # activate current figure

		existing_format_keys = list(kwargs.keys())
		for key in existing_format_keys:
			if key.upper() != key: kwargs[key.upper()] = kwargs[key]
		if 'GRAPHWIDTH' not in kwargs and 'GRAPHHEIGHT' in kwargs:
			kwargs['GRAPHWIDTH'] = kwargs['GRAPHHEIGHT']
		if 'GRAPHWIDTH' in kwargs and 'GRAPHHEIGHT' not in kwargs:
			kwargs['GRAPHHEIGHT'] = kwargs['GRAPHWIDTH'] * .67
		imgbuffer = BytesIO()
		fig.savefig(
			imgbuffer,
			dpi=None,
			facecolor='w',
			edgecolor='w',
			orientation='portrait',
			format=format,
			transparent=transparent,
			bbox_inches=bbox_inches,
			pad_inches=0.1,
		)
		x = cls("div", {'class': classname})
		try:
			x << cls.from_any(imgbuffer.getvalue())
		except:
			print(imgbuffer.getvalue())
			raise
		if tooltip is not None and format=='svg':
			x[0][1].insert(0, cls("title", text=tooltip))

		if close_after and fig_number is not None:
			plt.close(fig_number)
		return x

	def put(self, tag, attrib=None, text=None, tail=None, **extra):
		if 'cls' in extra:
			extra['class'] = extra.pop('cls')
		if attrib is None:
			attrib = {}
		attrib = attrib.copy()
		attrib.update(extra)
		element = Elem(tag, attrib)
		if text: element.text = str(text)
		if tail: element.tail = str(tail)
		self.append(element)
		return element

	elem = put

	def __call__(self, *arg, **attrib):
		for a in arg:
			if isinstance(a, dict):
				for key, value in a.items():
					self.set(str(key), str(value))
			if isinstance(a, str):
				if self.text is None:
					self.text = a
				else:
					self.text += a
		for key, value in attrib.items():
			self.set(str(key), str(value))
		return self

	def append(self, arg):
		try:
			super().append(arg)
		except TypeError:
			if callable(arg):
				super().append(arg())
			else:
				if isinstance(arg, bytes) and arg[:5] == b'<svg ':
					super().append(Elem.from_string(arg))
				elif isinstance(arg, str) and arg[:5] == '<svg ':
					super().append(Elem.from_string(arg))
				elif isinstance(arg, str) and arg[:6] == '<?xml ':
					super().append(Elem.from_string(arg))
				elif isinstance(arg, bytes) and arg[:4] == b'\x89PNG':
					super().append(Elem.from_png_raw(arg))
				elif hasattr(arg, '__xml__'):
					super().append(arg.__xml__())
				elif hasattr(arg, 'get_png'):
					self.append(arg.get_png())
				elif isinstance(arg, str) and arg[:2] == '# ':
					self.hn_(1, arg[2:])
				elif isinstance(arg, str) and arg[:3] == '## ':
					self.hn_(2, arg[3:])
				elif isinstance(arg, str) and arg[:4] == '### ':
					self.hn_(3, arg[4:])
				elif isinstance(arg, str) and arg[:5] == '#### ':
					self.hn_(4, arg[5:])
				elif hasattr(arg, '_repr_html_'):
					super().append(Elem.from_string(arg._repr_html_()))
				elif isinstance(arg, str):
					super().append(Elem.from_rst(arg))
				else:
					raise

	def __lshift__(self, other):
		if other is not None:
			if isinstance(other, list):
				for i in other:
					self.append(i)
			else:
				self.append(other)
		return self

	def tobytes(self):
		return xml.etree.ElementTree.tostring(self, encoding="utf8", method="html")

	def tostring(self):
		return self.tobytes().decode()

	@property
	def is_svg(self):
		if len(self) == 1 and self[0].tag[-3:]=='svg':
			return True
		elif self.tag[-3:]=='svg':
			return True
		else:
			pass # raise TypeError("must be a svg element, or a div containing only a svg")
			return False

	def to_svg_doc(self, filename=None, mode='w'):
		if not self.is_svg:
			raise TypeError("must be a svg element, or a div containing only a svg")
		x = """<?xml version="1.0" standalone="no"?>
		<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
		"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
		"""+xml.etree.ElementTree.tostring(self[0]).decode()
		if filename is None:
			return x
		import pathlib
		if isinstance(filename, (str, pathlib.Path)):
			if os.path.splitext(filename)[1] == '.svgz':
				import gzip
				with gzip.open(filename, mode) as f:
					f.write(x)
			else:
				with open(filename, mode) as f:
					f.write(x)
		else:
			# assume filename is file-like obj
			filename.write(x)
			filename.flush()

	def to_png(self, filename=None):
		if not self.is_svg:
			raise TypeError("must be a svg element, or a div containing only a svg")
		import cairosvg
		import tempfile
		tempfile = tempfile.NamedTemporaryFile(suffix='.svg', mode='w+', delete=False)
		self.to_svg_doc(filename=tempfile)
		return cairosvg.svg2png(
			url=tempfile.name,
			write_to=filename,
		)

	@classmethod
	def from_png_raw(cls, png_raw):
		return cls(
			tag='img',
			src="data:image/png;base64,{}".format(base64.standard_b64encode(png_raw).decode()),
		)

	@classmethod
	def from_png_b64(cls, png_b64):
		return cls(
			tag='img',
			src="data:image/png;base64,{}".format(png_b64),
		)

	def _repr_html_(self):
		return self.tostring()
		# if self.__do_html_repr == 0:
		# 	return None
		# elif self.__do_html_repr == -1:
		# 	return self[-1]._repr_html_()
		# elif self.__do_html_repr == -2:
		# 	result = "".join(_._repr_html_() for _ in self[self.__seen_html_repr:])
		# 	self.__seen_html_repr = len(self)
		# 	return result
		# else:
		# 	return self.tostring()

	def pprint(self, indent=0, hush=False):
		dent = "  " * indent
		if hush and self.tag in ('div', 'tfoot', 'a') and not self.text and len(self) == 0:
			if self.tail:
				return dent + self.tail
			else:
				return ""
		s = "{}<{}>".format(dent, self.tag)
		if self.text:
			s += self.text
		any_subs = False
		for i in self():
			if isinstance(i, Elem):
				sub = i.pprint(indent=indent + 1, hush=True)
				if sub:
					s += "\n{}".format(sub)
					any_subs = True
			else:
				s += "\n{}{}".format(dent, i)
				any_subs = True
		if any_subs:
			s += "\n{}".format(dent)
		s += "</{}>".format(self.tag)
		if self.tail:
			s += self.tail
		return s

	def __repr__(self):
		return "<xmle.Elem '{}' with {} children>".format(self.tag, (len(self)))  # +self.pprint()

	def save(self, filename, overwrite=True):
		"""
		Save this Elem to a file.

		Parameters
		----------
		filename : Path-like or File-like
		overwrite : bool or str
			If False, files will not be overwritten.  Or give 'archive:...' to copy an existing
			file to the indicated archive location.

		"""
		if isinstance(overwrite, str) and overwrite[:8] == 'archive:':
			filedirname, filebasename = os.path.split(filename)

			if os.path.isabs(overwrite[8:]):
				# archive path is absolute
				archive_path = overwrite[8:]
			else:
				# archive path is relative
				archive_path = os.path.normpath(os.path.join(filedirname, overwrite[8:]))
			os.makedirs(archive_path, exist_ok=True)
			if os.path.exists(filename):
				# send existing file to archive
				from .file_util import next_filename, creation_date, append_date_to_filename
				import shutil
				epoch = creation_date(filename)
				new_name = next_filename(
					append_date_to_filename( os.path.join(archive_path, filebasename), epoch),
					allow_natural=True
				)
				shutil.move(filename, new_name)
			overwrite = False
		if isinstance(filename, BufferedIOBase):
			filename.write(self.tobytes())
		elif isinstance(filename, TextIOBase):
			filename.write(self.tostring())
		else:
			if os.path.exists(filename) and not overwrite:
				raise FileExistsError("file {0} already exists".format(filename))
			with open(filename, 'wt') as f:
				f.write(self.tostring())

	def anchor(self, ref, reftxt, cls, toclevel):
		self.put("a", {'name': ref, 'reftxt': reftxt, 'class': cls, 'toclevel': toclevel})

	def hn(self, n, content, attrib=None, anchor=None, **extra):
		if attrib is None:
			attrib = {}
		if anchor:
			h_elem = self.put("h{}".format(n), attrib, **extra)
			h_elem.put("a", {'name': _uid(), 'reftxt': anchor if isinstance(anchor, str) else content, 'class': 'toc',
			                 'toclevel': '{}'.format(n)}, tail=content)
		else:
			self.put("h{}".format(n), attrib, text=content, **extra)

	def hn_(self, n, content, attrib=None):
		anchor = True
		if "|" in content:
			content, anchor = content.split("|", 1)
			anchor = anchor.strip()
		self.hn(n, content.strip(), anchor=anchor, attrib=attrib)

	def __xml__(self):
		return self

	@property
	def metadata(self):
		if 'metadata' in self.attrib:
			return pickle.loads( base64.standard_b64decode(self.attrib['metadata'].encode()) )
		return None

	@metadata.setter
	def metadata(self, meta):
		self.attrib['metadata'] = base64.standard_b64encode(cloudpickle.dumps(meta)).decode()

	def __eq__(self, other):
		import cloudpickle
		if not isinstance(other, Elem):
			return False
		return (cloudpickle.dumps(self) == cloudpickle.dumps(other))

	# def render_html(self, width=1024, trim_uniform_border=False):
	# 	from .temporaryfile import TemporaryFile
	# 	from .visual_processing import screenshot
	# 	with TemporaryFile(suffix='.html') as t:
	# 		t.write(self.tostring())
	# 		t.flush()
	# 		s = screenshot(
	# 			"file:///" + t.name,
	# 			window_size=(width, None),
	# 		)
	#
	# 	if trim_uniform_border:
	# 		from .visual_processing import trim_uniform_border as trim_uniform_border_
	# 		from io import BytesIO
	# 		from PIL import Image
	# 		im = Image.open(BytesIO(base64.b64decode(s)))
	# 		im = trim_uniform_border_( im )
	# 		buffered = BytesIO()
	# 		im.save(buffered, format="PNG")
	# 		s = base64.b64encode( buffered.getvalue() ).decode()
	# 	return Elem.from_png_b64(s)
	#




class ElemTable(Elem):

	def __str__(self):
		from .html_table_to_txt import xml_table_to_txt
		try:
			return xml_table_to_txt(self)
		except IndexError:
			return "<xmle.ElemTable render fail>"

	def __repr__(self):
		from .html_table_to_txt import xml_table_to_txt
		try:
			return xml_table_to_txt(self)
		except IndexError:
			return "<xmle.ElemTable render fail>"
