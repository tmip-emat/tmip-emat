
import os
import xml.etree.ElementTree
import tempfile
import shutil

from .elem import Elem, Element
from .uid import uid as _uid
from . import styles

try:
	from local_logo import local_logo, local_favicon
except ImportError:
	local_logo = lambda *x: None
	local_favicon = lambda *x: None

TemporaryBucket = []

def TemporaryBucketCleanUp():
	global TemporaryBucket
	for i in TemporaryBucket:
		try:
			os.remove(os.path.realpath(i.name))
		except PermissionError:
			try:
				shutil.rmtree(i.name)
			except:
				import traceback
				#traceback.print_exc()
		except:
			import traceback
			#traceback.print_exc()
	del TemporaryBucket

import atexit
atexit.register(TemporaryBucketCleanUp)

def TemporaryFile(suffix='', mode='w+', use_chrome=True):
	t = tempfile.NamedTemporaryFile(suffix=suffix,mode=mode,delete=False)
	global TemporaryBucket
	TemporaryBucket.append(t)
	return t


def _try_write(self, content):
	try:
		self.write_(content)
	except:
		try:
			self.write_(str(content))
		except:
			self.write_(str(content).encode('utf-8'))


def TemporaryHtml(style=None, *, nohead=False, mode='wb+', content=None, **tagheads):
	t = TemporaryFile(suffix='.html', mode=mode)
	if 'b' in mode:
		t.write_ = t.write
		t.write = lambda x: _try_write(t,x)
	if not nohead and (style or len(tagheads)>0):
		t.write("<head>")
		if style:
			t.write("<style>{}</style>".format(style))
		for tag, content in tagheads.items():
			t.write("<{0}>{1}</{0}>".format(tag.lower(),content))
		t.write("</head>")
	if content is not None:
		t.write(content)
		t.view()
	return t


class XHTML():
	"""A class used to conveniently build xhtml documents."""

	def __init__(
			self,
			filename=None,
			*,
			overwrite=False,
			archive_dir='./archive/',
			title='Untitled',
			css=None,
			extra_css=None,
			jquery=True,
			jqueryui=True,
			floating_tablehead=True,
			metadata=None,
			toc=True,
			favicon=None,
	):
		"""

		Parameters
		----------
		filename : Path-like
			A filename to save to.

		overwrite : {True, False, 'spool', 'archive'}, default False
			Indicates what to do with an existing file at the same location.
			True will simply overwrite the existing file.
			False will raise a `FileExistsError`.
			'archive' will rename and/or move the existing file so that it
			will not be overwritten.
			'spool' will add a number to the filename of the file to be
			created, so that it will not overwrite the existing file.

		archive_dir : Path-like
			Gives the location to move existing files when overwrite is set to
			'archive'. If given as a relative path, this is relative to the
			dirname of `file`, not relative to the current working directory.
			Has no effect for other overwrite settings.

		title : str, optional
			Title for use for the HTML document.  Won't appear in the body of the
			document in a browser, but may be used in tabs or on the title bar.

		css : Path-like, optional
			An existing css file to load and embed in the file.

		extra_css : str, optional
			Additional css statements to append to the stylesheet loaded
			from `css` (if any).

		jquery, jqueryui : bool, defaults True
			Whether to include script links to jquery and jqueryui in the HTML file.

		floating_tablehead : bool, defaults True
			Whether to include script links to floating table head in the HTML file.

		metadata : Pickle-able object, optional
			An object that will be pickled and embedded in the file as meta-data.

		toc : bool, defaults True
			Whether to build and include the table of contents sidebar.

		favicon : bytes, optional
			Base64 encoded png for favicon
		"""
		self.root = Elem(tag="html", xmlns="http://www.w3.org/1999/xhtml")
		self.head = Elem(tag="head")
		self.body = Elem(tag="body")
		self.root << self.head
		self.root << self.body
		if filename is None or filename is False:
			import io
			filemaker = lambda: io.BytesIO()
		elif filename.lower() == "temp":
			filemaker = lambda: TemporaryHtml(nohead=True)
		else:
			from ..filez import open_file_writer
			filemaker = lambda: open_file_writer(filename, binary=True, overwrite=overwrite, archive_dir=archive_dir)
		self._filename = filename
		self._f = filemaker()
		self.title = Elem(tag="title")
		self.style = Elem(tag="style")

		if favicon is None:
			favicon = local_favicon()

		if favicon is not None:
			self.favicon = Elem(tag="link",
								attrib={'href': "data:image/png;base64,{}".format(favicon),
										'rel': "shortcut icon",
										'type': "image/png"}
								)
		else:
			self.favicon = None

		if jquery:
			self.jquery = Elem(tag="script", attrib={
				'src': "https://code.jquery.com/jquery-3.0.0.min.js",
				'integrity': "sha256-JmvOoLtYsmqlsWxa7mDSLMwa6dZ9rrIdtrrVYRnDRH0=",
				'crossorigin': "anonymous",
			})
			self.head << self.jquery

		if jqueryui:
			self.jqueryui = Elem(tag="script", attrib={
				'src': "https://code.jquery.com/ui/1.11.4/jquery-ui.min.js",
				'integrity': "sha256-xNjb53/rY+WmG+4L6tTl9m6PpqknWZvRt0rO1SRnJzw=",
				'crossorigin': "anonymous",
			})
			self.head << self.jqueryui

		if floating_tablehead:
			self.floatThead = Elem(tag="script", attrib={
				'src': "https://cdnjs.cloudflare.com/ajax/libs/floatthead/1.4.0/jquery.floatThead.min.js",
			})
			self.floatTheadA = Elem(tag="script")
			self.floatTheadA.text = """
			$( document ).ready(function() {
				var $table = $('table.floatinghead');
				$table.floatThead({ position: 'absolute' });
				var $tabledf = $('table.dataframe');
				$tabledf.floatThead({ position: 'absolute' });
			});
			$(window).on("hashchange", function () {
				window.scrollTo(window.scrollX, window.scrollY - 50);
			});
			"""
			self.head << self.floatThead
			self.head << self.floatTheadA

		self.head << self.favicon
		self.head << self.title
		self.head << self.style

		self.toc_color = 'lime'

		from .styles import default_css

		if toc:
			self.with_toc = True
			toc_width = 200
			default_css_ = default_css() + """

			body { margin-left: """ + str(toc_width) + """px; }
			.table_of_contents_frame { width: """ + str(
				toc_width - 13) + """px; position: fixed; margin-left: -""" + str(toc_width) + """px; top:0; padding-top:10px; z-index:2000;}
			.table_of_contents { width: """ + str(toc_width - 13) + """px; position: fixed; margin-left: -""" + str(
				toc_width) + """px; font-size:85%;}
			.table_of_contents_head { font-weight:700; padding-left:25px;  }
			.table_of_contents ul { padding-left:25px;  }
			.table_of_contents ul ul { font-size:75%; padding-left:15px; }
			.larch_signature {""" + styles.signature_font + """ width: """ + str(toc_width - 30) + """px; position: fixed; left: 0px; bottom: 0px; padding-left:20px; padding-bottom:2px; background-color:rgba(255,255,255,0.9);}
			.larch_name_signature {""" + styles.signature_name_font + """}
			a.parameter_reference {font-style: italic; text-decoration: none}
			.strut2 {min-width:2in}
			.histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }
			table.floatinghead thead {background-color:#FFF;}
			table.dataframe thead {background-color:#FFF;}
			@media print {
			   body { color: #000; background: #fff; width: 100%; margin: 0; padding: 0;}
			   /*.table_of_contents { display: none; }*/
			   @page {
				  margin: 1in;
			   }
			   h1, h2, h3 { page-break-after: avoid; }
			   img { max-width: 100% !important; }
			   ul, img, table { page-break-inside: avoid; }
			   .larch_signature {""" + styles.signature_font + """ padding:0; background-color:#fff; position: fixed; bottom: 0;}
			   .larch_name_signature {""" + styles.signature_name_font + """}
			   .larch_signature img {display:none;}
			   .larch_signature .noprint {display:none;}
			}
			"""
		else:
			self.with_toc = False
			default_css_ = default_css() + """

		   .larch_signature {""" + styles.signature_font + """ padding:0; background-color:#fff; }
			.larch_name_signature {""" + styles.signature_name_font + """}
			a.parameter_reference {font-style: italic; text-decoration: none}
			.strut2 {min-width:2in}
			.histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }
			table.floatinghead thead {background-color:#FFF;}
			table.dataframe thead {background-color:#FFF;}
			@media print {
			   body { color: #000; background: #fff; width: 100%; margin: 0; padding: 0;}
			   /*.table_of_contents { display: none; }*/
			   @page {
				  margin: 1in;
			   }
			   h1, h2, h3 { page-break-after: avoid; }
			   img { max-width: 100% !important; }
			   ul, img, table { page-break-inside: avoid; }
			   .larch_signature {""" + styles.signature_font + """ padding:0; background-color:#fff; position: fixed; bottom: 0;}
			   .larch_name_signature {""" + styles.signature_name_font + """}
			   .larch_signature img {display:none;}
			   .larch_signature .noprint {display:none;}
			}
			"""

		css = styles.load_css(css)

		self.title.text = str(title)

		if css is None:
			css = default_css_
		if extra_css is not None:
			css += extra_css
		self.style.text = css.replace('\n', ' ').replace('\t', ' ')

		if metadata is not None:
			import cloudpickle, base64, zlib
			_meta = (base64.standard_b64encode(zlib.compress(cloudpickle.dumps(metadata)))).decode()
			self.head << Elem(tag="meta", name='pythonmetadata', content=_meta)

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		if type or value or traceback:
			# traceback.print_exception(type, value, traceback)
			return False
		else:
			self.dump(toc=self.with_toc)
			self._f.close()

	def toc(self, insert=False):
		xtoc = Elem("div", {'class': 'table_of_contents'})
		logo = local_logo()
		if logo is not None:
			if isinstance(logo, bytes):
				logo = logo.decode()
			xtoc << Elem('img', attrib={'width': '150', 'src': "data:image/png;base64,{}".format(logo),
			                          'style': 'display: block; margin-left: auto; margin-right: auto'})
		xtoc << Elem('p', text="Table of Contents", attrib={'class': 'table_of_contents_head'})
		toclvl = 0
		current_toc = xtoc.put('div')
		xtoc_tree = [current_toc.put('ul')]

		min_anchor_lvl = 5
		for anchor in self.root.findall('.//a[@toclevel]'):
			anchor_lvl = int(anchor.get('toclevel'))
			if anchor_lvl < min_anchor_lvl:
				min_anchor_lvl = anchor_lvl

		for anchor in self.root.findall('.//a[@toclevel]'):
			anchor_ref = anchor.get('name')
			anchor_text = anchor.get('reftxt')
			anchor_lvl = int(anchor.get('toclevel')) - min_anchor_lvl + 1
			while anchor_lvl > len(xtoc_tree):
				xtoc_tree.append(xtoc_tree[-1].put('ul'))
			while anchor_lvl < len(xtoc_tree):
				xtoc_tree = xtoc_tree[:-1]
			xtoc_tree[-1] << (Elem('li') << Elem('a', text=anchor_text, attrib={'href': '#{}'.format(anchor_ref)}))
		if insert:
			self.body.insert(0, xtoc)
		return xtoc

	def toc_iframe(self, insert=False):
		css = """
		.table_of_contents { font-size:85%; """ + styles.body_font + """ }
		.table_of_contents a:link { text-decoration: none; }
		.table_of_contents a:visited { text-decoration: none; }
		.table_of_contents a:hover { text-decoration: underline; }
		.table_of_contents a:active { text-decoration: underline; }
		.table_of_contents_head { font-weight:700; padding-left:20px }
		.table_of_contents ul { padding-left:20px; }
		.table_of_contents ul ul { font-size:75%; padding-left:15px; }
		::-webkit-scrollbar {
			-webkit-appearance: none;
			width: 7px;
		}
		::-webkit-scrollbar-thumb {
			border-radius: 4px;
			background-color: rgba(0,0,0,.5);
			-webkit-box-shadow: 0 0 1px rgba(255,255,255,.5);
		"""
		xtoc_html = XHTML(css=css)
		xtoc_html.head << Elem(tag='base', target="_parent")
		xtoc_html.body << self.toc()

		BLAH = xml.etree.ElementTree.tostring(xtoc_html.root, method="html", encoding="unicode")

		from .colors import strcolor_rgb256
		toc_elem = Elem(tag='iframe', attrib={
			'class': 'table_of_contents_frame',
			'style': '''height:calc(100% - 100px); border:none; /*background-color:rgba(128,189,1, 0.95);*/
			  background: -webkit-linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* For Safari 5.1 to 6.0 */
			  background: -o-linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* For Opera 11.1 to 12.0 */
			  background: -moz-linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* For Firefox 3.6 to 15 */
			  background: linear-gradient(rgba({0}, 0.95), rgba(255, 255, 255, 0.95)); /* Standard syntax */
			'''.format(strcolor_rgb256(self.toc_color)),
			'srcdoc': BLAH,
		})

		if insert:
			self.body.insert(0, toc_elem)
		return toc_elem

	def sign(self, insert=False):
		xsign = Elem("div", {'class': 'larch_signature'})
		import time

		p = Elem('p')
		p << Elem('br', tail="Report generated on ")
		p << Elem('br', attrib={'class': 'noprint'}, tail=time.strftime("%A %d %B %Y "))
		p << Elem('br', attrib={'class': 'noprint'}, tail=time.strftime("%I:%M:%S %p"))
		xsign << p
		if insert:
			self.body.append(xsign)
		return xsign

	def finalize(self, toc=True, sign=True):
		if sign:
			self.sign(True)
		if toc:
			self.toc_iframe(True)

		c = self.root.copy()

		if sign:
			try:
				s = self.root.find(".//div[@class='larch_signature']/..")
			except TypeError:
				pass
			else:
				if s is not None:
					s.remove(s.find(".//div[@class='larch_signature']"))
		if toc:
			try:
				s = self.root.find(".//div[@class='table_of_contents']/..")
			except TypeError:
				pass
			else:
				if s is not None:
					s.remove(s.find(".//div[@class='table_of_contents']"))
		return c

	def dump(self, toc=True, sign=True):
		if sign:
			self.sign(True)
		if toc:
			self.toc_iframe(True)
		self._f.write(
			b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">')
		xml.etree.ElementTree.ElementTree(self.root).write(self._f, xml_declaration=False, method="html")
		self._f.flush()
		if sign:
			try:
				s = self.root.find(".//div[@class='larch_signature']/..")
			except TypeError:
				pass
			else:
				if s is not None:
					s.remove(s.find(".//div[@class='larch_signature']"))
		if toc:
			try:
				s = self.root.find(".//div[@class='table_of_contents']/..")
			except TypeError:
				pass
			else:
				if s is not None:
					s.remove(s.find(".//div[@class='table_of_contents']"))
		try:
			return self._f.getvalue()  # for BytesIO
		except AttributeError:
			return

	def dump_seg(self):
		xml.etree.ElementTree.ElementTree(self.root).write(self._f, xml_declaration=False, method="html")
		self._f.flush()
		try:
			return self._f.getvalue()  # for BytesIO
		except AttributeError:
			return

	def view(self):
		try:
			self._f.view()
		except AttributeError:
			pass

	def append(self, node):
		if isinstance(node, Element):
			self.body.append(node)
		elif hasattr(node, '__xml__'):
			self.body.append(node.__xml__())
		elif node is None:
			pass
		else:
			raise TypeError(
				"must be xml.etree.ElementTree.Element or XML_Builder or TreeBuilder or something with __xml__ defined, not {!s}".format(
					type(node)))

	def __lshift__(self, other):
		self.append(other)
		return self

	def anchor(self, ref, reftxt, cls, toclevel):
		self.append(Elem(tag="a", attrib={'name': ref, 'reftxt': reftxt, 'class': cls, 'toclevel': toclevel}))

	def hn(self, n, content, attrib=None, anchor=None):
		if attrib is None:
			attrib = {}
		if anchor:
			h_elem = Elem(tag="h{}".format(n), attrib=attrib)
			h_elem.put("a", {'name': _uid(), 'reftxt': anchor if isinstance(anchor, str) else content, 'class': 'toc',
			                 'toclevel': '{}'.format(n)}, tail=content)
		else:
			h_elem = Elem(tag="h{}".format(n), attrib=attrib, text=content)
		self.append(h_elem)





def load_metadata(filename):
	"""
	Extract metadata from an HTML file previously saved by `XHTML`.

	Parameters
	----------
	filename : Path-like
		The filename of the HTML file that contains the metadata.

	Returns
	-------
	object
		The unpickled meta-data saved in the file.
	"""
	import pickle, zlib, base64
	if not os.path.exists(filename):
		raise FileNotFoundError(filename)

	if (len(filename)>5 and filename[-5:]=='.html') or (len(filename)>6 and filename[-6:]=='.xhtml'):
		from html.parser import HTMLParser
		class HTMLParser_MetaDataLoader(HTMLParser):
			def handle_starttag(subself, tag, attrs):
				global self
				if tag=='meta':
					use = False
					for attrname,attrval in attrs:
						if attrname=='name' and attrval=='pythonmetadata':
							use = True
					if use:
						for attrname,attrval in attrs:
							if attrname=='content':
								attrval = base64.standard_b64decode(attrval.encode())
								attrval = zlib.decompress(attrval)
								subself.self = pickle.loads(attrval)
		parser = HTMLParser_MetaDataLoader()
		with open(filename) as f:
			parser.feed(f.read())
		try:
			if parser.self is None:
				raise ValueError('nothing loaded')
		except AttributeError:
			raise ValueError('nothing found to load')
		return parser.self
	else:
		raise ValueError('must load html')


