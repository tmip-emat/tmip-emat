# encoding: utf-8



import textwrap
from docutils.core import publish_parts
from itertools import chain
from .elem import Elem

class DictOfRST(dict):
	@staticmethod # because this doesn't make sense as a global function.
	def _process_args(mapping=(), **kwargs):
		if hasattr(mapping, 'items'):
			mapping = getattr(mapping, 'items')()
		return ((k, v) for k, v in chain(mapping, getattr(kwargs, 'items')()))
	def __init__(self, mapping=(), **kwargs):
		if mapping is None:
			mapping = ()
		super().__init__(self._process_args(mapping, **kwargs))
	def __getitem__(self, k):
		return super().__getitem__(str(k))
	def __setitem__(self, k, v):
		return super().__setitem__(str(k), str(v))
	def setdefault(self, k, default=""):
		return super().setdefault(str(k), str(default))
	def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
		return type(self)(self)
	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, super().__repr__())
	def __xml__(self):
		from pprint import pformat
		from .elem import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		t.elem('caption', text=f"<xmle.{self.__class__.__name__}>", style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;font-style:normal;font-size:100%;padding:0px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="key")
			tr.elem('th', text='value')
			for k,v in self.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				tr.elem('td') << rst_to_xhtml(v)
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x
	def _repr_html_(self):
		return self.__xml__().tostring()
	def xml(self, key):
		return rst_to_xhtml(self[key])


def rst_to_xhtml(blurb_rst, h_stepdown=2, *, factory=None, **format):
	if factory is None:
		factory = Elem
	if isinstance(blurb_rst, bytes):
		blurb_rst = blurb_rst.decode()
	if not isinstance(blurb_rst, str):
		raise TypeError('blurb must be reStructuredText as str ot bytes')
	blurb_rst = textwrap.dedent(blurb_rst).strip()
	blurb_div = factory.from_string(publish_parts(blurb_rst, writer_name='html')['html_body'])
	blurb_div.attrib['class'] = 'blurb'
	for hlevel in (8,7,6,5,4,3,2,1):
		for bh1 in blurb_div.iter('h{}'.format(hlevel)):
			bh1.tag = 'h{}'.format(hlevel+h_stepdown)
	return blurb_div


def cut_extra_whitespace(x):
	if x is None: return ""
	x = x.replace('\t',' ').replace('\n',' ')
	x1 = x.replace('  ',' ')
	while x1!=x:
		x = x1
		x1 = x.replace('  ',' ')
	return x


class RstRenderer(object):
	"""
	Service class that knows how to render a RestructuredText string to
	a python-docx Document object.
	"""
	def __init__(self, blkcntnr, rst, style_overrides={}):
		self._blkcntnr = blkcntnr
		self._rst = rst
		self._style_overrides = style_overrides
		self._depth = 1


	def render(self, h_stepdown=2):
		"""
		Parse the RestructuredText in *rst* and render it into *blkcntnr* as
		paragraphs, bullets, etc., including recognizing and rendering bold
		and italic runs within block elements.
		"""
		t = rst_to_xhtml(self._rst, h_stepdown=h_stepdown)
#		from pprint import pprint
#		pprint(ElementTree.tostring(t).decode())
		self._render_container(t)

	@property
	def _styles(self):
		"""
		The dict providing lookup for style names for this RST document.
		"""
		if not hasattr(self, '_styles_'):
			self._styles_ = {
				'h1': 'Heading 1',
				'h2': 'Heading 2',
				'h3': 'Heading 3',
				'h4': 'Heading 4',
				'h5': 'Heading 5',
				'h6': 'Heading 6',
				'h7': 'Heading 7',
				'h8': 'Heading 8',
				'p':  'Body Text',
				'li': 'List Bullet',
				'b':  'Strong',
				'i':  'Emphasis',
			}
			self._styles_.update(self._style_overrides)
		return self._styles_

	def _render_container(self, container):
		"""
		Render each element in *container* in turn.
		"""
		for element in container:
			tag = element.tag
#			print("_ "*self._depth,'tag:',tag)
#			print("_ "*self._depth,'  text:',((element.text.replace('\n',' ')) if element.text is not None else ""))
#			print("_ "*self._depth,'  tail:',((element.tail.replace('\n',' ')) if element.tail is not None else ""))
			if tag == 'section':
				self._render_container(element)
			elif tag == 'div':
				self._render_container(element)
			elif tag == 'blockquote':
				self._render_blockquote(element)
			elif tag == 'ul':
				self._render_container(element)
			elif tag == 'title':
				self._render_paragraph(element, self._styles['h1'])
			elif tag in set('h{}'.format(z) for z in range(1,9)) or tag in ('p', 'i', 'b'):
				self._render_paragraph(element, self._styles[tag])
			elif tag == 'paragraph':
				self._render_paragraph(element, self._styles['p'])
			elif tag == 'transition':
				self._render_transition(self._styles['p'])
			elif tag == 'hr':
				self._render_transition(self._styles['p'])
			elif tag == 'li':
				#self._render_bullet_list(element)
				self._render_paragraph(element, self._styles['li'], depth=self._depth)
			else:
				raise NotImplementedError('unrecognized tag %s' % tag)

	@property
	def _rst_etree(self):
		"""
		Return the root element of a RestructuredText XML document produced by
		converting *rst* to XML and then parsing that XML using lxml.
		"""
		return rst_to_xhtml(self._rst)


	def _render_blockquote(self, bq):
		"""
		Add one level of depth to the contents.
		"""
		self._depth += 1
		try:
			for sub_item in bq:
				self._render_container(sub_item)
		finally:
			self._depth -= 1




	def _render_bullet_list(self, bullet_list):
		"""
		Add a bullet to *blkcntnr* for each list item in *bullet_list*.
		"""
		def render_list_item(list_item):
			for idx, para in enumerate(list_item):
				style_key = 'li' if idx == 0 else 'lc'
				self._render_paragraph(para, self._styles[style_key])

		for list_item in bullet_list:
			render_list_item(list_item)

	def _render_paragraph(self, para, style, depth=None):
		"""
		Add a new paragraph to *blkcntnr* containing the content in the
		`paragraph` element *para*. Create appropriate runs for text having
		strong and emphasis inline formatting.
		"""
#		print("RENDER PARAG style",style)
		if depth and depth>1:
			paragraph = self._blkcntnr.add_paragraph(style=style+" {}".format(depth))
		else:
			paragraph = self._blkcntnr.add_paragraph(style=style)
		if para.text is not None:
			paragraph.add_run(cut_extra_whitespace(para.text))
		for child in para:
#			print("child.tag",child.tag,child.text,child.tail)
			if child.tag in ('p',):
				paragraph.add_run(cut_extra_whitespace(child.text)+" ")
				paragraph.add_run(cut_extra_whitespace(child.tail)+ " ")
			elif child.tag in ('strong','b','em','i'):
				style_key = {'strong': 'b', 'em':'i','b':'b', 'i':'i'}.get(child.tag)
				if child.text is not None:
					paragraph.add_run(cut_extra_whitespace(child.text), self._styles[style_key])
				if child.tail is not None:
					paragraph.add_run(cut_extra_whitespace(child.tail))
			elif child.tag=='span':
				for sub_item in child:
					self._render_span(paragraph, sub_item, {'strong': 'b', 'em':'i','b':'b', 'i':'i', 'sub':'i', }.get(sub_item.tag))
				paragraph.add_run(cut_extra_whitespace(child.tail)+ " ")
			else:
				self._depth += 1
				try:
					for sub_item in para:
						self._render_container(sub_item)
				finally:
					self._depth -= 1

	def _render_span(self, paragraph, span, style):
#		print("RENDER SPAN style",style, ":", span.text, span.tail)
		if style=='b':
			paragraph.add_run(span.text).bold = True
		elif style=='i':
			paragraph.add_run(span.text).italic = True
		else:
			paragraph.add_run(span.text)
		paragraph.add_run(span.tail)


	def _render_transition(self, style='p'):
		"""
		Add a new paragraph to *blkcntnr* containing the content in the
		`paragraph` element *para*. Create appropriate runs for text having
		strong and emphasis inline formatting.
		"""
		paragraph = self._blkcntnr.add_paragraph(style=style)
		paragraph.add_run('---------------------------------')



