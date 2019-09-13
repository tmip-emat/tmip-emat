
from ipywidgets import Box, widget_selection, ToggleButton, Layout, Label, Widget, Checkbox
import traitlets


class MultiToggleButtons_AllOrSome(Box):
	description = traitlets.Unicode()
	value = traitlets.Tuple()
	options = traitlets.Union([traitlets.List(), traitlets.Dict()])
	style = traitlets.Dict()

	def __init__(self, *, short_label_map=None, **kwargs):
		if short_label_map is None:
			short_label_map = {}
		super().__init__(**kwargs)
		self._selection_obj = widget_selection._MultipleSelection()
		traitlets.link((self, 'options'), (self._selection_obj, 'options'))
		traitlets.link((self, 'value'), (self._selection_obj, 'value'))

		@observer(self, 'options')
		def _(*_):
			self.buttons = []
			for label in self._selection_obj._options_labels:
				short_label = short_label_map.get(label, label)
				self.buttons.append(ToggleButton(
					description=short_label if len(short_label)<15 else short_label[:12]+"…",
					tooltip=label,
					layout=Layout(
						margin='1',
						width='auto',
					),
				))
			if self.description:
				self.label = Label(self.description, layout=Layout(width=self.style.get('description_width', '100px')))
			else:
				self.label = Label(self.description, layout=Layout(width=self.style.get('description_width', '0px')))
			self.children = [self.label]+self.buttons

			@observer(self.buttons, 'value')
			def _(*_):
				proposed_value = tuple(value
								   for btn, value in zip(self.buttons, self._selection_obj._options_values)
								   if btn.value)
				# When nothing is selected, treat as if everything is selected.
				if len(proposed_value) == 0:
					proposed_value = tuple(value
									   for btn, value in zip(self.buttons, self._selection_obj._options_values)
									   )
				self.value = proposed_value

		self.add_class('btn-group')

	def reset(self):
		opts = self.options
		self.options = []
		self.options = opts

	def set_value(self, x):
		for b, opt in zip(self.buttons, self.options):
			b.value = (opt in x)

	def set_all_on(self):
		for b, opt in zip(self.buttons, self.options):
			b.value = True

	def set_all_off(self):
		for b, opt in zip(self.buttons, self.options):
			b.value = False


def observer(widgets, trait_name):
	def wrapper(func):
		if isinstance(widgets, Widget):
			widgets.observe(func, trait_name)
		else:
			for w in widgets:
				w.observe(func, trait_name)
		func()

	return wrapper


from traitlets import Unicode
from ipywidgets import DOMWidget, register



@register
class NamedCheckbox(Checkbox):
	name = Unicode('').tag(sync=True)


