
# Based on https://stackoverflow.com/a/49912639
# LICENSE: CC BY-SA 3.0  https://creativecommons.org/licenses/by-sa/3.0/

import os
from pathlib import Path

class DisplayablePath(object):
	display_filename_prefix_middle = '├──'
	display_filename_prefix_last = '└──'
	display_parent_prefix_middle = '    '
	display_parent_prefix_last = '│   '

	def __init__(self, path, parent_path, is_last):
		self.path = Path(str(path))
		self.parent = parent_path
		self.is_last = is_last
		if self.parent:
			self.depth = self.parent.depth + 1
		else:
			self.depth = 0

	@property
	def displayname(self):
		if self.path.is_dir():
			return self.path.name + '/'
		return self.path.name

	@classmethod
	def make_tree(cls, root, parent=None, is_last=False, criteria=None):
		root = Path(str(root))
		criteria = criteria or cls._default_criteria

		if os.path.basename(os.path.normpath(root)) == '.git': return

		displayable_root = cls(root, parent, is_last)
		yield displayable_root

		children = sorted(list(path
							   for path in root.iterdir()
							   if criteria(path)),
						  key=lambda s: str(s).lower())
		count = 1
		for path in children:
			is_last = count == len(children)
			if path.is_dir():
				yield from cls.make_tree(path,
										 parent=displayable_root,
										 is_last=is_last,
										 criteria=criteria)
			else:
				yield cls(path, displayable_root, is_last)
			count += 1

	@classmethod
	def _default_criteria(cls, path):
		return True

	def displayable(self):
		if self.parent is None:
			return self.displayname

		_filename_prefix = (self.display_filename_prefix_last
							if self.is_last
							else self.display_filename_prefix_middle)

		parts = ['{!s} {!s}'.format(_filename_prefix,
									self.displayname)]

		parent = self.parent
		while parent and parent.parent is not None:
			parts.append(self.display_parent_prefix_middle
						 if parent.is_last
						 else self.display_parent_prefix_last)
			parent = parent.parent

		return ''.join(reversed(parts))

def show_dir(basedir):
	paths = DisplayablePath.make_tree(Path(basedir))
	for path in paths:
		if path.displayname in (".DS_Store", ): continue
		print(path.displayable())

def show_file_contents(*filepath, mode='rt'):
	if os.path.splitext(os.path.join(*filepath))[1]=='.gz' and 'z' not in mode:
		mode = f"{mode}z"
	if 'z' in mode:
		mode = mode.replace('z','')
		import gzip
		with gzip.open(os.path.join(*filepath), mode) as f:
			print(f.read())
	else:
		with open(os.path.join(*filepath), mode) as f:
			print(f.read())
