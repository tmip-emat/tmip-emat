
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('outdir', type=str, help='sphinx output directory')
args = parser.parse_args()


import re

duplicate_tag = '''(<script src="https://unpkg.com/@jupyter-widgets/html-manager@\^[0-9]*\.[0-9]*\.[0-9]*/dist/embed-amd.js"></script>)'''
bad1 = re.compile(duplicate_tag)
bad2 = re.compile(duplicate_tag+"(.*)"+duplicate_tag)


def dedupe_jupyter_widgets_manager(filename):
	with open(filename, 'rt') as html_in:
		content = html_in.read()
	n = len(bad1.findall(content))
	if n>1:
		content_1 = bad1.sub("", content, count=n-1)
		print(f"FIXING [{n}]:",filename)
		with open(filename, 'wt') as html_out:
			html_out.write(content_1)
	else:
		print(f"PASSED [{n}]:",filename)

def fixing_walker(filename):
	directory = (os.path.abspath(filename))
	for dirpath, dirnames, filenames in os.walk(directory):
		for f in filenames:
			if f[-5:]==".html":
				this_file = os.path.join(dirpath, f)
				dedupe_jupyter_widgets_manager(this_file)


fixing_walker(args.outdir)

