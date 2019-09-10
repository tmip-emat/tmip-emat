

import re

def clean_name(s):

	# Ensure s is a string
	s = str(s)

	# Remove spaces at beginning and end
	s = s.strip(' ')

	# Convert all other spaces to underscores
	s = s.replace(' ', '_')

	# Remove other invalid characters
	s = re.sub('[^0-9a-zA-Z_]', '', s)

	# Remove leading characters until we find a letter or underscore
	s = re.sub('^[^a-zA-Z_]+', '', s)

	return s
