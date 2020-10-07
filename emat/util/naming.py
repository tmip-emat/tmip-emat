
import pandas as pd
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

def multiindex_to_strings(index):
	if index.nlevels == 2:
		# index has both experiment_id and run_id
		tags = []
		for i in index:
			if pd.isna(i[1]):
				tags.append(f"{i[0]}")
			else:
				tags.append(f"{i[0]} {{{i[1]}}}")
		return tags
	else:
		return index

def reset_multiindex_to_strings(df):
	df = df.copy(deep=False)
	df.index = multiindex_to_strings(df.index)
	return df.reset_index()
