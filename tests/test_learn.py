

import pandas
from emat.learn.feature_selection import SelectUniqueColumns


def test_select_unique_columns():

	df = pandas.DataFrame({
		'Aa': [1,2,3,4,5,6,7],
		'Bb': [4,6,5,4,6,2,2],
		'Cc': [1,2,3,4,5,6,7],
		'Dd': [4,5,6,7,8,8,2],
		'Ee': [10,20,30,40,50,60,70],
		'Ff': [44,55,66,77,88,88,22],
	})
	s = SelectUniqueColumns().fit(df)
	pandas.testing.assert_frame_equal(s.transform(df), df[['Aa','Bb','Dd']])

