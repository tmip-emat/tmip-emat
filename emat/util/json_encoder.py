# From: https://github.com/hmallen/numpyencoder
#
# MIT License
#
# Copyright (c) 2018 Hunter Allen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
	""" Custom encoder for numpy data types """

	def default(self, obj):
		if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
		                    np.int16, np.int32, np.int64, np.uint8,
		                    np.uint16, np.uint32, np.uint64)):

			return int(obj)

		elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
			return float(obj)

		elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
			return {'real': obj.real, 'imag': obj.imag}

		elif isinstance(obj, (np.ndarray,)):
			return obj.tolist()

		elif isinstance(obj, (np.bool_)):
			return bool(obj)

		elif isinstance(obj, (np.void)):
			return None

		return json.JSONEncoder.default(self, obj)


def dumps(*args, **kwargs):
	return json.dumps(*args, **kwargs, cls=NumpyEncoder)