# From https://github.com/jfc43/robust-ood-detection/blob/master/utils/lib.py
# Commit 7133fbaeb38efb64bb876e268a9008385aaa68c6

from __future__ import print_function
import numpy as np

def softmax(x):
	x = x - np.max(x)
	x = np.exp(x)/np.sum(np.exp(x))
	return x