from __future__ import division
import numpy as np
from _epsilon_generator import _epsilon

def derivative(f,x,method='central',eps=None):
	''' Dervative'''
	x = np.asarray(x)

	if method is 'central':
		s = 3
	else:
		s = 2

	epsilon = _epsilon(x,s,eps)
	der = np.empty(x.size)

	for i in range(x.size):
		der[i] = (f(x[i] + epsilon[i]) - f(x[i]))/epsilon[i]

	return der




