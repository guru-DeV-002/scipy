from __future__ import division
import numpy as np

EPS = np.MachAr().eps

def approx_fprime(f,x,method='central',eps=None):
	''' Dervative'''

	if method is 'central':
		if eps is None:
			eps = EPS**(1./3)*np.maximum(np.abs(x), 0.1)/2
	else:
		if eps is None:
			eps = EPS**(1./2)*np.maximum(np.abs(x), 0.1)

	der = (f(x+eps) - f(x))/eps

	return der






