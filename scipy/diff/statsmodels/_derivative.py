from __future__ import division
import numpy as np
from _epsilon_generator import _epsilon

def derivative(f,x,method='central',eps=None):
	'''
    Derivative of a function

    Parameters
    ----------
    x : array
        parameters at which the derivative is evaluated
    f : function
        `f(x)` returning one value.
    method : {'central','foward'}
		method for computing the derivative
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.
    Returns
    -------
    der : array
        derivative
    '''

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
