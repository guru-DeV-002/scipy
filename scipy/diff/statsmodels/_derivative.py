from __future__ import division
import numpy as np
from _epsilon_generator import _epsilon

def Derivative(f,x,**options):
    '''
    Derivative of a function

    Parameters
    ----------
    x : array
        parameters at which the derivative is evaluated

    f : function
        `f(x)` returning one value.
        `f(x)` should be univariate.

    **options : dict
        options for specifying the method and epsilon.
        method : {'central','foward'}.
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.

    Returns
    -------
    der : array
        derivative
    '''

    x = np.asarray(x)
    method = 'central'
    if 'method' in options:
        method = options['method']

    eps = None
    if 'eps' in options:
        eps = options['eps']    

    if method is 'central':
        s = 3
    else:
        s = 2

    epsilon = _epsilon(x,s,eps)
    der = np.empty(x.size)
    der = (f(x+epsilon)-f(x))/epsilon
    return der

def Gradient(f,x,**options):
    '''
    Gradient of a function

    Parameters
    ----------
    x : 2d array
        parameters at which the derivative is evaluated.

    f : function
        `f(x)` returning one value.

    **options : dict
        options for specifying the method and epsilon.
        method : {'central','foward'}.
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.

    Returns
    -------
    grad : array
        gradient
    '''

    x = np.asarray(x)

    method = 'central'
    if 'method' in options:
        method = options['method']

    eps = None
    if 'eps' in options:
        eps = options['eps']

    if method is 'central':
        s = 3
    else:
        s = 2

    h = _epsilon(x,s,eps)
    h_transpose = np.transpose(h)
    epsilon = np.repeat(h_transpose,x.shape[1],axis=1)
    identity = np.column_stack([np.identity(x.shape[1])]*x.shape[0])
    epsilon = np.multiply(np.asarray(identity),np.asarray(epsilon))
    x_transpose = np.transpose(x)
    x_transpose = np.repeat(x_transpose,x.shape[1],axis=1)
    grad = np.reshape(f(*(x_transpose + epsilon))-f(*x_transpose),(x.shape))
    grad = grad/h
    return grad

def Jacobian(f,x,**options):
    '''
    Jacobian of a function

    Parameters
    ----------
    x : 2d array
        parameters at which the derivative is evaluated.

    f : function

    **options : dict
        options for specifying the method and epsilon.
        method : {'central','foward'}.
        Stepsize, if None, optimal stepsize is used. This is EPS**(1/2)*x for
        `centered` == False and EPS**(1/3)*x for `centered` == True.

    Returns
    -------
    jac : array
        Jacobian
    '''

    x = np.asarray(x)

    method = 'central'
    if 'method' in options:
        method = options['method']

    eps = None
    if 'eps' in options:
        eps = options['eps']

    if method is 'central':
        s = 3
    else:
        s = 2

    h = _epsilon(x,s,eps)
    h_transpose = np.transpose(h)
    epsilon = np.repeat(h_transpose,x.shape[1],axis=1)
    identity = np.column_stack([np.identity(x.shape[1])]*x.shape[0])
    epsilon = np.multiply(np.asarray(identity),np.asarray(epsilon))
    x_transpose = np.transpose(x)
    x_transpose = np.repeat(x_transpose,x.shape[1],axis=1)
    f0 = np.asarray(f(*x_transpose))

    if f0.ndim == 1:
        f0 = np.reshape(f(*x_transpose),(1,f0.shape[0]))

    jac = np.asarray(f(*(x_transpose + epsilon)))-f0
    shape = jac.shape
    jac = np.reshape(jac,((shape[0],)+(x.shape)))
    jac = jac/h
    return jac.swapaxes(1,0)
    