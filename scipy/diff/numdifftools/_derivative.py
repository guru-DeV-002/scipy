from __future__ import division
import numpy as np
from _step_generators import _generate_step
from scipy import misc
from scipy.ndimage.filters import convolve1d


def derivative(f, x, **options):
    x = np.asarray(x)
    method = options.pop('method', 'central')
    n = options.pop('n', 1)
    order = options.pop('order', 2)
    step = options.pop('step', None)
    step_ratio = options.pop('step_ratio', None)
    if step_ratio is None:
        if n == 1:
            step_ratio = 2.0
        else:
            step_ratio = 1.6
    if step is None:
        step = 'max_step'
    options.update(x=x)
    step_gen = _generate_step(**options)
    steps = [stepi for stepi in step_gen]
    fact = 1.0
    step_ratio_inv = 1.0 / step_ratio
    if n % 2 == 0 and method is 'central':
        fxi = f(x)
        results = [((f(x + h) + f(x - h)) / 2.0 - fxi) for h in steps]
        fd_step = 2
        offset = 2
    if n % 2 == 1 and method is 'central':
        fxi = 0.0
        results = [((f(x + h) - f(x - h)) / 2.0) for h in steps]
        fd_step = 2
        offset = 1
    if method is 'forward':
        fxi = f(x)
        results = [(f(x + h) - fxi) for h in steps]
        fd_step = 1
        offset = 1
    if method is 'backward':
        fxi = f(x)
        results = [(fxi - f(x - h)) for h in steps]
        fd_step = 1
        offset = 1
    fun = np.vstack(list(np.ravel(r)) for r in results)
    h = np.vstack(list(
            np.ravel(np.ones(np.shape(results[0]))*step)) for step in steps)
    # assert
    # n==0
    richardson_step = 1
    if method is 'central':
        richardson_step = 2
    richardson_order = max(
            (order // richardson_step) * richardson_step, richardson_step)
    richarson_terms = 2
    num_terms = (n+richardson_order-1) // richardson_step
    term = (n-1) // richardson_step
    c = fact / misc.factorial(
            np.arange(offset, fd_step * num_terms + offset, fd_step))
    [i, j] = np.ogrid[0:num_terms, 0:num_terms]
    fd = np.linalg.inv(np.atleast_2d(
                        c[j] * step_ratio_inv ** (i * (fd_step * j + offset))))
    if n % 2 == 0 and method is 'backward':
        fdi = -fd[term]
    else:
        fdi = fd[term]
    fdiff = convolve1d(fun, fdi[::-1], axis=0, origin=(fdi.size - 1) // 2)
    derivative = fdiff / (h ** n)
    return derivative


def fun(x):
    return x


print derivative(fun, [1, 2])
