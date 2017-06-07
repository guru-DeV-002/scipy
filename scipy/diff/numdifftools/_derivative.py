from __future__ import division
import numpy as np
from collections import namedtuple
from scipy.ndimage.filters import convolve1d
from scipy import misc
from numpy import linalg
import scipy.linalg as lin
import warnings

_STATE = namedtuple('State', ['x', 'method', 'n', 'order']) 
EPS = np.finfo(float).eps
_EPS = EPS
_TINY = np.finfo(float).tiny
FD_RULES = {}
info = namedtuple('info', ['error_estimate', 'final_step', 'index'])

def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)

def _central_even(f, f_x0i, x0i, h, *args, **kwds):
    return (f(x0i + h, *args, **kwds) +
            f(x0i - h, *args, **kwds)) / 2.0 - f_x0i

def _central(f, f_x0i, x0i, h, *args, **kwds):
    return (f(x0i + h, *args, **kwds) -
            f(x0i - h, *args, **kwds)) / 2.0

def _forward(f, f_x0i, x0i, h, *args, **kwds):
    return f(x0i + h, *args, **kwds) - f_x0i

def _backward(f, f_x0i, x0i, h, *args, **kwds):
    return f_x0i - f(x0i - h, *args, **kwds)

def valarray(shape, value=np.NaN, typecode=None):
    if typecode is None:
        typecode = bool
    out = np.ones(shape, dtype=typecode) * value

    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out

def _vstack(sequence, steps):
        original_shape = np.shape(sequence[0])
        f_del = np.vstack(list(np.ravel(r)) for r in sequence)
        h = np.vstack(list(np.ravel(np.ones(original_shape)*step))
                      for step in steps)
        _assert(f_del.size == h.size, 'fun did not return data of correct '
                'size (it must be vectorized)')
        return f_del, h, original_shape

def _fd_matrix(step_ratio, parity, nterms):
        """
        Return matrix for finite difference and complex step derivation.

        Parameters
        ----------
        step_ratio : real scalar
            ratio between steps in unequally spaced difference rule.
        parity : scalar, integer
            0 (one sided, all terms included but zeroth order)
            1 (only odd terms included)
            2 (only even terms included)
            3 (only every 4'th order terms included starting from order 2)
            4 (only every 4'th order terms included starting from order 4)
            5 (only every 4'th order terms included starting from order 1)
            6 (only every 4'th order terms included starting from order 3)
        nterms : scalar, integer
            number of terms
        """
        _assert(0 <= parity <= 6,
                'Parity must be 0, 1, 2, 3, 4, 5 or 6! ({0:d})'.format(parity))
        step = [1, 2, 2, 4, 4, 4, 4][parity]
        inv_sr = 1.0 / step_ratio
        offset = [1, 1, 2, 2, 4, 1, 3][parity]
        c0 = [1.0, 1.0, 1.0, 2.0, 24.0, 1.0, 6.0][parity]
        c = c0 / \
            misc.factorial(np.arange(offset, step * nterms + offset, step))
        [i, j] = np.ogrid[0:nterms, 0:nterms]
        return np.atleast_2d(c[j] * inv_sr ** (i * (step * j + offset)))

def convolve(sequence, rule, **kwds):
    """Wrapper around scipy.ndimage.convolve1d that allows complex input."""
    dtype = np.result_type(float, np.ravel(sequence)[0])
    seq = np.asarray(sequence, dtype=dtype)
    if np.iscomplexobj(seq):
        return (convolve1d(seq.real, rule, **kwds) + 1j *
                convolve1d(seq.imag, rule, **kwds))
    return convolve1d(seq, rule, **kwds)

def MaxStep(**options):
    x = options.pop('x',np.asarray(1))
    if x is None:
        x = np.asarray(1)
    n = options.pop('n',1)
    if n is None:
        n = 1
    order = options.pop('order',2)
    if order is None:
        order = 2
    method = options.pop('method','forward')
    if method is None:
        method = 'forward'
    print x
    if x is None:
        y = 1.0
    else:
        y = np.log1p(np.abs(x)).clip(min=1.0)
    step_nom = options.pop('step_nom',None)
    if step_nom is None:
        step_nom = y
    else:
        step_nom = valarray(x.shape, value=self._step_nom)
    default_step_ratio = 1.6
    if n == 1:
        default_step_ratio = 2.0
    step_ratio = options.pop('step_ratio',default_step_ratio)
    if step_ratio is None:
        step_ratio = default_step_ratio
    num_steps = options.pop('num_steps',15)
    offset = options.pop('offest',0)
    if offset is None:
        offset = 0
    num_extrap = options.pop('num_extrap',0)
    if num_extrap is None:
        num_extrap = 0
    use_exact_steps=options.pop('use_exact_steps',True)
    if use_exact_steps is None:
        use_exact_steps = True
    check_num_steps=options.pop('check_num_steps',True)
    if check_num_steps is None:
        check_num_steps = True
    order2 = max(order // 2 - 1, 0)
    default_scale = 2.5 + int(n-1)*1.3 + order2*dict(central=3, forward=2, backward=2).get(method, 0)
    scale = options.pop('scale',default_scale)
    if scale is None:
        scale = 500
    base_step = options.pop('base_step',2.0)
    if base_step is None:
        base_step = EPS ** (1./scale)
    num_steps2=int(n+order-1)
    if method in ['central', 'central2']:
        num_steps2 = num_steps2 // step
    min_num_steps = max(num_steps2,1)
    if num_steps is not None:
        num_steps = int(num_steps)
        if check_num_steps:
            num_steps = max(num_steps, min_num_steps)
    else:
        num_steps = min_num_steps + int(num_extrap)
    base_step = base_step * step_nom
    if use_exact_steps:
        base_step = (base_step + 1.0) - 1.0
        step_ratio = (step_ratio + 1.0) -1.0
    for i in range(num_steps):
        step = base_step * step_ratio ** (-i + offset)
        if (np.abs(step) > 0).all():
            yield step

def MinStep(**options):
    x = options.pop('x',np.asarray(1))
    if x is None:
        x = np.asarray(1)
    n = options.pop('n',1)
    if n is None:
        n = 1
    order = options.pop('order',2)
    if order is None:
        order = 2
    method = options.pop('method','forward')
    if method is None:
        method = 'forward'
    if x is None:
        y = 1.0
    else:
        y = np.log1p(np.abs(x)).clip(min=1.0)
    step_nom = options.pop('step_nom',None)
    if step_nom is None:
        step_nom = y
    else:
        step_nom = valarray(x.shape, value=self._step_nom)
    default_step_ratio = 1.6
    if n == 1:
        default_step_ratio = 2.0
    step_ratio = options.pop('step_ratio',default_step_ratio)
    if step_ratio is None:
        step_ratio = default_step_ratio
    num_steps = options.pop('num_steps',None)
    offset = options.pop('offest',0)
    if offset is None:
        offset = 0
    num_extrap = options.pop('num_extrap',0)
    if num_extrap is None:
        num_extrap = 0
    use_exact_steps=options.pop('use_exact_steps',True)
    if use_exact_steps is None:
        use_exact_steps = True
    check_num_steps=options.pop('check_num_steps',True)
    if check_num_steps is None:
        check_num_steps = True
    order2 = max(order // 2 - 1, 0)
    default_scale = 2.5 + int(n-1)*1.3 + order2*dict(central=3, forward=2, backward=2).get(method, 0)
    scale = options.pop('scale',default_scale)
    if scale is None:
        scale = default_scale
    base_step = options.pop('base_step',EPS ** (1. / scale))
    num_steps2=int(n+order-1)
    if method in ['central', 'central2']:
        num_steps2 = num_steps2 // step
    min_num_steps = max(num_steps2,1)
    if num_steps is not None:
        num_steps = int(num_steps)
        if check_num_steps:
            num_steps = max(num_steps, min_num_steps)
    else:
        num_steps = min_num_steps + int(num_extrap)
    base_step = base_step * step_nom
    if use_exact_steps:
        base_step = (base_step + 1.0) - 1.0
        step_ratio = (step_ratio + 1.0) -1.0
    for i in range(num_steps-1,-1,-1):
            step = base_step * step_ratio ** (i + offset)
            if (np.abs(step) > 0).all():
                yield step

def dea3(v0, v1, v2, symmetric=False):
    """
    Extrapolate a slowly convergent sequence

    Parameters
    ----------
    v0, v1, v2 : array-like
        3 values of a convergent sequence to extrapolate

    Returns
    -------
    result : array-like
        extrapolated value
    abserr : array-like
        absolute error estimate

    Description
    -----------
    DEA3 attempts to extrapolate nonlinearly to a better estimate
    of the sequence's limiting value, thus improving the rate of
    convergence. The routine is based on the epsilon algorithm of
    P. Wynn, see [1]_.

     Example
     -------
     # integrate sin(x) from 0 to pi/2

     >>> import numpy as np
     >>> import numdifftools as nd
     >>> Ei= np.zeros(3)
     >>> linfun = lambda i : np.linspace(0, np.pi/2., 2**(i+5)+1)
     >>> for k in np.arange(3):
     ...    x = linfun(k)
     ...    Ei[k] = np.trapz(np.sin(x),x)
     >>> [En, err] = nd.dea3(Ei[0], Ei[1], Ei[2])
     >>> truErr = Ei-1.
     >>> (truErr, err, En)
     (array([ -2.00805680e-04,  -5.01999079e-05,  -1.25498825e-05]),
     array([ 0.00020081]), array([ 1.]))

     See also
     --------
     dea

     Reference
     ---------
     .. [1] C. Brezinski and M. Redivo Zaglia (1991)
            "Extrapolation Methods. Theory and Practice", North-Holland.

    ..  [2] C. Brezinski (1977)
            "Acceleration de la convergence en analyse numerique",
            "Lecture Notes in Math.", vol. 584,
            Springer-Verlag, New York, 1977.

    ..  [3] E. J. Weniger (1989)
            "Nonlinear sequence transformations for the acceleration of
            convergence and the summation of divergent series"
            Computer Physics Reports Vol. 10, 189 - 371
            http://arxiv.org/abs/math/0306302v1
    """
    e0, e1, e2 = np.atleast_1d(v0, v1, v2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore division by zero and overflow
        delta2, delta1 = e2 - e1, e1 - e0
        err2, err1 = np.abs(delta2), np.abs(delta1)
        tol2, tol1 = max_abs(e2, e1) * _EPS, max_abs(e1, e0) * _EPS
        delta1[err1 < _TINY] = _TINY
        delta2[err2 < _TINY] = _TINY  # avoid division by zero and overflow
        ss = 1.0 / delta2 - 1.0 / delta1 + _TINY
        smalle2 = abs(ss * e1) <= 1.0e-3
        converged = (err1 <= tol1) & (err2 <= tol2) | smalle2
        result = np.where(converged, e2 * 1.0, e1 + 1.0 / ss)
    abserr = err1 + err2 + np.where(converged, tol2 * 10, np.abs(result - e2))
    if symmetric and len(result) > 1:
        return result[:-1], abserr[1:]
    return result, abserr

def max_abs(a1, a2):
    return np.maximum(np.abs(a1), np.abs(a2))

def _estimate_error(new_sequence, old_sequence, steps, rule):
    m = new_sequence.shape[0]
    mo = old_sequence.shape[0]
    cov1 = np.sum(rule**2)  # 1 spare dof
    fact = np.maximum(12.7062047361747 * np.sqrt(cov1), EPS * 10.)
    if mo < 2:
        return (np.abs(new_sequence) * EPS + steps) * fact
    if m < 2:
        delta = np.diff(old_sequence, axis=0)
        tol = max_abs(old_sequence[:-1], old_sequence[1:]) * fact
        err = np.abs(delta)
        converged = err <= tol
        abserr = (err[-m:] +
                  np.where(converged[-m:], tol[-m:] * 10,
                           abs(new_sequence - old_sequence[-m:]) * fact))
        return abserr
#         if mo>2:
#             res, abserr = dea3(old_sequence[:-2], old_sequence[1:-1],
#                               old_sequence[2:] )
#             return abserr[-m:] * fact
    err = np.abs(np.diff(new_sequence, axis=0)) * fact
    tol = max_abs(new_sequence[1:], new_sequence[:-1]) * EPS * fact
    converged = err <= tol
    abserr = err + np.where(converged, tol * 10,
                            abs(new_sequence[:-1] -
                                old_sequence[-m+1:]) * fact)
    return abserr

def _r_matrix(num_terms,step,step_ratio,order):
    i, j = np.ogrid[0:num_terms + 1, 0:num_terms]
    r_mat = np.ones((num_terms + 1, num_terms + 1))
    r_mat[:, 1:] = (1.0 / step_ratio) ** (i * (step * j + order))
    return r_mat

def rule(num_terms,step,step_ratio,order,sequence_length=None,):
    if sequence_length is None:
        sequence_length = num_terms + 1
    num_terms = min(num_terms, sequence_length - 1)
    if num_terms > 0:
        r_mat = _r_matrix(num_terms,step,step_ratio,order)
        return lin.pinv(r_mat)[0]
    return np.ones((1,))

def Richardson(sequence,steps,order,num_terms,step,step_ratio):
    ne = sequence.shape[0]
    rule1 = rule(num_terms,step,step_ratio,order,ne)
    nr = rule1.size - 1
    m = ne - nr
    mm = min(ne, m+1)
    new_sequence = convolve(sequence, rule1[::-1], axis=0, origin=nr // 2)
    abserr = _estimate_error(new_sequence[:mm], sequence, steps, rule1)
    return new_sequence[:m], abserr[:m], steps[:m]

def _get_arg_min(errors):
    shape = errors.shape
    try:
        arg_mins = np.nanargmin(errors, axis=0)
        min_errors = np.nanmin(errors, axis=0)
    except ValueError as msg:
        warnings.warn(str(msg))
        return np.arange(shape[1])

    for i, min_error in enumerate(min_errors):
        idx = np.flatnonzero(errors[:, i] == min_error)
        arg_mins[i] = idx[idx.size // 2]
    return np.ravel_multi_index((arg_mins, np.arange(shape[1])), shape)

def _add_error_to_outliers(der, trim_fact=10):
    # discard any estimate that differs wildly from the
    # median of all estimates. A factor of 10 to 1 in either
    # direction is probably wild enough here. The actual
    # trimming factor is defined as a parameter.
    try:
        median = np.nanmedian(der, axis=0)
        p75 = np.nanpercentile(der, 75, axis=0)
        p25 = np.nanpercentile(der, 25, axis=0)
        iqr = np.abs(p75-p25)
    except ValueError as msg:
        warnings.warn(str(msg))
        return 0 * der

    a_median = np.abs(median)
    outliers = (((abs(der) < (a_median / trim_fact)) +
                (abs(der) > (a_median * trim_fact))) * (a_median > 1e-8) +
                ((der < p25-1.5*iqr) + (p75+1.5*iqr < der)))
    errors = outliers * np.abs(der - median)
    return errors

def _get_best_estimate(der, errors, steps, shape):
    errors += _add_error_to_outliers(der)
    ix = _get_arg_min(errors)
    final_step = steps.flat[ix].reshape(shape)
    err = errors.flat[ix].reshape(shape)
    return der.flat[ix].reshape(shape), info(err, final_step, ix)

def _wynn_extrapolate(der, steps):
    der, errors = dea3(der[0:-2], der[1:-1], der[2:], symmetric=False)
    return der, errors, steps[2:]

def extrapolate(order,num_terms,step,step_ratio,results,steps,shape):
    der1, errors1, steps = Richardson(results, steps,order,num_terms,step,step_ratio)
    if len(der1) > 2:
        der1, errors1, steps = _wynn_extrapolate(der1, steps)
    der, info = _get_best_estimate(der1, errors1, steps, shape)
    return der, info

def derivative(fun,x,args=(),kwds={},**options):
    method = options.pop('method','central')
    if method is None:
        method = 'central'
    n = options.pop('n',1)
    if n is None:
        n = 1
    order = options.pop('order',2)
    if order is None:
        order = 2
    step = options.pop('step',None)

    if step is 'MaxStep':
        MaxStep(**options)
    if step is 'MinStep':
        MinStep(**options)
    else:
        step_options = dict(x=x,step_ratio=None, num_extrap=14)
        if step is None:
            step_options.update(**options)
            step_gen = MaxStep(**step_options)
            steps =  [step for step in step_gen]
        else:
            step_options['num_extrap'] = 0
            step_options['base_step'] = step
            step_options.update(**options)
            step_gen = MinStep(**step_options)
            steps =  [step for step in step_gen]
    default_step_ratio = 1.6
    if n == 1:
        default_step_ratio = 2.0
    step_ratio = options.pop('step_ratio',default_step_ratio)
    if step_ratio is None:
        step_ratio = default_step_ratio
    print steps,step_ratio
    eval_first_check = (n%2==0 and method in ('central', 'central2') or method in ['forward', 'backward'])
    if not eval_first_check:
        x = np.asarray(x)
        fxi = f(x,*args,**kwds)
    else:
        fxi = 0.0
    if n%2==0 and method in ('central'):
        results = [_central_even(f, fxi, x, h, *args, **kwds) for h in steps]
    if n%2==1 and method in ('central'):
        results = [_central(f, fxi, x, h, *args, **kwds) for h in steps]
    if method is 'forward':
        results = [_forward(f, fxi, x, h, *args, **kwds) for h in steps]
    if method is 'backward':
        results = [_backward(f, fxi, x, h, *args, **kwds) for h in steps]
    richardson_step = 1
    if method in ('central','central2'):
        richardson_step = 2
    richardson_order = max((order // richardson_step) * richardson_step, richardson_step)
    richarson_terms = 2
    f_del, h, original_shape = _vstack(results,steps)
    if n == 0:
        fd_rule = np.ones((1,))
    else:
        if method.startswith('central'):
            parity = ((n-1)%2) + 1
        else:
            parity = 0
        order1 = n-1
        num_terms1, ix = (order1 + richardson_order) // richardson_step, order1 // richardson_step
        fd_rules = FD_RULES.get((step_ratio, parity, num_terms1))
        if fd_rules is None:
            fd_mat = _fd_matrix(step_ratio, parity, num_terms1)
            fd_rules = linalg.pinv(fd_mat)
            FD_RULES[(step_ratio, parity, num_terms1)] = fd_rules

        if n%2==0 and (method == 'backward'):
            fd_rule = -fd_rules[ix]
        else:
            fd_rule = fd_rules[ix]
    ne = h.shape[0]
    nr = fd_rule.size - 1
    _assert(nr < ne, 'num_steps ({0:d}) must  be larger than '
                '({1:d}) n + order - 1 = {2:d} + {3:d} -1'
                ' ({4:s})'.format(ne, nr+1, n, order, method))
    f_diff = convolve(f_del, fd_rule[::-1], axis=0, origin=nr // 2)
    der_init = f_diff / (h ** n)
    ne = max(ne - nr, 1)
    results1 = der_init[:ne], h[:ne], original_shape
    derivative, info = extrapolate(order,richarson_terms,richardson_step,step_ratio,*results1)
    return derivative

def f(x):
    return x**2

print derivative(f,[1,2,3],n=1)
