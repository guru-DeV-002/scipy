from __future__ import division,absolute_import
import numpy as np
from numpy.testing import (assert_raises, assert_allclose, assert_equal,assert_, run_module_suite)
from _derivative import Derivative, Gradient, Jacobian

def fun(x):
    return x**3 + x**2

def fun2(x,y):
    return x**3 + y**2

def fun3(x,y):
    return [x**3 + y**2, x*y**3]

class Test(object):
    def test_derivative(self):
        for method in ['central','forward']:
            # test for singular point
            df = Derivative(fun,[2],method=method)
            print df
            assert_(np.allclose(df,16))

            #test for multiple points 
            df = Derivative(fun,[2,3,4],method=method)
            assert_(np.allclose(df,[16,33,56]))

    def test_gradient(self):
        for method in ['central','forward']:
            # test for univariate function
            df = Gradient(fun,[[2],[3],[4]],method=method)
            assert_(np.allclose(df,[[16],[33],[56]]))

            #test for multivariate function
            df = Gradient(fun2,[[2,3],[3,2],[1,1]],method=method)
            assert_(np.allclose(df,[[12,6],[27,4],[3,2]]))

    def test_jacobian(self):
        for method in ['central','forward']:
            # test for univariate function
            df = Jacobian(fun,[[2],[3],[4]],method=method)
            assert_(np.allclose(df,[[[16]],[[33]],[[56]]]))

            # test for multivariate function returning a scalar
            df = Jacobian(fun2,[[2,3],[3,2],[1,1]],method=method)
            assert_(np.allclose(df,[[[12,6]],[[27,4]],[[3,2]]]))

            # test for multivariate function returning a vector
            df = Jacobian(fun3,[[2,3],[3,2],[1,1]],method=method)
            assert_(np.allclose(df,[[[12,6],[27,54]],[[27,4],[8,36]],[[3,2],[1,3]]]))

if __name__ == '__main__':
    run_module_suite()
