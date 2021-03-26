import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit

class optimROBD(Object):

    def __init__(self, lam=0, Cs, p, T, d):
        self._lam = lam
        self._Cs = Cs
        self._p = p
        self._T = T
        self._prevH = 0 # what is h_0?
        self._yhats = np.ndarray((T, d))
        self._d = d


    # does a specific instance of oROBD
    def step(self, v_tminus, h_t, omega_t, t):
        prevH = self._prevH

        prevFunc = hittingCost(prevH, v_tminus)

        # building out the yhat sequence
        self._yhats[i-1, :] = robdSub(prevFunc, t-1)

        #TODO: understand the double min thing and recover vtilde
        vtilde = findSetMin(someFunc, omega_t)

        fhatFunc = hittingCost(h_t, vtilde)
        self._prevH = h_t

        y_t = robdSub(fhatFunc, t)

        return y_t

    ''' To implement from here '''

    #TODO: implement this double min loss function
    def doubleFunc(h_t, t):
        def func(y, v):
            return
        return
    
    #TODO: implement this double minimizer
    def findSetMin(function, omega_t):
        return
    
    # Find the d x 1 vector y that minimizes the function parameter
    def _findMin(func):
        #TODO: implement gradient descent algorithm here
        return 0 #TODO: change
    
    ''' End to implement here'''

    # subroutine for ROBD and optimistic ROBD
    def robdSub(fun, t):
        vel = findMin(fun)

        # below line: find minimum of entire expression with respect to y
        out = findMin(totalCost(fun, vel, t))
        return out
    
    def totalCost(fun, v_t, t):
        def func(y):
            return fun(y) + self._l1*cost(t) + self._l2 * dist(v_t)
        return func
    
    def cost(t):
        def func (y):
            #TODO: switching cost function
            # each element in decisions is a d x 1 vector, and C is dxd (for each time step)

            decisions = self._yhats[t - self._p : t, :]
            Cs = self._Cs

            summ = np.ndarray((self._d, 1))
            for i in range(self._p):
                C = Cs[i]
                summ += np.matmul(C, decisions[t-i].T) #TODO: check
            norm = np.linalg.norm(y - summ)
            return (norm**2)/2
        return func

    #assuming this is y_t
    def dist(v_t):
        def func(y):
            norm = np.linalg.norm(y-v_t)
            return (norm**2)/2
        return func

    def hittingCost(h, v_t):
        def func(y):
            return h(y-v_t)
        return func