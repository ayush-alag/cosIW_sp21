import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit
import cvxopt # TODO: speed up via convex approach
from scipy.optimize import minimize

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

        # returns a function of y
        prevFunc = hittingCost(prevH, v_tminus)

        # building out the yhat sequence
        self._yhats[i-1, :] = robdSub(prevFunc, t-1)

        #TODO: understand the double min thing and recover vtilde
        vtilde = findSetMin(doubleFunc(h_t, t), omega_t)

        fhatFunc = hittingCost(h_t, vtilde)
        self._prevH = h_t

        y_t = robdSub(fhatFunc, t)

        return y_t

    def doubleFunc(h_t, t):
        def func(params):
            y, v = params
            return h_t(y - v) + self._lam * cost(t)(y)
        return func
    
    def constraint(omega_t):
        def func(params):
            y, v = params
            if v in omega_t:
                return 0
            return .1 #return a non-zero element
        return func

    ''' To implement from here '''  
    
    #TODO: need to really check/test this
    def findSetMin(function, omega_t):
        x0 = np.random.rand(self._d, 2)
        # constraint: must be in the set omega_t
        cons = ({'type': 'eq', 'fun': constraint(omega_t)})

        result = minimize(function, x0, method = 'SLSQP', constraints=cons)
        if result.success:
            fitted_params = result.x[:, 1] #want to return v?
            return fitted_params
        else:
            raise ValueError(result.message)
    
    # Find the d x 1 vector y that minimizes the function parameter
    # TODO: change to convex optimizer
    def _findMin(func):
        x0 = np.random.rand(self._d)
        res = minimize(func, x0, method='BFGS', options={'disp':True})
        return res.x

    ''' End to implement here'''

    # subroutine for ROBD and optimistic ROBD
    def robdSub(fun, t):
        vel = findMin(fun)

        # below line: find minimum of entire expression with respect to y
        out = findMin(totalCost(fun, vel, t))
        return out
    
    def totalCost(fun, v_t, t):
        def func(y):
            return fun(y) + self._l1*cost(t)(y) + self._l2 * dist(v_t)(y)
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
                summ += np.matmul(C, decisions[t-i, :].T) #TODO: check
            norm = np.linalg.norm(y - summ)
            return (norm**2)/2
        return func

    def dist(v_t):
        def func(y):
            norm = np.linalg.norm(y-v_t)
            return (norm**2)/2
        return func

    def hittingCost(h, v_t):
        def func(y):
            return h(y-v_t)
        return func