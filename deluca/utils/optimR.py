import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit
import cvxopt # TODO: speed up via convex approach
from scipy.optimize import minimize

''' How to use:
1) instantiate and pass the first h_0 function during control Alg 3
2) call optim.step() every time you want to get the next output
'''

class optimROBD(object):

    ''' TODOS: 
    1) what is prevH_0? Answer: must create when sending it in
    2) how do we do the projection using scipy?
    '''

    def __init__(self, Cs, p, T, d, prevH, lam=0):
        self._lam = lam
        self._l1 = lam
        self._l2 = 0
        self._Cs = np.array(Cs)
        self._p = p
        self._T = T
        self._prevH = prevH
        self._yhats = np.ndarray((T, d))
        self._d = d


    # does a specific instance of oROBD
    # h_t comes from the control algorithm
    def step(self, v_tminus, h_t, omega_t, t):
        prevH = self._prevH

        # returns a function of y
        prevFunc = self.hittingCost(prevH, v_tminus)

        # building out the yhat sequence
        self._yhats[t-1, :] = self.robdSub(prevFunc, t-1)

        vtilde = self.findSetMin(self.doubleFunc(h_t, t), omega_t)

        fhatFunc = self.hittingCost(h_t, vtilde)
        self._prevH = h_t

        y_t = self.robdSub(fhatFunc, t)

        return y_t

    #TODO: change if necessary
    def doubleFunc(self, h_t, t):
        def func(params):
            y = params[:self._d]
            v = params[self._d:]
            return h_t(y - v) + self._lam * self.cost(t)(y)
        return func
    
    #TODO: change if necessary
    def constraint(self, omega_t):
        def func(params):
            y = params[:self._d]
            v = params[self._d:]
            if tuple(v) in omega_t:
                return 0
            return 1 #return a non-zero element
        return func

    ''' To implement from here '''  

    #TODO: incorporate projection
    def findSetMin(self, function, omega_t):
        x0 = (np.random.randn(d), np.random.randn(d))
        # constraint: must be in the set omega_t
        '''
        cons = ({'type': 'eq', 'fun': self.constraint(omega_t)})

        result = minimize(function, x0, method = 'SLSQP', constraints=cons)
        '''

        result = minimize(function, x0, method = 'COBYLA')
        if result.success:
            fitted_params = np.array(result.x[1]) #want to return v?
            return fitted_params
        else:
            raise ValueError(result.message)
    
    # Find the d x 1 vector y that minimizes the function parameter
    # TODO: change to convex optimizer
    def _findMin(self, func):
        x0 = np.random.rand(self._d)
        res = minimize(func, x0, method='BFGS', options={'disp':False})
        return np.array(res.x)

    ''' End to implement here'''

    # subroutine for ROBD and optimistic ROBD
    def robdSub(self, fun, t):
        vel = self._findMin(fun)

        # below line: find minimum of entire expression with respect to y
        out = self._findMin(self.totalCost(fun, vel, t))
        return out
    
    # precondition: v_t must be a numpy array
    def totalCost(self, fun, v_t, t):
        def func(y):
            return fun(y) + self._l1 * self.cost(t)(y) + self._l2 * self.dist(v_t)(y)
        return func
    
    #precondition: yhats, Cs must be numpy arrays
    def cost(self, t):
        def func (y):
            # each element in decisions is a d x 1 vector, and C is dxd (for each time step)

            decisions = self._yhats[t - self._p : t]
            Cs = self._Cs

            summ = np.zeros(self._d)
            for i in range(self._p):
                C = Cs[i]
                summ += np.matmul(C, decisions[self._p-i-1].T) #TODO: check
            norm = np.linalg.norm(y - summ)
            return (norm**2)/2
        return func

    def dist(self, v_t):
        def func(y):
            norm = np.linalg.norm(y-v_t)
            return (norm**2)/2
        return func

    # must be numpy arrays
    def hittingCost(self, h, v_t):
        def func(y):
            return h(y-v_t)
        return func