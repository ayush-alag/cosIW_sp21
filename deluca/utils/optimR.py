import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit
#import cvxopt # TODO: speed up via convex approach
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

    def __init__(self, Cs, p, T, d, prevH, n, ps, lam=0):
        self._lam = lam
        self._l1 = lam
        self._l2 = 0
        self._Cs = np.array(Cs)
        lsum = 0
        for C in Cs:
            lsum += np.linalg.norm(C)
        print("alpha: " + str(lsum))
        self._p = p
        self._T = T
        self._prevH = prevH
        self._yhats = np.zeros((T, d))
        self._d = d
        self._n = n
        self._ps = ps

    # does a specific instance of oROBD
    # h_t comes from the control algorithm
    def step(self, v_tminus, h_t, omega_t, radius_t, t):
        prevH = self._prevH

        # returns a function of y
        prevFunc = self.hittingCost(prevH, v_tminus)

        # building out the yhat sequence
        self._yhats[t-1, :] = self.robdSub(prevFunc, t-1, v_tminus)

        print("double min")
        vtilde = self.findSetMin(self.doubleFunc(h_t, t), omega_t, radius_t)
        # vtilde = omega_t
        print(vtilde)

        fhatFunc = self.hittingCost(h_t, vtilde)
        self._prevH = h_t

        y_t = self.robdSub(fhatFunc, t, vtilde)

        return y_t

    #TODO: change if necessary
    def doubleFunc(self, h_t, t):
        def func(params):
            y = params[:self._d]
            v = params[self._d:]
            val = h_t(y - v) + self._lam * self.cost(t)(y)
            return val
        return func

    #TODO: incorporate projection
    def findSetMin(self, function, omega_t, radius_t):
        x0 = (np.random.randn(self._d), np.random.randn(self._d))
        # constraint: must be in the set omega_t

        result = minimize(function, x0, method='Nelder-Mead')
        
        if result.success:
            fitted_params = np.array(result.x[1]) #want to return v?
            diff = fitted_params - omega_t
            norm = np.linalg.norm(diff)
            q = (radius_t/norm)*diff
            final = q + omega_t
            return final
        else:
            raise ValueError(result.message)
    
    '''
    no longer necessary

    # Find the d x 1 vector y that minimizes the function parameter
    # TODO: change to convex optimizer
    def _findMin(self, func):
        x0 = np.random.rand(self._d)
        res = minimize(func, x0, method='BFGS', options={'disp':False})
        return np.array(res.x)
        
    # precondition: v_t must be a numpy array
    def totalCost(self, fun, v_t, t):
        def func(y):
            return fun(y) + self._l1 * self.cost(t)(y) + self._l2 * self.dist(v_t)(y)
        return func

    '''

    # subroutine for ROBD and optimistic ROBD
    def robdSub(self, fun, t, v_tminus):
        vel = v_tminus
        # print("scipy val: " + str(self._findMin(fun)))
        # print("numerical val: " + str(vel))

        # below line: find minimum of entire expression with respect to y
        # print("scipy val 2: " + str(self._findMin(self.totalCost(fun, vel, t))))

        if self._l1 == 0:
            val = vel
        else:
            lsum = 0
            for i in range (1, self._p+1):
                addVal = self._Cs[i-1] @ self._yhats[t - i]
                lsum += addVal
            
            val = (self._l1 * lsum + np.multiply(self._ps, vel)) / (self._l1 + self._ps)

        # print("numerical val 2: " + str(val))
        return val
    
    #precondition: yhats, Cs must be numpy arrays
    def cost(self, t):
        def func (y):
            Cs = self._Cs

            summ = np.zeros(self._d)
            for idx, C in enumerate(self._Cs):
                if t - idx > 0:
                    summ += np.matmul(C, self._yhats[t-idx-1].T)
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
