import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit
#import cvxopt # TODO: speed up via convex approach
from scipy.optimize import minimize
import random

''' How to use:
1) instantiate and pass the first h_0 function during control Alg 3
2) call optim.step() every time you want to get the next output
'''

class optimROBD(object):

    def __init__(self, Cs, p, T, d, n, ps):
        self._Cs = np.array(Cs)
        lsum = 0
        for C in Cs:
            lsum += np.linalg.norm(C)
        print("alpha: " + str(lsum))
        self._p = p
        self._T = T
        self._yhats = np.zeros((T, d))
        self._d = d
        self._n = n
        self._ps = ps

    # h_t comes from the control algorithm
    def step(self, v_tminus, h_t, omega_t, radius_t, t, lam):
        '''
        for the more general case of an arbitrary function. not needed because we know the function 
        
#         prevH = self._prevH

#         # returns a function of y
#         prevFunc = self.hittingCost(prevH, v_tminus)

        # building out the yhat sequence
        #self._yhats[t-1, :] = self.robdSub(prevFunc, t-1, v_tminus)
        
        '''
        
        self._lam = lam
        self._yhats[t-1, :] = self.robdSub(t-1, v_tminus)
        print("yhat: " + str(self._yhats[t-1, :]))

#         print("double min")
        vtilde = self.findSetMin(self.doubleFunc(h_t, t), omega_t, radius_t, v_tminus)
#         print(vtilde)
        print("vtilde: " + str(vtilde))

#         fhatFunc = self.hittingCost(h_t, vtilde)
#         self._prevH = h_t

        y_t = self.robdSub(t, vtilde)
        print("y_t: " + str(y_t))
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
    def findSetMin(self, function, omega_t, radius_t, v_tminus):
        if self._lam ==. 0:
            return omega_t
        else:
#         return omega_t
            x0 = (v_tminus+np.random.randn(self._d), v_tminus+np.random.randn(self._d))

            result = minimize(function, x0, method='Nelder-Mead')

            if result.success:
                fitted_params = np.array(result.x[self._d:]) #want to return v?
                print("fitted: " + str(fitted_params))
                print("omega: " + str(omega_t))
                diff = fitted_params - omega_t
                norm = np.linalg.norm(diff)

                if norm != 0:
                    q = (radius_t/norm)*diff
                else:
                    q = 0

                final = q + omega_t
                return -1*final # projected answer
                #return omega_t
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
    def robdSub(self, t, v_tminus):
        vel = v_tminus
        print(self._ps)
        # print("scipy val: " + str(self._findMin(fun)))
        # print("numerical val: " + str(vel))

        # below line: find minimum of entire expression with respect to y
        # print("scipy val 2: " + str(self._findMin(self.totalCost(fun, vel, t))))

        if self._lam == 0:
            val = vel
        else:
            lsum = 0
            for i in range (1, self._p+1):
                lsum += self._Cs[i-1] @ self._yhats[t - i]
            
            val = (self._lam * lsum + np.multiply(self._ps, vel)) / (self._lam + self._ps)

        # print("numerical val 2: " + str(val))
        return val
    
    #precondition: yhats, Cs must be numpy arrays
    def cost(self, t):
        def func (y):
            summ = np.zeros(self._d)
            for idx, C in enumerate(self._Cs):
                if t - idx > 0:
                    summ += np.matmul(C, self._yhats[t-idx-1])
            norm = np.linalg.norm(y - summ)
            return (norm**2)/2
        return func

    ''' 
    not needed since lambda_2 = 0
    def dist(self, v_t):
        def func(y):
            norm = np.linalg.norm(y-v_t)
            return (norm**2)/2
        return func
    '''

    # must be numpy arrays
    def hittingCost(self, h, v_t):
        def func(y):
            return h(y-v_t)
        return func
