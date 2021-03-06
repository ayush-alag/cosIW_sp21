# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""deluca.agents._mem"""
from numbers import Real
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax import jit

from deluca.agents._lqr import LQR
from deluca.agents.core import Agent

from deluca.utils.optimR import optimROBD
'''
Not Used

def sqRegCost(xs, us, qs, T):
    summ = 0
    for i in range(T):
        q = qs[i]
        x = xs[i]
        u = us[i]
        summ += q/2*(np.linalg.norm(x)**2) + (np.linalg.norm(u)**2)/2
    return summ
'''

# TODO: convert to JNP?
class Mem(Agent):
    def __init__(
      self,
      T: int,
      A,
      B): # scalar for the control cost

        self._A, self._B = np.array(A), np.array(B)
        self._n = A.shape[0]
        self._d = B.shape[1]
        assert self._n == B.shape[0]

        # Start From Uniform Distribution
        self._T = T

        # keep track of the current timestep
        self._t = 0
        assert self._t <= self._T

        # 0 to T-1: first row is already 0 (initial start) for xs, not for us
        self._xs = np.zeros((self._T, self._n))
        self._us = np.zeros((self._T+1, self._d))

        # used in the computation, set to 0's
        self._etas = np.zeros((self._T, self._d))
        self._ys = np.zeros((self._T, self._d))
        

        ''' starting the processes '''
        self.idx()        # creates ks,d
        self.defineP()    # creates ps, p
        self.defineCs()   # creates Cs
        
        print(self._Cs)
        print(self._ps)
        # instantiate the solver
        self._solver = optimROBD(self._Cs, self._p, self._T, self._d, self._n, self._ps)

    def idx(self):
        self._ks = np.where((self._B).any(axis=1))[0]
        assert self._d == len(self._ks)

    def defineP(self):
        ps = np.ndarray((self._d, ))
        ps[0] = self._ks[0]
        for i in range(1,len(self._ks)):
            ps[i] = self._ks[i] - self._ks[i-1]
        self._ps = ps.astype(int)
        self._p = int(np.amax(ps))


    def defineCs(self):
        Cs = np.ndarray((self._p, self._d, self._d))

        a_identity = self._A[self._ks, :] #check slicing
        for i in range(1, self._p+1):
            C = np.ndarray((self._d, self._d))
            for j in range(1, self._d+1):
                if i-1 <= self._ps[j-1]:
                    C[:, j-1] = a_identity[:, self._ks[j-1]+1-i]
                else:
                    C[:, j-1] = np.zeros(self._d)
            Cs[i-1] = C
        self._Cs = Cs

    def controlAlgo(self, w_t, radius_t, qs, lam):
        func = self.hitFunc(qs) #TODO: not really necessary as we're assuming it's a certain type in solver

        ''' changed here to single call '''
        
        assert self._t < self._T
        if self._t > 0:
            print(self._xs[self._t])
            print(self._xs[self._t-1])
            print(self._us[self._t - 1])
            subValue = self._xs[self._t]-np.matmul(self._A, self._xs[self._t - 1])-np.matmul(self._B, self._us[self._t-1])
            w_tminus = subValue[self._ks]
            print("wt: " + str(w_tminus))

            self._etas[self._t-1] = w_tminus + self.etaMult()
            v_tminus = -1 * self._etas[self._t-1]
            print("v_tminus: " + str(v_tminus)) 

            omega = -w_t - self.getOmega()
            print("omega: " + str(omega))

            self._ys[self._t] = self._solver.step(v_tminus, func, omega, radius_t, self._t, lam)

        self._us[self._t] = self.getOuts()
        print("u_t: " + str(self._us[self._t]))

        self._t += 1
        return self._us[self._t-1]

    def hitFunc(self, qs):
        def func(y):
            masterSum = 0
            for i in range(self._d):
                littleSum = 0
                for j in range(int(self._ps[i])):
                    littleSum += qs[self._t+j]
                masterSum += (littleSum * (y[i] ** 2))
            return masterSum/2
        return func

    ''' Todo (stretch): combine these methods into one super method to avoid duplicate errors '''
    
    # for multiplying the zetas
    def etaMult(self):
        summ = 0
        for idx, C in enumerate(self._Cs):
            if self._t-2-idx >= 0: 
                summ += np.matmul(C, self._etas[self._t-2-idx])
        return summ

    def getOmega(self):
        summ = 0
        for idx, C in enumerate(self._Cs):
            if self._t-1-idx >= 0: 
                summ += np.matmul(C, self._etas[self._t-1-idx])
        return summ

    def getOuts(self):
        lsum = 0
        for idx, C in enumerate(self._Cs):
            if self._t-1-idx >= 0: 
                lsum += np.matmul(C, self._ys[self._t-1-idx])
        return self._ys[self._t] - lsum
    
    # takes in state, disturbance, disturbance radius, qs, lambda
    def __call__(self, state, w = 0, radius_t=1, qs = False, q_t = 0, lam=0):
        if qs == False:
            q_t = np.ones((self._p+self._T))
        self._xs[self._t] = state[0] # add it to the list
        return self.controlAlgo(w, radius_t, q_t, lam)