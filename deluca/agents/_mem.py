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
          base_controller,
          A: jnp.ndarray,
          B: jnp.ndarray,
          qs,
          Wt,
          xt):

            self._A, self._B = A, B

            self._base_controller = base_controller

            # Start From Uniform Distribution
            self._T = T

            # Store Model Hyperparameters
            self._eta, self._eps, self._inf = eta, eps, inf

            self._x0 = np.zeros((A.shape[1], 1))

            # State and Action TODO:change
            # self._x, self._u = jnp.zeros((self.n, 1)), jnp.zeros((self.m, 1))

            # self._w = jnp.zeros((HH, self.n, 1))


        def etaMult(self, Cs, etas, t):
            summ = 0
            for i in range(1, p+1):
                summ += np.matmul(Cs[i], etas[t-1-i])
            return summ

        def idx(self):
            self._ks = np.where((self._B).any(axis=1))[0]
            self._d = len(self._ks)

        def defineP(self):
            ps = np.ndarray((self._d, 1))
            ps[0] = self._ks[0]
            for i in range(1,len(ks)):
                ps[i] = ks[i] - ks[i-1]
            self._ps = ps.astype(int)
            self._p = int(np.amax(ps))

        #TODO: fix this to make it work
        #TODO: check indices all over (1-indexed vs 0-indexed)
        def defineCs(self):
            Cs = np.ndarray((self._p, self._d, self._d))

            #TODO: change from 0-indexed to 1-indexed (add dummy)
            for i in range(self._p):
                a_identity = self._A[self._ks, :] #check slicing
                C = np.ndarray((self._d, self._d))
                for j in range(self._d):
                    if i <= self._ps[j]:
                        C[:, j] = a_identity[:, self._ks[j]+1-i]
                    else:
                        C[:, j] = np.zeros(self._d)
                Cs[i] = C
            self._Cs = Cs

        def hitFunc(self, qs, t):
            # define a function of y that can be optimized
            # is y n-dimensional or d-dimensional? 
            # TODO: clarify and change to d-dimensional based on ks
            def func(y):
                masterSum = 0
                for i in range(self._d):
                    littleSum = 0
                    for j in range(1, self._ps[i]):
                        littleSum += qs[t+j]
                    masterSum += (littleSum * (y[self._ks[i]] ** 2))
                return masterSum/2
            return func

        # TODO: change to a continuous type
        def getOmega(self, W, etas, t):
            #TODO: change omega's type to a set? clarify meaning of W
            omega = np.ndarray((len(W), self._d))
            for i, w in enumerate(W):
                littleS = 0
                for i in range(1, self._p+1):
                    littleS += np.matmul(self._Cs[i], etas[t-i])
                omega[i, :] = (-w - littleS)
            omega_t = {tuple(row) for row in omega} # changed this to hashable set
            return omega_t

        def getOuts(self, ys, t):
            lsum = 0
            for i in range(self._p):
                lsum += np.matmul(self._Cs[i], ys[t-i])
            return ys[t] - lsum

        def controlAlgo(self, xs, Ws, qs):
            self.idx(B)
            self.defineP(ks)
            self.defineCs(A, ks, p, ps)

            etas = np.ndarray((self._T, self._d))
            outs = np.ndarray((self._T, self._d))
            us = np.ndarray((self._T, self._d))

            #TODO: instantiate solver
            solver = optimROBD(self._Cs, self._p, self._T, self._d)

            for t in range(self._T):
                if t > 0:
                    subValue = xs[t]-np.matmul(self._A, xs[t-1])-np.matmul(self._B, us[t-1])
                    w_tminus = subvalue[self._ks]

                    etas[t-1] = w_tminus + self.etaMult(self._Cs, etas, t)
                    v_tminus = -1 * etas[t-1]

                func = self.hitFunc(qs, t)
                omega = self.getOmega(Ws[t], etas, t)

                # TODO: what is lambda? to be handled by solver
                outs[t] = solver.step(v_tminus, func, omega, t)

                us[t] = getOuts(outs, t)
            us[T] = 0
            return us

        def policy_loss(controller, A, B, x, w):

            def evolve(x, h):
                """Evolve function"""
                return A @ x + B @ controller.get_action(x) + w[h], None

            final_state, _ = jax.lax.scan(evolve, x, jnp.arange(HH))
            return cost_fn(final_state, controller.get_action(final_state))

        self.policy_loss = policy_loss


      def __call__(self, x, A, B):

        u = self.controlAlgo(A,B,T, xs, Ws, qs) # to implement

        return self.u
