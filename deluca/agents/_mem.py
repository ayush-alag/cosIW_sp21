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

from optimR import optimROBD

def sqRegCost(xs, us, qs, T):
    summ = 0
    for i in range(T):
        q = qs[i]
        x = xs[i]
        u = us[i]
        summ += q/2*(np.linalg.norm(x)**2) + (np.linalg.norm(u)**2)/2
    return summ

def etaMult(Cs, etas, t):
    summ = 0
    for i in range(1, p+1):
        summ += np.matmul(Cs[i], etas[t-1-i])
    return summ

def idx(B):
    return np.where(a.any(axis=1))[0] #return indices of non-zero rows

def defineP(ks):
    ps = np.ndarray((len(ks), 1))
    ps[0] = ks[0]
    for i in range(1,len(ks)):
        ps[i] = ks[i] - ks[i-1]
    return ps, np.amax(ps)

#TODO: define d and other undefined variables throughout
#TODO: check indices all over (1-indexed vs 0-indexed)
def defineCs(A, ks, p, ps):
    d = len(ks)
    Cs = np.ndarray((p, d, d))
    #TODO: change from 0-indexed to 1-indexed (add dummy)
    for i in range(p):
        a_identity = A[ks, :] #check slicing
        C = np.ndarray((d, d))
        for j in range(d):
            if i < ps[j]:
                C[:, j] = a_identity(:, ks[j]+1-i)
            else:
                C[:, j] = np.zeros(d)
        Cs[i] = C
    return Cs

def hitFunc(qs, ps, ks, t):
    # define a function of y that can be optimized
    # is y n-dimensional or d-dimensional? 
    # TODO: clarify and change to d-dimensional based on ks
    def func(y):
        masterSum = 0
        d = len(ks)
        for i in range(d):
            littleSum = 0
            for j in range(1, ps[i]):
                littleSum += qs[t+j]
            masterSum += (littleSum * (y[ks[i]] ** 2))
        return masterSum/2

    return func

def getOmega(W, p, Cs, etas, t, d):
    #TODO: change omega's type to a set? clarify meaning of W
    omega = np.ndarray((len(W), d))
    for i, w in enumerate(W):
        littleS = 0
        for i in range(1, p+1):
            littleS += np.matmul(Cs[i], etas[t-i])
        omega[i, :] = (-w - littleS)
    omega_t = {tuple(row) for row in omega} # changed this to hashable set
    return omega_t

def getOuts(ys, Cs, p, t):
    lsum = 0
    for i in range(p):
        lsum += np.matmul(Cs[i], ys[t-i])
    return ys[t] - lsum

def controlAlgo(A, B, T, xs, Ws, qs):
    ks = idx(B)
    ps, p = defineP(ks)
    Cs = defineCs(A, ks, p, ps)
    d = len(ks)

    etas = np.ndarray((T, d))
    outs = np.ndarray((T, d))
    us = np.ndarray((T, d))

    #TODO: instantiate solver
    solver = optimROBD(Cs, p, T, d)

    for t in range(T):
        if t > 0:
            subValue = xs[t]-np.matmul(A, xs[t-1])-np.matmul(B, us[t-1])
            w_tminus = subvalue[ks]

            etas[t-1] = w_tminus + etaMult(Cs, etas, t)
            v_tminus = -1 * etas[t-1]

        func = hitFunc(qs, ps, ks, t)
        omega = getOmega(Ws[t], p, Cs, etas, t, len(ks))

        # TODO: what is lambda? to be handled by solver
        outs[t] = solver.step(v_tminus, func, omega, t)

        us[t] = getOuts(outs, Cs, p, t)
    us[T] = 0
    return us
        


def quad_loss(x: jnp.ndarray, u: jnp.ndarray) -> Real:
    """
    Quadratic loss.

    Args:
        x (jnp.ndarray):
        u (jnp.ndarray):

    Returns:
        Real
    """
    return jnp.sum(x.T @ x + u.T @ u)


class Mem(Agent):
      def __init__(
          self,
          T: int,
          base_controller,
          A: jnp.ndarray,
          B: jnp.ndarray,
          cost_fn: Callable[[jnp.ndarray, jnp.ndarray], Real] = None,
          HH: int = 10,
          eta: Real = 0.5,
          eps: Real = 1e-6,
          inf: Real = 1e6,
          life_lower_bound: int = 100,
          expert_density: int = 64
      ) -> None:

        self.A, self.B = A, B
        self.n, self.m = B.shape

        cost_fn = cost_fn or quad_loss

        self.base_controller = base_controller

        # Start From Uniform Distribution
        self.T = T
        self.weights = np.zeros(T)
        self.weights[0] = 1.

        # Track current timestep
        self.t, self.expert_density = 0, expert_density

        # Store Model Hyperparameters
        self.eta, self.eps, self.inf = eta, eps, inf

        # State and Action
        self.x, self.u = jnp.zeros((self.n, 1)), jnp.zeros((self.m, 1))

        # Alive set
        self.alive = jnp.zeros((T,))

        # Precompute time of death at initialization
        self.tod = np.arange(T)
        for i in range(1, T):
          self.tod[i] = i + lifetime(i, life_lower_bound)
        self.tod[0] = life_lower_bound # lifetime not defined for 0

        # Maintain Dictionary of Active Learners
        self.learners = {}
        self.learners[0] = base_controller(A, B, cost_fn=cost_fn)

        self.w = jnp.zeros((HH, self.n, 1))

        def policy_loss(controller, A, B, x, w):

            def evolve(x, h):
                """Evolve function"""
                return A @ x + B @ controller.get_action(x) + w[h], None

            final_state, _ = jax.lax.scan(evolve, x, jnp.arange(HH))
            return cost_fn(final_state, controller.get_action(final_state))

        self.policy_loss = policy_loss


      def __call__(self, x, A, B):

        play_i = np.argmax(self.weights)
        self.u = self.learners[play_i].get_action(x)

        # Update alive models
        for i in jnp.nonzero(self.alive)[0]:
            loss_i = self.policy_loss(self.learners[i],A, B, x, self.w)
            self.weights[i] *= np.exp(-self.eta * loss_i)
            self.weights[i] = min(max(self.weights[i], self.eps), self.inf)
            self.learners[i].update(x, u=self.u)

        self.t += 1

        # One is born every expert_density steps
        if(self.t%self.expert_density==0):
            self.alive = jax.ops.index_update(self.alive, self.t, 1)
            self.weights[self.t] = self.eps
            self.learners[self.t] = self.base_controller(A, B, cost_fn=self.cost_fn)
            self.learners[self.t].x = x

        # At most one dies
        kill_list = jnp.where(self.tod == self.t)
        if len(kill_list[0]):
            kill = int(kill_list[0][0])
            if(self.alive[kill]):
                self.alive = jax.ops.index_update(self.alive, kill, 0)
                del self.learners[kill]
                self.weights[kill] = 0

        # Rescale
        max_w = np.max(self.weights)
        if(max_w<1):
            self.weights /= max_w

        # Get new noise (will be located at w[-1])
        self.w = jax.ops.index_update(self.w, 0, x - self.A @ self.x + self.B @ self.u)
        self.w = jnp.roll(self.w, -1, axis = 0)

        # Update System
        self.x, self.A, self.B = x, A, B

        return self.u
