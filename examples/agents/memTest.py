import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
print(sys.path)
sys.path.append('/Users/ayushalag/Documents/cosIW/cosIW_sp21/')
os.chdir('/Users/ayushalag/Documents/cosIW/cosIW_sp21/')
print(os.getcwd())

from deluca.agents import GPC, Adaptive, Mem
from deluca.envs import LDS
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

cummean = lambda x: np.cumsum(x)/(np.ones(T) + np.arange(T))
n, m = 2, 1

def get_errs(T, controller, A, B, noise):
    state = jnp.zeros((n, 1))
    errs = np.zeros((T, 1))
    actions = np.zeros((T,1))
    
    for i in tqdm(range(1, T)):
        try:
            action = controller(state, A, B)
        except:
            action = controller(state)
        actions[i] = action
            
        if(noise=="normal"):
            state = A @ state + B @ action #+ np.random.normal(0, 0.2, size=(n,1)) # gaussian noise
        else:
            state = A @ state + B @ action #+ np.random.normal(0, 0.2, size=(n,1)) *(i%300 < 150) + 0.4 * jnp.sin(i) *(i%300>= 150) # add sine noise every 150 steps
        
        if(i % T//2 == 0): # switch system
            A,B = jnp.array([[1.,1.5], [0,1.]]), jnp.array([[0],[0.9]])
        
        errs[i] = (jnp.linalg.norm(state)+jnp.linalg.norm(action))
    
    return errs, actions

T = 30

A,B = jnp.array([[1.,.5], [0,1.]]), jnp.array([[0],[1.2]])

ada = Adaptive(T, base_controller=GPC, A=A, B=B)
gpc = GPC(A, B)
mem = Mem(T, A, B)

mem_errs, m_acts = get_errs(T, mem, A, B, "normal")
print("Memory incurs ", np.mean(mem_errs), " loss under gaussian iid noise")

gpc_errs, g_acts = get_errs(T, gpc, A, B, "normal")
print("GPC incurs ", np.mean(gpc_errs), " loss under gaussian iid noise")

ada_errs, a_acts = get_errs(T, ada, A, B, "normal")
print("AdaGPC incurs ", np.mean(ada_errs), " loss under gaussian iid noise")

plt.title("Instantenous losses under gaussian iid noise")
plt.plot(cummean(gpc_errs), "green", label = "GPC")
plt.plot(cummean(ada_errs), "blue", label = "AdaGPC")
plt.plot(cummean(mem_errs), "red", label = "MemCR")
plt.legend()
plt.savefig('gaussianLoss.png')
plt.clf()

plt.title("Instantaneous actions under gaussian noise")
plt.plot(g_acts, "green", label = "GPC")
plt.plot(a_acts, "blue", label = "AdaGPC")
plt.plot(m_acts, "red", label = "MemCR")
plt.legend()
plt.savefig('gaussianAction.png')
plt.clf()

plt.title("Instantaneous actions under gaussian noise (no mem)")
plt.plot(g_acts, "green", label = "GPC")
plt.plot(a_acts, "blue", label = "AdaGPC")
plt.legend()
plt.savefig('gaussianActionRaw.png')
plt.clf()

plt.title("Instantaneous losses under gaussian noise")
plt.plot(gpc_errs, "green", label = "GPC")
plt.plot(ada_errs, "blue", label = "AdaGPC")
plt.plot(mem_errs, "red", label = "Mem")
plt.legend()
plt.savefig('gaussianErr.png')
plt.clf()


# sine noise
A,B = jnp.array([[1.,.5], [0,1.]]), jnp.array([[0],[1.2]])

ada = Adaptive(T, base_controller=GPC, A=A, B=B)
gpc = GPC(A, B)
mem = Mem(T, A, B)

gpc_errs, g_acts = get_errs(T, gpc, A, B, "sine")
print("GPC incurs ", np.mean(gpc_errs), " loss under intermittent sine noise")

ada_errs, a_acts = get_errs(T, ada, A, B, "sine")
print("AdaGPC incurs ", np.mean(ada_errs), " loss under intermittent sine noise")

mem_errs, m_acts = get_errs(T, mem, A, B, "sine")
print("Memory incurs ", np.mean(mem_errs), " loss under intermittent sine noise")

plt.title("Instantanous losses under intermittent sine noise")
plt.plot(cummean(gpc_errs), "green", label = "GPC")
plt.plot(cummean(ada_errs), "blue", label = "AdaGPC")
plt.plot(cummean(mem_errs), "red", label = "MemCR")
plt.legend()
plt.savefig('sineLoss.png')