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

np.seterr('raise')

def get_errs(T, controller, A, B, noise):
    states = np.zeros((T, n))
    errs = np.zeros((T, 1))
    actions = np.zeros((T,m))
    
    for i in tqdm(range(1, T)):
        print("action: ")
        print(actions[i-1])
        print("state: ")
        print(states[i-1])

        if (noise == "normal"):
            w_t = np.random.normal(0, 0.2, size=(m,1))[0]
        else:
            w_t = np.random.normal(0, 0.2, size=(m,1)) *(i%300 < 150) + 0.4 * jnp.sin(i) *(i%300>= 150)

        actions[i] = controller(states[i-1], w = w_t, lam = 0)
            
        states[i] = A @ states[i-1] + B @ (actions[i-1] + np.array(w_t)) # gaussian noise
        
        if(i % T//2 == 0): # switch system
            A,B = np.array([[1.,1.5], [0,1.]]), np.array([[0],[0.9]])
        
        errs[i] = (np.linalg.norm(states[i])+np.linalg.norm(actions[i]))
    
    return states, errs, actions

T = 20

A,B = np.array([[1.,.5], [0,1.]]), np.array([[0],[1.2]])

# A = np.array([[0, 1, 0, 0, 0], [1, 2, 3, 4, 5], [0, 0, 0, 1, 0], [1,1, 1, 1, 1], [0,1,0,1,0]])
# B = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])

ada = Adaptive(T, base_controller=GPC, A=A, B=B)
gpc = GPC(A, B)
mem = Mem(T, A, B)

mem_state, mem_errs, m_acts = get_errs(T, mem, A, B, "normal")
print("Memory incurs ", np.mean(mem_errs), " loss under gaussian iid noise")

# gpc_state, gpc_errs, g_acts = get_errs(T, gpc, A, B, "normal")
# print("GPC incurs ", np.mean(gpc_errs), " loss under gaussian iid noise")

# ada_errs, a_acts = get_errs(T, ada, A, B, "normal")
# print("AdaGPC incurs ", np.mean(ada_errs), " loss under gaussian iid noise")

plt.title("Cumulative mean losses under gaussian iid noise")
# plt.plot(cummean(gpc_errs), "green", label = "GPC")
#plt.plot(cummean(ada_errs), "blue", label = "AdaGPC")
plt.plot(cummean(mem_errs), "red", label = "MemCR")
plt.legend()
plt.savefig('gaussianLoss.png')
plt.clf()

plt.title("Instantaneous actions under gaussian noise")
# plt.plot(g_acts, "green", label = "GPC")
#plt.plot(a_acts, "blue", label = "AdaGPC")
plt.plot(m_acts, "red", label = "MemCR")
plt.legend()
plt.savefig('gaussianAction.png')
plt.clf()

plt.title("Instantaneous states under gaussian noise")
# plt.plot(gpc_state, "green", label = "GPC")
#plt.plot(ada_errs, "blue", label = "AdaGPC")
plt.plot(mem_state, "red", label = "Mem")
plt.legend()
plt.savefig('gaussianState.png')
plt.clf()

''' not doing beyond this for now

# sine noise
A,B = jnp.array([[1.,.5], [0,1.]]), jnp.array([[0],[1.2]])

ada = Adaptive(T, base_controller=GPC, A=A, B=B)
gpc = GPC(A, B)
mem = Mem(T, A, B)

gpc_errs, g_acts = get_errs(T, gpc, A, B, "sine")
print("GPC incurs ", np.mean(gpc_errs), " loss under intermittent sine noise")

# ada_errs, a_acts = get_errs(T, ada, A, B, "sine")
# print("AdaGPC incurs ", np.mean(ada_errs), " loss under intermittent sine noise")

mem_errs, m_acts = get_errs(T, mem, A, B, "sine")
print("Memory incurs ", np.mean(mem_errs), " loss under intermittent sine noise")

plt.title("Instantanous losses under intermittent sine noise")
plt.plot(cummean(gpc_errs), "green", label = "GPC")
#plt.plot(cummean(ada_errs), "blue", label = "AdaGPC")
plt.plot(cummean(mem_errs), "red", label = "MemCR")
plt.legend()
plt.savefig('sineLoss.png')

'''