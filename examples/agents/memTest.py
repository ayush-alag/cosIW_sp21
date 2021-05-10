import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
print(sys.path)
sys.path.append('/Users/ayushalag/Documents/cosIW/cosIW_sp21/')
os.chdir('/Users/ayushalag/Documents/cosIW/cosIW_sp21/')
print(os.getcwd())

from deluca.agents import GPC, Adaptive, Mem, PID, LQR
from deluca.envs import LDS
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

cummean = lambda x: np.cumsum(x)/(np.ones(T) + np.arange(T))
n, m = 5,3

np.seterr('raise')

def run_loop(T, controller, A, B, noise):
    lambdas = [0]
    losses = []
    for lam in lambdas:
        _, errs, _ = get_errs(T, controller, A, B, noise, lam)
        losses.append(np.mean(errs))
    return lambdas, losses
        

def get_errs(T, controller, A, B, noise): 
    states = np.zeros((T, n))
    errs = np.zeros((T, 1))
    actions = np.zeros((T,m))
    dists = np.zeros((T, m))

    for i in tqdm(range(1, T)):
        print("\ni: " + str(i) + "\n")
#         print("action: ")
#         print(actions[i-1])
#         print("state: ")
#         print(states[i-1])

        if (noise == "normal"):
            w_t = np.random.normal(0, 0.2, size=(m,1))
        elif (noise == "none"):
            w_t = np.zeros((m,1))
        else:
            w_t = np.random.normal(0, 0.2, size=(m,1)) *(i%50 < 25) + 0.4 * np.sin(i) *(i%50>= 25)
        
        dists[i] = w_t[0]

        try:
            actions[i] = controller(states[i-1], A, B, w=dists[i-1], lam=0)
        except:
            print("prevState: " + str(states[i-1]))
            print("noise: " + str(dists[i-1]))
            actions[i] = controller(states[i-1], w = dists[i-1], lam = 0)
            print("action: " + str(actions[i]))

        a_part = A@states[i-1]
        b_part = np.ndarray.flatten(B @ (actions[i] + dists[i-1]))
        states[i] += a_part
        states[i] += b_part
        print("state: " + str(states[i]))

#         if(i % T//2 == 0): # switch system
#             A,B = np.array([[1.,1.5], [0,1.]]), np.array([[0],[0.9]])

        errs[i] = (np.linalg.norm(states[i])+np.linalg.norm(actions[i]))

    return states, errs, actions

T = 200

#A, B = np.array([[2]]), np.array([[1]])
#A, B = np.array([[0, 1], [0, 1.2]]), np.array([[0], [1]])

#A,B = np.array([[1.,.5], [0,1.]]), np.array([[0],[1.2]])

A = np.array([[0, 1, 0, 0, 0], [1, 2, 3, 4, 5], [0, 0, 0, 1, 0], [1,1, 1, 1, 1], [0,1,0,1,0]])
B = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])

lqr = LQR(A, B)
# ada = Adaptive(T, base_controller=GPC, A=A, B=B)
gpc = GPC(A, B)
mem = Mem(T, A, B)

mem_state, mem_errs, m_acts = get_errs(T, mem, A, B, "none")
print("Memory incurs ", np.mean(mem_errs), " loss under no noise")

pid_state, pid_errors, pid_acts = get_errs(T, lqr, A, B, "none")
print("LQR incurs ", np.mean(pid_errors), " loss under no noise")

lqr = LQR(A, B)
# ada = Adaptive(T, base_controller=GPC, A=A, B=B)
gpc = GPC(A, B)
mem = Mem(T, A, B)

mem_state, mem_errs, m_acts = get_errs(T, mem, A, B, "normal")
print("Memory incurs ", np.mean(mem_errs), " loss under gaussian iid noise")

pid_state, pid_errors, pid_acts = get_errs(T, lqr, A, B, "normal")
print("LQR incurs ", np.mean(pid_errors), " loss under gaussian iid noise")

# gpc_state, gpc_errs, g_acts = get_errs(T, gpc, A, B, "normal")
# print("GPC incurs ", np.mean(gpc_errs), " loss under gaussian iid noise")

# ada_errs, a_acts = get_errs(T, ada, A, B, "normal")
# print("AdaGPC incurs ", np.mean(ada_errs), " loss under gaussian iid noise")

plt.title("Cumulative mean losses under gaussian iid noise")
# plt.plot(cummean(gpc_errs), "green", label = "GPC")
# plt.plot(cummean(ada_errs), "blue", label = "AdaGPC")
plt.plot(cummean(pid_errors), "blue", label = "LQR")
plt.plot(cummean(mem_errs), "red", label = "MemCR")
plt.legend()
plt.savefig('gaussianLoss.png')
plt.clf()

plt.title("Instantaneous actions under gaussian noise")
# plt.plot(g_acts, "green", label = "GPC")
# plt.plot(a_acts, "blue", label = "AdaGPC")
plt.plot(pid_acts, "blue", label = "LQR")
plt.plot(m_acts, "red", label = "MemCR")
plt.legend()
plt.savefig('gaussianAction.png')
plt.clf()

plt.title("Instantaneous states under gaussian noise")
# plt.plot(gpc_state, "green", label = "GPC")
# plt.plot(ada_errs, "blue", label = "AdaGPC")
plt.plot(pid_state[:, 0], "blue", label = "LQR")
plt.plot(mem_state[:, 0], "red", label = "Mem")
plt.legend()
plt.savefig('gaussianState.png')
plt.clf()


# # sine noise
# A,B = jnp.array([[1.,.5], [0,1.]]), jnp.array([[0],[1.2]])

# ada = Adaptive(T, base_controller=GPC, A=A, B=B)
# gpc = GPC(A, B)
lqr = LQR(A, B)
mem = Mem(T, A, B)

# gpc_errs, g_acts = get_errs(T, gpc, A, B, "sine")
# print("GPC incurs ", np.mean(gpc_errs), " loss under intermittent sine noise")

pid_state, pid_errors, pid_acts = get_errs(T, lqr, A, B, "sine")
print("LQR incurs ", np.mean(pid_errors), " loss under gaussian iid noise")

# ada_errs, a_acts = get_errs(T, ada, A, B, "sine")
# print("AdaGPC incurs ", np.mean(ada_errs), " loss under intermittent sine noise")

mem_state, mem_errs, m_acts = get_errs(T, mem, A, B, "sine")
print("Memory incurs ", np.mean(mem_errs), " loss under intermittent sine noise")


plt.title("Cumulative mean losses under intermittent sine noise")
# plt.plot(cummean(gpc_errs), "green", label = "GPC")
# plt.plot(cummean(ada_errs), "blue", label = "AdaGPC")
plt.plot(cummean(pid_errors), "blue", label = "LQR")
plt.plot(cummean(mem_errs), "red", label = "MemCR")
plt.legend()
plt.savefig('sineLoss.png')
plt.clf()

plt.title("Instantaneous actions under intermittent sine noise")
# plt.plot(g_acts, "green", label = "GPC")
# plt.plot(a_acts, "blue", label = "AdaGPC")
plt.plot(pid_acts, "blue", label = "LQR")
plt.plot(m_acts, "red", label = "MemCR")
plt.legend()
plt.savefig('sineAction.png')
plt.clf()

plt.title("Instantaneous states under intermittent sine noise")
# plt.plot(gpc_state, "green", label = "GPC")
# plt.plot(ada_errs, "blue", label = "AdaGPC")
plt.plot(pid_state[:, 0], "blue", label = "LQR")
plt.plot(mem_state[:, 0], "red", label = "Mem")
plt.legend()
plt.savefig('sineState.png')
plt.clf()
