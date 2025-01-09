import numpy as np
import matplotlib.pyplot as plt

'''Undone, 1D unsteady Diffusion equation, implicit time'''
def sourceTerm(x, t):
    # return np.sin(x - t)
    return 0

def q_exact(x, t):
    return np.sin(x - t)

def dq_dx_exact(x, t):
    # return np.cos(x - t)
    return 0

def rootMeanSquare(approx,exact):
    return np.sqrt(np.sum((approx-exact)**2)/approx.size)


def fTpsGeneration(m,x,f,fx,fxx):
    n=x.size
    for i in range(n):
        for j in range(n):
            if i!=j:
                r=np.sqrt((x[i]-x[j])**2)
                f[i,j]=r**m*np.log(r)
                fx[i,j]=(x[i]-x[j])*r**(m-2)*(m*np.log(r)+1)
                fxx[i,j]=r**(m-2)*(m*np.log(r)+1)+(x[i]-x[j])**2*r**(m-4)*(2*(m-1)+m*(m-2)*np.log(r))

def fMqGeneration(c,x,f,fx,fxx):
    n=x.size
    for i in range(n):
        for j in range(n):
            f[i,j]=np.sqrt((x[i]-x[j])**2+c**2)
            fx[i,j]=(x[i]-x[j])/f[i,j]
            fxx[i,j]=1/f[i,j]-(x[i]-x[j])**2/f[i,j]**3

'''Inputs'''
dt = .01 #units: s
nodeNo, length, totalTime = 201, 1, 1 #units: -, m, s
beta = 6
timeStep = int(totalTime / dt)
dx = length / (nodeNo-1)
x = np.zeros(nodeNo)
for i in range(nodeNo):
    x[i]=i*dx

f = np.zeros((nodeNo, nodeNo))
fx = np.zeros((nodeNo, nodeNo))
fxx = np.zeros((nodeNo, nodeNo))
sys = np.zeros((nodeNo, nodeNo))
rhs = np.zeros(nodeNo)
C = np.ones(nodeNo)
D = np.ones(nodeNo)
dumQ = np.ones(nodeNo)
alpha = np.zeros(nodeNo)
Q = np.zeros(nodeNo)
Q_old = np.zeros(nodeNo)

fTpsGeneration(beta, x, f, fx, fxx)
# fMqGeneration(.815*dx, x, f, fx, fxx)

for i in range(1, nodeNo - 1):
    sys[i, :] = f[i, :] - dt * (D[i] * fxx[i, :] - C[i] * fx[i, :])
sys[0, :] = f[0, :]
"""right boundary dirichlet or neumann"""
sys[-1, :] = f[-1, :]
# sys[-1, :] = fx[-1, :]

sysInv = np.linalg.inv(sys)

plt.ion()
for tt in range(1, timeStep):
    t = dt * (tt-1)

    '''backward euler'''
    rhs[1:-1] = dt * sourceTerm(x[1:-1], t) + Q_old[1:-1]
    rhs[0] = q_exact(x[0], t)
    """right boundary dirichlet or neumann"""
    rhs[-1] = q_exact(x[-1], t)
    # rhs[-1] = dq_dx_exact(x[-1], t)

    Q = np.matmul(f, np.matmul(sysInv, rhs))
    Q_old[:] = Q[:]
    t = dt * tt
    print(rootMeanSquare(Q, q_exact(x, t)))
    plt.plot(x,Q)
    plt.ylim(0,1)
    plt.pause(.2)
    plt.show()
    plt.cla()