import numpy as np
import matplotlib.pyplot as plt

'''1D unsteady Diffusion equation, TPS'''
def sourceTerm(x, t):
    return np.sin(x-t)

def q_exact(x, t):
    return np.sin(x - t)

def rootMeanSquare(approx,exact):
    return np.sqrt(np.sum((approx-exact)**2)/approx.size)

def dist(i, j, x):
    return np.abs(x[i] - x[j])

def fTpsGeneration(m,x,f,fx,fxx):
    n=x.size
    for i in range(n):
        for j in range(n):
            if i!=j:
                r=np.sqrt((x[i]-x[j])**2)
                f[i,j]=r**m*np.log(r)
                fx[i,j]=(x[i]-x[j])*r**(m-2)*(m*np.log(r)+1)
                fxx[i,j]=r**(m-2)*(m*np.log(r)+1)+(x[i]-x[j])**2*r**(m-4)*(2*(m-1)+m*(m-2)*np.log(r))

'''Inputs'''
dt = .001 #units: s
nodeNo, length, totalTime = 11, 1, 1 #units: -, m, s
beta = 2
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
alpha = np.zeros(nodeNo)
Q = q_exact(x, 0)
# Q_old = q_exact(x, 0)

fTpsGeneration(beta, x, f, fx, fxx)

fInv = np.linalg.pinv(f)

for i in range(1, nodeNo):
    sys[i, :] = (f[i, :] + dt * (D[i] * fxx[i, :] - C[i] * fx[i, :]))
# sys[-1, :] = fx[-1, :]

plt.ion()
for tt in range(1, timeStep):
    t = dt * tt
    alpha = np.matmul(fInv, Q)
    S = sourceTerm(x, t)
    Q[0] = q_exact(x[0], t)
    Q[1:] = np.matmul(sys[1:, :], alpha) + dt * S[1:]
    print(rootMeanSquare(Q,q_exact(x,t)))
    plt.plot(x,Q)
    plt.pause(.2)
    plt.ylim([0,1])
    plt.show()
    plt.cla()