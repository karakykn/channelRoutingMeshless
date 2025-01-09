import numpy as np
import matplotlib.pyplot as plt

'''Undone, 1D unsteady Diffusion equation, explicit time iterative reach'''
def sourceTerm(x, t):
    # return -2 * np.ones(x.size)
    return 0
    # return -np.cos(x - t) - np.sin(x - t)

def q_exact(x, t):
    # return x**2
    return x
    # return np.sin(x - t)

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
nodeNo, length, totalTime = 11, 1, 1 #units: -, m, s
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
R1 = np.ones(nodeNo)
R2 = np.ones(nodeNo)
R3 = np.ones(nodeNo)
R4 = np.ones(nodeNo)
dumQ = np.ones(nodeNo)
alpha = np.zeros(nodeNo)
# Q = q_exact(x, 0)
Q = np.zeros(nodeNo)
Q_old = np.zeros(nodeNo)

# fTpsGeneration(beta, x, f, fx, fxx)
fMqGeneration(.815*dx, x, f, fx, fxx)

fInv = np.linalg.pinv(f)

Q_old[0] = q_exact(x[0],0)
Q_old[-1] = q_exact(x[-1],0)

plt.ion()
for tt in range(1, timeStep):
    t = dt * tt

    '''rk'''
    R1[1:-1] = D[1:-1] * np.matmul(fxx[1:-1, :], np.matmul(fInv, Q_old)) + sourceTerm(x[1:-1], t)
    dumQ = Q_old + dt * R1
    R2[1:-1] = D[1:-1] * np.matmul(fxx[1:-1, :], np.matmul(fInv, dumQ)) + sourceTerm(x[1:-1], t + dt/2)
    dumQ = Q_old + dt/2 * R2
    R3[1:-1] = D[1:-1] * np.matmul(fxx[1:-1, :], np.matmul(fInv, dumQ)) + sourceTerm(x[1:-1], t + dt/2)
    dumQ = Q_old + dt / 2 * R3
    R4[1:-1] = D[1:-1] * np.matmul(fxx[1:-1, :], np.matmul(fInv, dumQ)) + sourceTerm(x[1:-1], t + dt)
    Q[1:-1] = Q_old[1:-1] + dt / 6 * (R1[1:-1] + 2 * R2[1:-1] + 2 * R3[1:-1] + R4[1:-1])

    # Q[1:-1] = Q_old[1:-1] + dt * (D[1:-1] * np.matmul(fxx[1:-1, :], np.matmul(fInv, Q_old)) + sourceTerm(x[1:-1], t))
    Q[0] = q_exact(x[0], t)
    Q[-1] = q_exact(x[-1], t)
    Q_old[:] = Q[:]

    print(rootMeanSquare(Q, q_exact(x, t)))
    plt.plot(x,Q)
    plt.ylim(0,1)
    plt.pause(.2)
    plt.show()
    plt.cla()