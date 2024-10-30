import numpy as np
import matplotlib.pyplot as plt
import time as tm

def leftBoundary(timeStep, dt):
    QQ = np.zeros(timeStep+1)
    for i in range(timeStep+1):
        if i * dt < 2*60*60:
            QQ[i] = Qin
        elif i * dt >= 2 * 60 * 60 and i * dt < 3 * 60 * 60:
            QQ[i] = Qin + 5 * (i*dt - 2*60*60) / 60 / 60
        elif i * dt >= 3 * 60 * 60 and i * dt < 4 * 60 * 60:
            QQ[i] = Qin + 5 - 5 * (i * dt - 3 * 60 * 60) / 60 / 60
        else:
            QQ[i] = Qin
    return QQ

def lateralFlow(x, t):
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
dt = 18 #units: s
nodeNo, length, totalTime = 51, 20000, 20*60*60   #units: -, m, s
beta = 6
timeStep = int(totalTime / dt)
dx = length / (nodeNo-1)
x = np.zeros(nodeNo)
mannings = np.ones(nodeNo+1) * .012
S_0 = 1/10000
for i in range(nodeNo):
    x[i]=i*dx

global Qin, Hin
Qin = 5
Hin = 4

qsbd = leftBoundary(timeStep, dt)
qsds = [Qin]
qsds1 = [Qin]
qsds2 = [Qin]
qsds3 = [Qin]
Nind = np.arange(nodeNo-2, -1, -1, dtype = int)

f = np.zeros((nodeNo, nodeNo))
fx = np.zeros((nodeNo, nodeNo))
fxx = np.zeros((nodeNo, nodeNo))
sys = np.zeros((nodeNo, nodeNo))
rhs = np.zeros(nodeNo)

C = np.ones(nodeNo)
D = np.ones(nodeNo)
width = 40
wettedPerimeter = 40 + 2 * Hin
csArea = 40 * Hin
R = csArea / wettedPerimeter
S_fForward = mannings[0] ** 2 / (csArea * R ** (2 / 3)) ** 2 * Qin ** 2
S_f = mannings[0] ** 2 / (csArea * R ** (2 / 3)) ** 2 * Qin ** 2
C[:] = 5 * (S_f) ** .3 * np.abs(Qin) ** .4 / 3 / width ** .4 / mannings[0] ** .6
D[:] = (np.abs(Qin) / 2 / (S_f) / width)

Q = np.zeros(nodeNo)
Q_old = np.ones(nodeNo) * Qin
h = np.ones(nodeNo) * Hin

start_time = tm.time()
fTpsGeneration(beta, x, f, fx, fxx)
# fMqGeneration(.815*dx, x, f, fx, fxx)

sys[0, :] = f[0, :]
"""right boundary dirichlet or neumann"""
# sys[-1, :] = f[-1, :]
sys[-1, :] = fx[-1, :]

# plt.ion()
for tt in range(1, timeStep+1):

    for i in range(1, nodeNo - 1):
        sys[i, :] = f[i, :] - dt * (D[i] * fxx[i, :] - C[i] * fx[i, :])
    sysInv = np.linalg.inv(sys)

    t = dt * tt

    '''backward euler'''
    rhs[1:-1] = dt * C[1:-1] * lateralFlow(x[1:-1], t) + Q_old[1:-1]
    rhs[0] = qsbd[tt]
    rhs[-1] = 0  # dQ_dx = 0 et right end
    Q = np.matmul(f, np.matmul(sysInv, rhs))

    width = 40
    wettedPerimeter = 40 + 2 * h[-1]
    csArea = 40 * h[-1]
    R = csArea / wettedPerimeter
    S_fForward = mannings[-1] ** 2 / (csArea * R ** (2 / 3)) ** 2 * Q[-1] ** 2
    for i in Nind:
        wettedPerimeter = 40 + 2 * h[i]
        csArea = 40 * h[i]
        R = csArea / wettedPerimeter
        S_f = mannings[i]**2 / ( csArea * R ** (2 / 3)) ** 2 * Q[i] ** 2
        # celerity2[i] = 5 * (S_f) ** .3 * np.abs(Q[i]) ** .4 / 3 / width ** .4 / mannings[i] ** .6
        # diffusivity2[i] = (np.abs(Q[i]) / 2 / (S_f) / width)
        # C[i] = 5 * (S_f) ** .3 * np.abs(Q[i]) ** .4 / 3 / width ** .4 / mannings[i] ** .6
        # D[i] = (np.abs(Q[i]) / 2 / (S_f) / width)
        C[i] = 5 * (S_f) ** .3 * np.abs(Q[i]) ** .4 / 3 / width ** .4 / mannings[i] ** .6
        D[i] = (np.abs(Q[i]) / 2 / (S_f) / width)
        h[i] = h[i + 1] + (S_0 + (S_f + S_fForward)/2) * dx
        S_fForward = S_f

    # D =  np.ones(nodeNo) * np.min([120, np.mean(diffusivity2[1:-1])])
    # C = np.mean(celerity2[1:-1]) * np.ones(nodeNo)



    Q_old[:] = Q[:]
    # print(Q)
    qsds.append(Q[-1])
    # qsds1.append(Q[5])
    # qsds2.append(Q[10])
    # qsds3.append(Q[15])
    # plt.plot(x,Q)
    # plt.ylim(0,1)
    # plt.pause(.2)
    # plt.show()
    # plt.cla()

end_time = tm.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")

time = np.arange(0, timeStep * dt + dt, dt)
time = time / 3600
plt.plot(time, qsbd, label='upstream (x=0)')
# plt.plot(time, qsds1, 'k',label='(x=5km)', alpha=.25)
# plt.plot(time, qsds2, 'k',label='(x=10km)', alpha=.25)
# plt.plot(time, qsds3, 'k',label='(x=15km)', alpha=.25)
plt.plot(time, qsds, 'k', linestyle='--',label='downstream (x=20km)')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('discharge')
plt.title("diffusive wave model Rbfcm (TPS, beta=6)")
plt.savefig('diffRBF.png')
plt.show()