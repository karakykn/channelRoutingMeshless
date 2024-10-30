import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Solution parameters"""
theta = .5
dt = 18 # sec
timeStep = 5000

"""Geometry and related"""
np.random.seed(3)
L = 20000 # length of the channel
N = 20 # no of intervals not nodes
# dx = np.random.rand(N)
dx = np.ones(N)* L / (N)
alpha = np.zeros(N+1)
x = np.zeros(N+1)
mannings = np.ones(N+1)*0.012
S_0 = 1/10000
z = np.zeros(N+1)
hLast = 4
z[0] = 2
for i in range(1,N+1):
    x[i] = x[i-1] + dx[i-1]
    z[i] = z[i-1] - S_0 * dx[i-1]
alpha[1:-1] = dx[1:] / dx[:-1]
alpha[-1] = 1

"""Variables"""
qp = np.ones(N+1)*20
qOld = np.ones(N+1)*20
q_x = np.zeros(N+1)
q_xOld = np.zeros(N+1)
hOld, h = np.ones(N+1)*4, np.ones(N+1)*4
hOld[-1], h[-1] = hLast, hLast
# hOld = hOld + z
qUpSt = [20]
qDoSt = [20]

Nind = np.arange(N-1, -1, -1, dtype = int)

E, F, E_x, F_x = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)

"""Preparing variables before time integration"""
b=1
celerity2 = np.ones(N+1)
celerity = .5
diffusivity = np.ones(N+1)*250
diffusivity2 = np.ones(N+1)*250

for t in range(timeStep):
    # print(t)

    E[0] = 1
    F[0] = 0
    E_x[0] = 0
    F_x[0] = 0

    for i in range(1, N + 1):

        C = dt / dx[i-1]
        Cr = np.abs(celerity) * C

        a1 = 3 * Cr ** 2 - 2 * Cr ** 3
        a2 = 1 - a1
        a3 = (Cr ** 2 - Cr ** 3) * dx[i - 1]
        a4 = (-Cr + 2 * Cr ** 2 - Cr ** 3) * dx[i - 1]

        b1 = (6 * Cr - 6 * Cr ** 2) / (-dx[i - 1])
        b2 = (-6 * Cr + 2 * Cr ** 2) / (-dx[i - 1])
        b3 = -(2 * Cr - 3 * Cr ** 2)
        b4 = -(-1 + 4 * Cr - 3 * Cr ** 2)

        d1 = (6 - 12 * Cr ** 2) / (dx[i - 1] ** 2)
        d2 = (-6 + 12 * Cr ** 2) / (dx[i - 1] ** 2)
        d3 = (2 - 6 * Cr) / (dx[i - 1])
        d4 = (4 - 6 * Cr) / (dx[i - 1])

        h1 = 12 / dx[i - 1] ** 3
        h2 = -12 / dx[i - 1] ** 3
        h3 = 6 / dx[i - 1] ** 2
        h4 = 6 / dx[i - 1] ** 2

        if i == N:
            alpha = 1
        else:
            alpha = dx[i] / dx[i-1]

        qEps = a1 * qOld[i-1] + a2 * qOld[i] + a3 * q_x[i-1] + a4 * q_x[i]
        q_xEps = b1 * qOld[i-1] + b2 * qOld[i] + b3 * q_x[i-1] + b4 * q_x[i]
        q_xxEps = d1 * qOld[i-1] + d2 * qOld[i] + d3 * q_x[i-1] + d4 * q_x[i]
        q_xxxEps = h1 * qOld[i-1] + h2 * qOld[i] + h3 * q_x[i-1] + h4 * q_x[i]

        p = -theta * diffusivity[i] * dt / (dx[i-1]**2) * 2 / (alpha * (alpha+1)) * alpha
        q = 1 - p * (1 + alpha) / alpha
        r = p / alpha
        s = qEps + dt * diffusivity[i] * (1-theta) * q_xxEps
        s_x = q_xEps + dt * diffusivity[i] * (1-theta) * q_xxxEps

        E[i] = -r / (p * E[i-1] + q)
        F[i] = (s - p * F[i-1]) / (p * E[i-1] + q)
        E_x[i] = -r / (p * E_x[i-1] + q)
        F_x[i] = (s_x - p * F_x[i - 1]) / (p * E[i - 1] + q)

    qp[-1] = qOld[-2]
    # qp[-1] = qEps # bc at the end
    q_x[-1] = 0

    for i in Nind:
        qp[i] = E[i] * qp[i+1] + F[i]
        q_x[i] = E_x[i] * q_x[i+1] + F_x[i]

    if t*dt < 2*60*60:
        qp[0] = 20
    elif t * dt >= 2 * 60 * 60 and t * dt < 3 * 60 * 60:
        qp[0] = 20 + 5 * (t*dt - 2*60*60) / 60 / 60
    elif t * dt >= 3 * 60 * 60 and t * dt < 4 * 60 * 60:
        qp[0] = 25 - 5 * (t * dt - 3 * 60 * 60) / 60 / 60
    else:
        qp[0] = 20

    # if t*dt < 1*60*60:
    #     qp[0] = 20 + 5 * (t*dt ) / 60 / 60
    # elif t * dt >= 1 * 60 * 60 and t * dt < 2 * 60 * 60:
    #     qp[0] = 25 - 5 * (t*dt - 1*60*60) / 60 / 60
    # else:
    #     qp[0] = 20

    ## h calcs comes here

    width = 10
    wettedPerimeter = 10 + 2 * np.sqrt(hLast ** 2 + (hLast / 2) ** 2)
    csArea = hLast * 10 + hLast ** 2 / 2
    R = csArea / wettedPerimeter
    S_fForward = 1 / ((1 / mannings[-1]) * csArea * R ** (2 / 3)) ** 2 * qp[-1] ** 2
    # S_fForward = mannings[-1] ** 2 / (csArea * R ** (2 / 3)) ** 2 * qp[-1] ** 2
    for i in Nind:
        if h[i] <= 10:
            width = 10
            wettedPerimeter = 10 + 2 * np.sqrt(h[i] ** 2 + (h[i] / 2) ** 2)
            csArea = h[i] * 10 + h[i] ** 2 / 2
        else:
            width = 10
            wettedPerimeter = 10 + 2 * np.sqrt(10 ** 2 + 5 ** 2) + 20 + 2 * (h[i] - 10)
            csArea = 100 + 10 ** 2 / 2 + (h[i] - 10) * 40
        R = csArea / wettedPerimeter
        S_f = ((1 / mannings[i]) * csArea * R ** (2 / 3)) ** -2 * qp[i] ** 2
        # S_f = mannings[i]**2 / ( csArea * R ** (2 / 3)) ** 2 * qp[i] ** 2
        celerity2[i] = 5 * S_fForward ** .3 * np.abs(qp[i]) ** .4 / 3 / width ** .4 / mannings[i] ** .6
        # celerity2[i] = 0
        # celerity2[i] = np.min([3*qp[i]/(h[i]-z[i]), 5 * S_fForward ** .3 * np.abs(qp[i]) ** .4 / 3 / width ** .4 / mannings[i] ** .6])
        diffusivity2[i] = (np.abs(qp[i]) / 2 / S_fForward / width)
        # diffusivity2[i] = 10
        h[i] = h[i + 1] + dx[i] * (S_0 - (S_f + S_fForward) / 2)
        # h[i] = hOld[i + 1]  + dx[i] * (S_f + S_fForward) / 2
        S_fForward = S_f


    #bounds on diff and celerity
    # lb = 50
    # ub = 400
    # if np.mean(diffusivity2[1:-1])<lb:
    #     diffusivity = np.ones(N + 1) * lb
    # elif np.mean(diffusivity2[1:-1])<ub:
    #     diffusivity = np.ones(N + 1) * np.mean(diffusivity2[1:-1])
    # else:
    #     diffusivity = np.ones(N + 1) * ub
    # lb = 0.5
    # ub = 1000
    # if np.mean(celerity2[1:-1])<lb:
    #     celerity = lb
    # elif np.mean(celerity2[1:-1])<ub:
    #     celerity = np.mean(celerity2[1:-1])
    # else:
    #     celerity = np.ones(N + 1) * ub

    diffusivity =  np.ones(N+1) * np.min([250, np.mean(diffusivity2[1:-1])])
    celerity = np.max([.5, np.mean(celerity2[1:-1])])
    # celerity = np.mean(celerity2[1:-1])

    qOld[:] = qp[:]
    q_xOld[:] = q_x[:]
    hOld[:] = h[:]
    if t == 40000:
        print(q_x)

    # if t == 1000:
    #     print(qp, q_x)

    qUpSt.append(qp[0])
    qDoSt.append(qp[-1])



time = np.arange(0, timeStep * dt + dt, dt)
time = time / 3600
plt.plot(time, qUpSt, label='upstream')
plt.plot(time, qDoSt, 'k', linestyle='--',label='downstream')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('discharge')
plt.title("upper limit diffusi=250, lower limit cele=.5, theta=0.5")
plt.savefig('d250-c5-t8.png')
plt.show()