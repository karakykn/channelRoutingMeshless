import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def hermiteInterp(C, dt, dx):
    """
        hermiteInterp(C, dt, dx)

            Calculates the Hermitian Interpolation for given parameters.

            Parameters
            ----------
            C : float
                Celerity.
            dt : float
                Time step
            dx : array_like
                The distance between adjacent nodes.

            Returns
            -------
            a1, a2, a3, a4, b1, b2, b3, b4, d1, d2, d3, d4, h1, h2, h3, h4: Each array-like
                Coefficients of Hermitian.
        """

    n = dx.shape[0]
    Cr, a1, a2, a3, a4, b1, b2, b3, b4, d1, d2, d3, d4, h1, h2, h3, h4 = np.zeros(n), +\
    np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), +\
    np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), +\
    np.zeros(n), np.zeros(n)

    for i in range(1, n):
        Cr[i] = dt / dx[i-1] * C

        a1[i] = 3*Cr[i]**2 - 2*Cr[i]**3
        a2[i] = 1 - 3*Cr[i]**2 + 2*Cr[i]**3
        a3[i] = (Cr[i]**2 - Cr[i]**3) * dx[i-1]
        a4[i] = (-Cr[i] + 2*Cr[i]**2 - Cr[i]**3) * dx[i-1]

        b1[i] = (6*Cr[i] - 6*Cr[i]**2) / (-dx[i-1])
        b2[i] = (-6*Cr[i] + 2*Cr[i]**2) / (-dx[i-1])
        b3[i] = -(2*Cr[i] - 3*Cr[i]**2)
        b4[i] = -(-1 + 4*Cr[i] - 3*Cr[i]**2)

        d1[i] = (6 - 12*Cr[i]**2) / (dx[i-1]**2)
        d2[i] = (-6 + 12*Cr[i]**2) / (dx[i-1]**2)
        d3[i] = (2 - 6*Cr[i]) / (dx[i-1])
        d4[i] = (4 - 6*Cr[i]) / (dx[i-1])

        h1[i] = 12 / dx[i-1]**3
        h2[i] = -12 / dx[i-1]**3
        h3[i] = 6 / dx[i-1]**2
        h4[i] = 6 / dx[i-1]**2

    return a1, a2, a3, a4, b1, b2, b3, b4, d1, d2, d3, d4, h1, h2, h3, h4

def calculations_q_eps(a1,a2,a3,a4,d1,d2,d3,d4,qOld,q_xOld):

    qEps = np.zeros(N+1)
    q_xEps = np.zeros(N+1)
    q_xxEps = np.zeros(N+1)
    q_xxxEps = np.zeros(N+1)

    qEps[1:] = a1[1:] * qOld[:-1] + a2[1:] * qOld[1:] + a3[1:] * q_xOld[:-1] + a4[1:] * q_xOld[1:]
    q_xEps[1:] = b1[1:] * qOld[:-1] + b2[1:] * qOld[1:] + b3[1:] * q_xOld[:-1] + b4[1:] * q_xOld[1:]
    q_xxEps[1:] = d1[1:] * qOld[:-1] + d2[1:] * qOld[1:] + d3[1:] * q_xOld[:-1] + d4[1:] * q_xOld[1:]
    q_xxxEps[1:] = h1[1:] * qOld[:-1] + h2[1:] * qOld[1:] + h3[1:] * q_xOld[:-1] + h4[1:] * q_xOld[1:]

    return qEps, q_xEps, q_xxEps, q_xxxEps

def calculations_pqr(dt, dxSq, alpha, D, theta, N):

    p, q, r = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)

    for i in range(1,N):
        p[i] = dt * D * theta / dxSq[i] * 2 / (1 + alpha[i])
        q[i] = 1 - p[i] * (1 + alpha[i]) / alpha[i]
        r[i] = p[i] / alpha[i]

    return p, q, r

def calculations_s(dt, D, theta, qEps, q_xEps, q_xxEps, q_xxxEps, N):

    s = np.zeros(N+1)
    s_x = np.zeros(N+1)

    for i in range(1,N):
        s[i] = qEps[i] + dt * D * (1-theta) * q_xxEps[i]
        s_x[i] = q_xEps[i] + dt * D * (1-theta) * q_xxxEps[i]

    return s, s_x

def calculations_EF(p, q, r, s, s_x, N):

    E, F, E_x, F_x = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)

    """Boundaries"""
    E[0], E[-1], F[0], F[-1] = 1, 0, 1, 0
    E_x[0], E_x[-1], F_x[0], F_x[-1] = 1, 0, 1, 0

    for i in range(1,N):
        E[i] = -r[i] / (p[i] * E[i-1] + q[i])
        F[i] = (s[i] - p[i] * F[i-1]) / (p[i] * E[i-1] + q[i])
        E_x[i] = -r[i] / (p[i] * E_x[i-1] + q[i])
        F_x[i] = (s_x[i] - p[i] * F_x[i-1]) / (p[i] * E_x[i-1] + q[i]) #This is different than paper, paper uses F, E instead

    return E, F, E_x, F_x

def solution(E, F, E_x, F_x, N, Q, Q_x):

    dummy = np.arange(N-1, -1, -1)
    for i in dummy:
        Q[i] = E[i] * Q[i-1] + F[i]
        Q_x[i] = E_x[i] * Q_x[i] + F_x[i]

    return Q, Q_x

def calculations_hCD(dx, Q, n, z, N):

    S_f = np.zeros(N+1)
    R = np.zeros(N+1) # Hydraulic radius
    D = np.zeros(N+1)
    C = np.zeros(N+1)
    h = np.zeros(N+1)

    h[-1] = 4 # Water level at the downstream node

    residual = 1
    eps = 1e-4
    while residual > eps:
        for i in range(N+1):
            if h[i] <= 10:
                width = 10 + h[i]
                wettedPerimeter = 10 + 2 * np.sqrt(h[i]**2 + (h[i] / 2)**2)
                csArea = h[i] * 10 + h[i]**2 / 2
            else:
                width = 40
                wettedPerimeter = 10 + 2 * np.sqrt(10**2 + 5**2) + 20 + 2 * (h[i] - 10)
                csArea = 100 + 10**2 / 2 + (h[i] - 10) * 40
            R[i] = csArea / wettedPerimeter
            S_f[i] = ((1 / n[i]) * csArea * R[i]**(2/3))**2 / Q[i]**2
            C[i] = 5 * S_f[i]**.3 * Q[i]**.4 / 3 / width**.4 / n[i]**.6
            D[i] = np.abs(Q[i]) / 2 / S_f[i] / width

        for i in range(N):
            h[N-i] = h[N-i+1] + z[N-i] - z[N-i+1] + dx[N-i+1] * (S_f[N-i] + S_f[N-i+1]) / 2

    return h, np.mean(C), np.mean(D)


"""Solution parameters"""
theta = .5
dt = .01
C = 1
timeStep = 1000

"""Geometry"""
np.random.seed(3)
N = 20
dx = np.random.rand(N)
dxSq = np.zeros(N+1)
alpha = np.zeros(N+1)
dx = dx / dx.sum()
x = np.zeros(N+1)
for i in range(1,N+1):
    x[i] = x[i-1] + dx[i-1]
    dxSq[i] = dx[i-1] * dx[i]
alpha[1:-1] = dx[1:] / dx[:-1]

"""Variables"""
q = np.zeros(N+1)
qOld = np.zeros(N+1)
q_x = np.zeros(N+1)

"""Preparing variables before time integration"""
a1, a2, a3, a4, b1, b2, b3, b4, d1, d2, d3, d4, h1, h2, h3, h4 = hermiteInterp(C, dt, dx)

b=1

for t in range(timeStep):

