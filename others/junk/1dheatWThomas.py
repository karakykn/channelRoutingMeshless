import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def thomasAlgorithm(A,B,C,D,N,soln):
    dummyC = np.zeros(N)
    dummyD = np.zeros(N)
    dummyC[0] = C[0] / B[0]
    dummyD[0] = D[0] / B[0]
    for i in range(1,N):
        dummyD[i] = (D[i] - A[i]*dummyD[i-1]) / (B[i] - A[i] * dummyC[i - 1])
        dummyC[i] = C[i] / (B[i] - A[i]*dummyC[i-1])

    soln[-1] = dummyD[-1]
    dummy = np.arange(N-2, -1, -1)
    for i in dummy:
        soln[i] = dummyD[i] - dummyC[i] * soln[i+1]

    return soln

a, b, n = 0, 1, 101
dx = (b-a) / (n-1)
dt = .05
maxTimeStep = 1000
x = np.arange(a, b + dx, dx)
r = dt / dx**2

A = np.ones(n) * (-r / 2)
B = np.ones(n) * (1 + r)
C = np.ones(n) * (-r / 2)
D = np.zeros(n)

"""Dirichlet"""
# A[-1] = 0
# B[0], B[-1] = 1, 1
# C[0] = 0
# D[0] = 100
# D[-1] = 20

"""neumann at other end"""
A[-1] = -1
B[0], B[-1] = 1, 1
C[0] = 0
D[0] = 100
D[-1] = 0

u, uOld = np.zeros(n), np.zeros(n)

i = 0
plt.ion()
while i < maxTimeStep:
    D[1:-1] = uOld[:-2] * r / 2 + uOld[1:-1] * (1 - r) + uOld[2:] * r / 2

    u = thomasAlgorithm(A, B, C, D, n, u)
    uOld[:] = u[:]
    print(u)

    plt.plot(x,u)
    plt.axis([0, 1, 0, 110])
    plt.grid()
    plt.pause(.2)
    plt.show()
    plt.cla()