import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a, b, n = 0, 1, 101
dx = (b-a) / (n-1)
dt = .01
maxTimeStep = 1000
x = np.arange(a, b + dx, dx)
r = dt / dx**2
A = np.ones(n) * (-r / 2)
B = np.ones(n) * (1 + r)
C = np.ones(n) * (-r / 2)
D = np.zeros(n)
system = np.zeros((n,n))

for i in range(1,n-1):
    system[i,i] = B[i]
    system[i,i-1] = A[i]
    system[i, i+1] = C[i]

system[0,0] = 1
system[-1,-1] = B[-1]
system[-1,-2] = A[-1]
invSys = np.linalg.inv(system)

u = np.zeros(n)
# u[0] = 100
uOld = u

D[0] = 100
i = 0

plt.ion()
while i < maxTimeStep:
    D[1:-1] = uOld[:-2] * r / 2 + uOld[1:-1] * (1 - r) + uOld[2:] * r / 2
    D[-1] = uOld[-2] * r / 2 + uOld[-1] * (1 - r)
    u = np.matmul(invSys,D)
    uOld[:] = u[:]
    # print(u)

    plt.plot(x,u)
    plt.pause(.2)
    plt.show()
    plt.cla()