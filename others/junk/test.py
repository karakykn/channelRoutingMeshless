import numpy as np

def calculations_hCD(dx, Q, n, z, N):

    S_f = np.ones(N + 1) * .1
    R = np.zeros(N + 1)  # Hydraulic radius
    D = np.zeros(N + 1)
    C = np.zeros(N + 1)
    h = np.ones(N + 1)*.1
    hOld = np.zeros(N + 1)

    h[-1] = 4  # Water level at the downstream node

    residual = 1
    eps = 1e-4
    jj = 0
    while residual > eps:
        jj += 1
        print(jj)
        for i in range(N + 1):
            if h[i] <= 10:
                width = 10 + h[i]
                wettedPerimeter = 10 + 2 * np.sqrt(h[i] ** 2 + (h[i] / 2) ** 2)
                csArea = h[i] * 10 + h[i] ** 2 / 2
            else:
                width = 40
                wettedPerimeter = 10 + 2 * np.sqrt(10 ** 2 + 5 ** 2) + 20 + 2 * (h[i] - 10)
                csArea = 100 + 10 ** 2 / 2 + (h[i] - 10) * 40
            R[i] = csArea / wettedPerimeter
            S_f[i] = ((1 / n[i]) * csArea * R[i] ** (2 / 3)) ** 2 / Q[i] ** 2
            C[i] = 5 * S_f[i] ** .3 * Q[i] ** .4 / 3 / width ** .4 / n[i] ** .6
            D[i] = np.abs(Q[i]) / 2 / S_f[i] / width

        for i in range(N):
            h[N - i - 1] = h[N - i] + z[N - i - 1] - z[N - i] + dx[N - i - 1] * (S_f[N - i - 1] + S_f[N - i]) / 2

        residual = np.mean(h**2 - hOld**2) / N
        hOld[:] = h[:]

    return h, np.mean(C), np.mean(D)

N = 20
dx = np.ones(N) * 1 / N
Q = np.ones(N+1)
n = np.ones(N+1) * .012
S_0 = .01
z = np.zeros(N+1)
for i in range(1,N+1):
    z[i] = z[i-1] + dx[i-1]*S_0

h, C, D = calculations_hCD(dx, Q, n, z, N)

print(h)
print(C)
print(D)