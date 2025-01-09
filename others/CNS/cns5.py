import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as tm

"""Solution parameters"""
theta = .0
dtt = np.array([500])  # sec
g = 9.81
hyperparam = 10

"""Geometry and related"""
np.random.seed(3)
L = 20000  # length of the channel
Nn = np.array([20])  # no of intervals not nodes
# dx = np.random.rand(N)

qdst = np.array([0, 1])

tol = .5
for dt in dtt:
    timeStep = int(90000 / dt)
    for N in Nn:
        res = 10
        diflim = 50
        while res > tol:
            dx = np.array(np.ones(N) * L / (N))
            print(f"$\Delta x={dx[0]:.0f}$, $\Delta t = {dt:.0f}$, $D_L = {diflim:.0f}$")

            alpha = np.zeros(N + 1)
            x = np.zeros(N + 1)
            mannings = np.ones(N + 1) * .012
            S_0 = 1 / 10000
            z = np.zeros(N + 1)
            hLast = 4
            z[0] = S_0 * L
            for i in range(1, N + 1):
                x[i] = x[i - 1] + dx[i - 1]
                z[i] = z[i - 1] - S_0 * dx[i - 1]
            alpha[1:-1] = dx[1:] / dx[:-1]
            alpha[-1], alpha[0] = 1, 1

            """Variables"""
            qp = np.ones(N + 1) * 20
            qOld = np.ones(N + 1) * 20
            flowU = np.ones(N + 1) * 1
            q_x = np.zeros(N + 1)
            q_xOld = np.zeros(N + 1)
            yy, h = np.ones(N + 1) * hLast, np.ones(N + 1) * hLast
            yy = h + z
            qUpSt = [20]
            qDoSt = [20]

            Nind = np.arange(N - 1, -1, -1, dtype=int)

            E, F, E_x, F_x = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)

            """Preparing variables before time integration"""
            b = 1
            celerity2 = np.ones(N + 1)
            celerity = 1
            diffusivity = np.ones(N + 1) * 250
            diffusivity2 = np.ones(N + 1) * 250
            start_time = tm.time()

            for t in range(timeStep):
                # print(t)

                E[0] = 1
                F[0] = 0
                E_x[0] = 0
                F_x[0] = 0

                for i in range(1, N + 1):

                    C = dt / dx[i - 1]
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
                        alpha = dx[i] / dx[i - 1]

                    qEps = a1 * qOld[i - 1] + a2 * qOld[i] + a3 * q_x[i - 1] + a4 * q_x[i]
                    q_xEps = b1 * qOld[i - 1] + b2 * qOld[i] + b3 * q_x[i - 1] + b4 * q_x[i]
                    q_xxEps = d1 * qOld[i - 1] + d2 * qOld[i] + d3 * q_x[i - 1] + d4 * q_x[i]
                    q_xxxEps = h1 * qOld[i - 1] + h2 * qOld[i] + h3 * q_x[i - 1] + h4 * q_x[i]

                    p = -theta * diffusivity[i] * dt / (dx[i - 1] ** 2) * 2 / (alpha * (alpha + 1)) * alpha
                    q = 1 - p * (1 + alpha) / alpha
                    r = p / alpha
                    s = qEps + dt * diffusivity[i] * (1 - theta) * q_xxEps
                    s_x = q_xEps + dt * diffusivity[i] * (1 - theta) * q_xxxEps

                    E[i] = -r / (p * E[i - 1] + q)
                    F[i] = (s - p * F[i - 1]) / (p * E[i - 1] + q)
                    E_x[i] = -r / (p * E_x[i - 1] + q)
                    F_x[i] = (s_x - p * F_x[i - 1]) / (p * E[i - 1] + q)

                qp[-1] = qOld[-2]
                # qp[-1] = qEps # bc at the end
                q_x[-1] = 0

                for i in Nind:
                    qp[i] = E[i] * qp[i + 1] + F[i]
                    q_x[i] = E_x[i] * q_x[i + 1] + F_x[i]

                if t * dt < 2 * 60 * 60:
                    qp[0] = 20
                elif t * dt >= 2 * 60 * 60 and t * dt < 3 * 60 * 60:
                    qp[0] = 20 + 5 * (t * dt - 2 * 60 * 60) / 60 / 60
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
                flowU[-1] = qp[-1] / csArea
                S_fForward = mannings[-1] ** 2 / (csArea * R ** (2 / 3)) ** 2 * qp[-1] ** 2
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
                    S_f = mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * qp[i] ** 2
                    celerity2[i] = 5 * (S_f) ** .3 * np.abs(qp[i]) ** .4 / 3 / width ** .4 / mannings[i] ** .6
                    diffusivity2[i] = (np.abs(qp[i]) / 2 / (S_f) / width)
                    h[i] = h[i + 1] + (S_0 + (S_f + S_fForward) / 2) * dx[i]
                    S_fForward = S_f
                    flowU[i] = qp[i] / csArea

                # bounds on diff and celerity
                # lbd = 50
                # ubd = 400
                # lbc = .5
                # for i in range(diffusivity2.size):
                # diffusivity2[i] = np.maximum(np.minimum(diffusivity[i], ubd), lbd)
                # celerity2[i] = np.maximum(np.minimum(celerity2[i], 3*flowU[i]), lbc)
                # diffusivity = np.ones(N+1) * np.mean(diffusivity2)
                # celerity = np.mean(celerity2)

                diffusivity = np.ones(N + 1) * np.min([diflim, np.mean(diffusivity2[1:-1])])
                # celerity = np.max([.5, np.mean(celerity2[1:-1])])
                celerity = np.mean(celerity2[1:-1])

                qOld[:] = qp[:]
                q_xOld[:] = q_x[:]
                # if t == 4500:
                #     print(h)
                #     print(qp)
                #     print(q_x)

                # if t == 1000:
                #     print(qp, q_x)

                qUpSt.append(qp[0])
                qDoSt.append(qp[-1])

            end_time = tm.time()
            runtime = end_time - start_time
            print(f"Runtime: {runtime} seconds")

            time = np.arange(0, timeStep * dt + dt, dt)
            time = time / 3600
            diflimN = diflim - (qDoSt[int(timeStep / 5)] - qDoSt[0]) / time[int(timeStep / 5)] * hyperparam
            res = np.abs(diflimN - diflim)
            diflim = diflimN
            print(diflimN)

        np.savetxt(f"cns5Results/qdos_dx{dx[0]:.0f}_dt{dt:.0f}_DL{diflim:.0f}.txt", np.array([qDoSt,time]))
        # plt.plot(time, qDoSt, label = f"$\Delta x={dx[0]:.0f}$, $\Delta t = {dt:.0f}$, $D_L = {diflim:.0f}$")
# plt.grid()
# plt.xlabel('time')
# plt.legend()
# plt.xlim(0, 25)
# plt.ylabel('discharge')
# plt.savefig('fig.pdf')
# plt.show()