import numpy as np

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

# N = 1000
# A = np.random.random(N)
# B = np.random.random(N)
# C = np.random.random(N)
# D = np.random.random(N)
# soln = np.zeros(N)
#
# system = np.zeros((N,N))
#
# for i in range(1,N-1):
#     system[i,i] = B[i]
#     system[i,i-1] = A[i]
#     system[i, i+1] = C[i]
#
# system[0,0] = B[0]
# system[0,1] = C[0]
# system[-1,-1] = B[-1]
# system[-1,-2] = A[-1]
# invSys = np.linalg.inv(system)
# aSoln = np.matmul(invSys, D)
# soln = thomasAlgorithm(A,B,C,D,N, soln)
#
# # print(aSoln)
# # print(soln)
#
# resDif = np.matmul(system, invSys) - np.eye(N)
# print(resDif.mean())
#
# res = (aSoln**2 - soln**2).mean()
# print(res)
# print(soln[5], aSoln[5])
