import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def buildMQ(topo, xEval):
    locs = topo[0][:]
    elevs = topo[1][:]
    N = locs.size
    # c = 4 * np.min(topo[0][1:]-topo[0][:-1])
    c = .815 * np.mean(topo[0][1:]-topo[0][:-1])
    elevEval = np.array([])
    phiHat = np.zeros((N,N))
    phiV = np.zeros(N)
    for i in range(N):
        for j in range(N):
            r = np.abs(locs[i] - locs[j])
            phiHat[i, j] = np.sqrt(r ** 2 + c ** 2)
    invPhi = np.linalg.pinv(phiHat)
    alpha = np.matmul(invPhi, elevs)
    for x in xEval:
        for i in range(N):
            r = np.abs(x - locs[i])
            phiV[i] = np.sqrt(r ** 2 + c ** 2)
        elevEval = np.append(elevEval, np.array([np.matmul(phiV, alpha)]))
    return elevEval

def buildIMQ(topo, xEval):
    locs = topo[0][:]
    elevs = topo[1][:]
    N = locs.size
    # c = 4 * np.min(topo[0][1:]-topo[0][:-1])
    c = .001 * np.mean(topo[0][1:]-topo[0][:-1])
    elevEval = np.array([])
    phiHat = np.zeros((N,N))
    phiV = np.zeros(N)
    for i in range(N):
        for j in range(N):
            r = np.abs(locs[i] - locs[j])
            phiHat[i, j] = 1 / np.sqrt(r ** 2 + c ** 2)
    invPhi = np.linalg.pinv(phiHat)
    alpha = np.matmul(invPhi, elevs)
    for x in xEval:
        for i in range(N):
            r = np.abs(x - locs[i])
            phiV[i] = 1 / np.sqrt(r ** 2 + c ** 2)
        elevEval = np.append(elevEval, np.array([np.matmul(phiV, alpha)]))
    return elevEval

def buildTPS(topo, xEval):
    locs = topo[0][:]
    elevs = topo[1][:]
    N = locs.size
    beta = 2
    elevEval = np.array([])
    phiHat = np.zeros((N,N))
    phiV = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i != j:
                r = np.sqrt((locs[i] - locs[j]) ** 2)
                phiHat[i, j] = r ** beta * np.log(r)
    invPhi = np.linalg.pinv(phiHat)
    alpha = np.matmul(invPhi, elevs)
    for x in xEval:
        for i in range(N):
            r = np.abs(x - locs[i])
            phiV[i] = r ** beta * np.log(r)
        elevEval = np.append(elevEval, np.array([np.matmul(phiV, alpha)]))
    return elevEval


# Load and process the data
df = pd.read_csv('hecrasXS2.csv', delimiter=',')
df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
xS = df.to_numpy(dtype=float).T  # Transpose to match the expected structure
eps = 1e-6
for i in range(1, xS[0].size):
    for j in range(i+1, xS[0].size):
        if np.abs(xS[0][i] - xS[0][j]) < eps:  # Check if the current value is less than or equal to the previous one
            xS[0][j] += eps  # Adjust the current value

# Set the x limits

wsXL = 1474
wsXR = 2269
# wsXL = 0.028 + eps
# wsXR = 0.035 - eps
xLocs = np.linspace(wsXL, wsXR, 20)

mask = (xS[0] <= wsXL) | (xS[0] > wsXR)
xSCrop = xS[:, mask]

approxElevs = buildTPS(xSCrop, xLocs)

xxx = np.linspace(0, 20000, 21)

buildTPS([xxx,xxx], xxx)

plt.plot(xS[0], xS[1], label="Original")
plt.plot(xSCrop[0], xSCrop[1], "x")
# plt.xlim(3700,5300)
plt.plot(xLocs, approxElevs, label="approximated")
plt.legend()
# plt.savefig("plot.png")
plt.show()
