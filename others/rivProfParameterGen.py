import numpy as np

def eucDist(x1, x2, delta):
    return np.sqrt((x2-x1)**2 + delta**2)

rivProf = np.loadtxt("rivProf.txt")
N = rivProf.shape[0]

wettedPerim = np.zeros(N)
csArea = np.zeros(N)
R = np.zeros(N)
wettedPerim[0] = rivProf[0, 2] - rivProf[0, 1]
csArea[0] = 0

for i in range(1,N):
    delta = rivProf[i, 0] - rivProf[i-1, 0]
    wettedPerim[i] = wettedPerim[i-1] + eucDist(rivProf[i,1], rivProf[i-1,1], delta) + eucDist(rivProf[i,2], rivProf[i-1,2], delta)
    csArea[i] = csArea[i-1] + ((rivProf[i,2] + rivProf[i-1,2]) / 2 - (rivProf[i,1] + rivProf[i-1,1]) / 2) * delta
R[1:] = wettedPerim[1:] / csArea[1:]

data = np.column_stack((rivProf[:,0], wettedPerim, csArea, R))

# Save to a text file
np.savetxt('../lowerMississippi/riverProfile.txt', data, fmt='%.6f', delimiter=',', header='h, wttedPerim, Area, R')