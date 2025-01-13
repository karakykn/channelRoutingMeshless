import numpy as np
import pandas as pd
"""This code is to generate river profile 
parameters"""

def interp(realH, upy, downy, upx1, downx1, upx2, downx2):
    ratio = (realH - downy) / (upy - downy)
    x1 = ratio * (upx1 - downx1) + downx1
    x2 = ratio * (upx2 - downx2) + downx2
    return x1, x2

def indFind(realH, data_1):
    N = data_1.size
    for i in range(N):
        if realH >= data_1[N - i - 1]:
            return N - i - 1

data = np.loadtxt('bos.txt', delimiter=' ', skiprows=1)

print(data)
print(data[-1,0])

maxH = data[-1,0]
count = 201
h = np.linspace(0, maxH, count)

datGen = np.zeros((count, 3))
datGen[:,0] = h
datGen[0,:] = data[0,:]
datGen[-1,:] = data[-1,:]

for i in range(1,count-1):
    k = indFind(h[i], data[:,0])
    datGen[i, 1:] = interp(h[i], data[k+1,0], data[k,0], data[k+1,1], data[k,1], data[k+1,2], data[k,2])

np.savetxt("rivProf.txt", datGen, delimiter=" ", fmt="%.4f", header="y, x1, x2")