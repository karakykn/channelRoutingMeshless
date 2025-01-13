import numpy as np
import pandas as pd
"""This code is to generate river profile 
parameters"""

maxH = 40
sideSlope = 2
bottomWidth = 10
extension1 = 20
trapLength = 10

y = np.linspace(0, 40, 4001)
wP = np.zeros(y.size)
csArea = np.zeros(y.size)
R = np.zeros(y.size)

sideL = trapLength / sideSlope
bigHipp = np.sqrt(sideL**2 + trapLength**2)
wpBase = bottomWidth + 2 * bigHipp + extension1
areaBase = bottomWidth * trapLength + sideL * trapLength
baseLength2 = bottomWidth + 2 * sideL + extension1

for i, yy in zip(range(y.size), y):
    if yy <= trapLength:
        horiz = yy / sideSlope
        topWidth = bottomWidth + 2 * horiz
        hippo = np.sqrt(yy**2 + horiz**2)
        wP[i] = bottomWidth + 2 * hippo
        csArea[i] = (bottomWidth + topWidth) / 2 * yy
        R[i] = csArea[i] / wP[i]
    else:
        yEx = yy - trapLength
        wP[i] = wpBase + 2 * yEx
        csArea[i] = areaBase + baseLength2 * yEx
        R[i] = csArea[i] / wP[i]

data = np.column_stack((y, wP, csArea, R))

# Save to a text file
np.savetxt('../lowerMississippi/riverProfile.txt', data, fmt='%.6f', delimiter=',', header='h, wttedPerim, Area, R')
