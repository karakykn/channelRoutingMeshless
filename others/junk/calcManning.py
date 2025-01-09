import numpy as np

def calcH(Q, n, slope, sideSlope, bottom):
    lhs = Q * n / np.sqrt(slope)
    tol = 1e-4
    hOld = 10
    res = 1
    while res > tol:
        area = bottom * hOld + hOld ** 2 * sideSlope
        h = (area**(5/2) * lhs**(-3/2) - bottom) / 2 / np.sqrt(1 + sideSlope ** 2)
        res = np.abs(h - hOld)
        hOld = h
        print(h)

calcH(20, .012, 1e-4, .5, 10)