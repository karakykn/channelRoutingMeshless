import numpy as np

dt = 500

endTime = 25 #hr
timeIter = int(25 * 60 * 60 / dt)
us = np.ones(timeIter) * 20
ds = np.zeros(timeIter)

for iter in range(timeIter):
    t = dt * iter
    if t >= 2 * 60 * 60 and t < 3 * 60 * 60:
        us[iter] = 20 + 5 * (t - 2*60*60) / 60 / 60
    elif t  >= 3 * 60 * 60 and t  < 4 * 60 * 60:
        us[iter] = 25 - 5 * (t - 3 * 60 * 60) / 60 / 60

np.savetxt("inputs/simplechannel/upstreamBC.txt", us)
np.savetxt("inputs/simplechannel/downstreamBC.txt", ds)