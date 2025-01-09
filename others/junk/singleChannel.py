import matplotlib.pyplot as plt
import numpy as np

from rbfcm import *

"""setting up the input variables"""
dt = 18 # sec
nodd = 21
x = np.linspace(0,20000, nodd)
qIni = np.ones(nodd) * 20
hIni = np.ones(nodd) * 4
mannings = np.ones(nodd) * .012
slope = np.ones(nodd) / 10000
np.savetxt("inputs/simplechannel/locations.txt", x)
np.savetxt("inputs/simplechannel/initialQ.txt", qIni)
np.savetxt("inputs/simplechannel/initialH.txt", hIni)
np.savetxt("inputs/simplechannel/mannings.txt", mannings)
np.savetxt("inputs/simplechannel/slopes.txt", slope)
bWidth = 10

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


"""This is where the solution starts"""
soln = Rbfcm("inputs/simplechannel/locations.txt", qIni, hIni, bWidth, mannings, slope)
N = soln.nodeNo


# soln.buildMQ(shapeParameter=-2)
soln.buildTPS(4)

bvpath1 = "inputs/simplechannel/upstreamBC.txt"
bvpath2 = "inputs/simplechannel/downstreamBC.txt"
outputFolder = "outputs/simplechannel/"
soln.solveUnsteadyChannel(np.zeros(N), qIni, bvpath1, bvpath2, dt = dt, printStep=100, printFolder=outputFolder)


"""visualize"""
recLength = int(timeIter / soln.printStep)
qs = np.zeros((N, recLength))
usq = []
dsq = []
plt.ion()
for i in range (1, recLength):
    qs[:, i] = np.loadtxt(outputFolder + "q" + str(i) + ".txt")
    usq.append(qs[0,i])
    dsq.append(qs[-1,i])
    plt.plot(soln.locations, qs[:,i])
    plt.grid()
    plt.ylim(0,40)
    plt.title(str((i+1) * dt * soln.printStep) + " sec")
    plt.pause(.1)
    plt.show()
    plt.cla()

# upstream downstream control
tst = np.arange(1,recLength,1)
plt.ioff()
plt.plot(tst, usq)
plt.plot(tst, dsq)
plt.grid()
plt.show()