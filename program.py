import matplotlib.pyplot as plt
from source.rbfcm import *

soln = Rbfcm("case1/inputs/input.txt")
soln.solveUnsteadyChannel()


"""visualize"""
timeIter = float(soln.USBC.size)
N = soln.nodeNo
recLength = int(timeIter / soln.printStep)
qs = np.zeros((N, recLength))
usq = []
dsq = []
plt.ion()
for i in range (1, recLength):
    qs[:, i] = np.loadtxt(soln.outputFolder + "q" + str(i) + ".txt")
    usq.append(qs[0,i])
    dsq.append(qs[-1,i])
    plt.plot(soln.locations, qs[:,i])
    plt.grid()
    plt.ylim(0,40)
    plt.title(str((i+1) * soln.dt * soln.printStep) + " sec")
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