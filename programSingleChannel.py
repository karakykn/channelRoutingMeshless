import matplotlib.pyplot as plt
from source.rbfcmSingleChannel import *

def trapezInteg(y, dt):
    n = y.size
    soln = np.zeros(n-1)
    soln[:] = (y[1:] + y[:-1]) / 2
    soln[:] = soln[:] * dt
    return np.sum(soln)

soln = SingleChannel("case1/inputG6/input.txt")
soln.solveUnsteadyChannel()

peakDischargeDs = np.max(soln.dsq)
peakDsTimeI = np.argmax(soln.dsq)
peakDsTime = soln.time[peakDsTimeI]
massIn = trapezInteg(soln.usq, soln.dt)
massOut = trapezInteg(soln.dsq, soln.dt)
errPerc = np.abs(massIn - massOut) / massIn * 100
print(f"Peak discharge at downstream (m3/s): {peakDischargeDs:.4f}\n Time of peak discharge at downstream (s): {peakDsTime:.0f}\n Error in mass conservation (%): {errPerc:.4f}")

plt.plot(soln.time/3600, soln.usq, "k", label="Upstream")
plt.plot(soln.time/3600, soln.dsq, "--", color = "k", label= "Downstream")
plt.ylabel("Discharge ($m^3s^{-1}$)")
plt.xlabel("Time (s)")
plt.title("RUN 1 (RBFCM)")
plt.grid()
plt.show()

"""visualize"""
# timeIter = float(soln.USBC.size)
# N = soln.nodeNo
# recLength = int(timeIter / soln.printStep)
# qs = np.zeros((N, recLength))
# usq = [20]
# plt.ion()
# for i in range (1, recLength):
#     qs[:, i] = np.loadtxt(soln.outputFolder + "q" + str(i) + ".txt")
#     usq.append(qs[0,i])
#     plt.plot(soln.locations, qs[:,i])
#     plt.grid()
#     plt.ylim(0,40)
#     plt.title(str((i+1) * soln.dt * soln.printStep) + " sec")
#     plt.pause(.1)
#     plt.show()
#     plt.cla()
#
# # upstream downstream control
# tst = np.arange(0,recLength,1)
# plt.ioff()


