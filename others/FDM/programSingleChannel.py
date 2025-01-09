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