import numpy as np
import matplotlib.pyplot as plt

dsq2000 = np.loadtxt("results/downstreamQ2000.txt")
dsq1000 = np.loadtxt("results/downstreamQ1000.txt")
dsq500 = np.loadtxt("results/downstreamQ500.txt")
dsq250 = np.loadtxt("results/downstreamQ250.txt")
time = np.loadtxt("results/time.txt")
upstream = np.loadtxt("results/upstreamQ250.txt")

plt.plot(time, upstream, "k", label="Inflow")
plt.plot(time, dsq2000, color="#2FF3E0", label="Outflow $(\Delta x = 2\,km)$")
plt.plot(time, dsq1000, color="#F8D210", label="Outflow $(\Delta x = 1\,km)$")
plt.plot(time, dsq500, color="#FA26A0", label="Outflow $(\Delta x = 0.5\,km)$")
plt.plot(time, dsq250, color="#F51720", label="Outflow $(\Delta x = 0.25\,km)$")
plt.ylabel("Discharge $(m^3s^{-1})$")
plt.xlabel("Time $(s)$")
plt.xlim(0,90000)
plt.title("Inflow and Outflow Hydrographs (RBFCM-TPS)")
plt.legend()
# plt.grid()
plt.show()