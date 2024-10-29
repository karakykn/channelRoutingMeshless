import numpy as np
import matplotlib.pyplot as plt

nn = [2000,1000,500]
errorMC = [0.0857, 0.0312, 0.0254]
errorMC.append([0.1123, 0.0599, 0.0544])

# nnlog = np.log(nn)
# erroLog = np.log(errorMC)

plt.loglog(nn,errorMC[:3], "k", marker="*")
plt.loglog(nn,errorMC[:3], "k")
plt.loglog(nn,errorMC[3], "k", marker="*")
plt.loglog(nn,errorMC[3], "k")
plt.xlim(100,10000)
plt.grid()
plt.show()