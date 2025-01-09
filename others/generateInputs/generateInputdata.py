import numpy as np

locs = np.linspace(0,20000, 1001)
np.savetxt("locations.txt", locs)