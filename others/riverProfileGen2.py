import numpy as np
import pandas as pd
"""This code is to generate river profile 
parameters"""

data = np.loadtxt('bos.txt', delimiter=' ', skiprows=1)

print(data)
print(data[1,2])

