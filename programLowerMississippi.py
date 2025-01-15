from source.rbfcmLowerMississippi import *
"""water level will be fixed at the upstream"""
import matplotlib.pyplot as plt
from source.rbfcmLowerMississippi import *

soln = SingleChannel("lowerMississippi/input.txt")
soln.solveUnsteadyChannel()



