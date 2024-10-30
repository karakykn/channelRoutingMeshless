import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Get the current directory where the script is located
current_directory = os.getcwd()

# Loop through all files in the current directory
for filename in os.listdir(current_directory):
    print(filename)
    file_path = os.path.join(current_directory, filename)

    if os.path.isfile(file_path):
        # Extract integers from the filename using regular expressions
        numbers_in_filename = re.findall(r'\d+', filename)
        if numbers_in_filename:
            # Convert strings to integers
            integers = [int(num) for num in numbers_in_filename]

            if len(integers) == 2:
                integers.append(-1)

            qdo = np.loadtxt(filename)
            plt.plot(qdo[1], qdo[0], label = f"$\Delta x={integers[0]:.0f}$, $\Delta t = {integers[1]:.0f}$, $D_L = {integers[2]:.0f}$")
plt.grid()
plt.xlabel('time')
plt.legend()
plt.xlim(0, 25)
plt.ylabel('discharge')
plt.savefig('fig.pdf')
plt.show()