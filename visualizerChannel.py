import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the text files
directory = 'case1N/output'

# Find all files and extract the maximum channel and iteration numbers
channels = set()
iterations = set()

for filename in os.listdir(directory):
    match = re.match(r"channel(\d)[Qh](\d+)\.txt", filename)
    if match:
        channels.add(int(match.group(1)))
        iterations.add(int(match.group(2)))

# Convert to sorted lists to index arrays correctly
channels = sorted(list(channels))
iterations = sorted(list(iterations), key=int)

# Initialize numpy arrays for Q and h first and last values for each channel and iteration
Q_values_first = np.full((len(channels), len(iterations)), np.nan)
Q_values_last = np.full((len(channels), len(iterations)), np.nan)
h_values_first = np.full((len(channels), len(iterations)), np.nan)
h_values_last = np.full((len(channels), len(iterations)), np.nan)

# Load data from files
for filename in os.listdir(directory):
    match = re.match(r"channel(\d)([Qh])(\d+)\.txt", filename)
    if match:
        channel = int(match.group(1))
        parameter = match.group(2)
        iteration = int(match.group(3))

        # Find the index in the arrays
        channel_idx = channels.index(channel)
        iteration_idx = iterations.index(iteration)

        # Read first and last values from the file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            lines = file.readlines()
            if lines:
                first_value = float(lines[0].strip())
                last_value = float(lines[-1].strip())

                # Assign first and last values to the correct arrays based on parameter
                if parameter == 'Q':
                    Q_values_first[channel_idx, iteration_idx] = first_value
                    Q_values_last[channel_idx, iteration_idx] = last_value
                elif parameter == 'h':
                    h_values_first[channel_idx, iteration_idx] = first_value
                    h_values_last[channel_idx, iteration_idx] = last_value

# Plotting first and last values for Q
Q_values_first[0, :] -= 20
Q_values_last[0, :] -= 20
plt.figure(figsize=(10, 6))
for i, channel in enumerate(channels):
    plt.plot(iterations, Q_values_first[i, :], '--', label=f'Channel {channel} - Q (First Value)')
    plt.plot(iterations, Q_values_last[i, :], label=f'Channel {channel} - Q (Last Value)')


plt.xlabel('Time Iteration')
plt.ylabel('Q Value')
# plt.ylim(19,25)
plt.title('First and Last Q Values for Each Channel Over Time Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Plotting first and last values for h
# plt.figure(figsize=(10, 6))
# for i, channel in enumerate(channels):
#     plt.plot(iterations, h_values_first[i, :], 'o--', label=f'Channel {channel} - h (First Value)')
#     plt.plot(iterations, h_values_last[i, :], label=f'Channel {channel} - h (Last Value)')
#
# plt.xlabel('Time Iteration')
# plt.ylabel('h Value')
# plt.title('First and Last h Values for Each Channel Over Time Iterations')
# plt.legend()
# plt.grid(True)
# plt.show()
