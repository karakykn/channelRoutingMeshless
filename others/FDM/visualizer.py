import os
import numpy as np
import matplotlib.pyplot as plt

# Base directory
base_dir = 'case1'

# Output folders to use
output_folders = ['output1', 'output26', 'output51']

# Define linestyles for each plot
# linestyles = ['-', '--', ':', '-.', (0, (3, 5, 1, 5, 1, 5))]  # Solid, dashed, and dotted lines
linestyles = ['-', '--', ':']  # Solid, dashed, and dotted lines
dx = ["$2000\,m$", "$1000\,m$", "$500\,m$"]
dt = ["$1000\,s$", "$500\,s$", "$250\,s$", "$125\,s$"]
dl = ["$0\,m^2s^{-1}$", "$100\,m^2s^{-1}$", "$200\,m^2s^{-1}$", "$300\,m^2s^{-1}$", "$400\,m^2s^{-1}$", ]

# Initialize a list to store time and downstreamQ data for each folder
time_data_list = []
downstreamQ_data_list = []

# Iterate over the specified output folders
for output_folder in output_folders:
    folder_path = os.path.join(base_dir, output_folder)

    # Check if time.txt file exists
    time_file = os.path.join(folder_path, 'time.txt')
    if not os.path.exists(time_file):
        print(f"Error: {time_file} not found")
        continue

    # Load time data
    time_data = np.loadtxt(time_file)
    time_data_list.append(time_data)

    # Load downstreamQxxxx.txt file
    downstreamQ_files = [f for f in os.listdir(folder_path) if f.startswith('downstreamQ')]
    if len(downstreamQ_files) == 0:
        print(f"Error: No downstreamQxxxx.txt file found in {folder_path}")
        continue

    downstreamQ_path = os.path.join(folder_path, downstreamQ_files[0])

    # Load downstreamQ data
    downstreamQ_data = np.loadtxt(downstreamQ_path)
    downstreamQ_data_list.append(downstreamQ_data)

# Plotting (only if data is loaded)
if time_data_list and downstreamQ_data_list:
    plt.figure(figsize=(10, 6))

    # Plot each downstreamQxxxx data against its corresponding time data with different linestyles
    for i in range(len(time_data_list)):
        plt.plot(time_data_list[i]/3600, downstreamQ_data_list[i], label='$D_{limit}=$' + f'{dl[i]}', linestyle=linestyles[i], color="k")

    # Labeling the plot
    plt.xlabel('Time (h)')
    plt.ylabel('Discharge ($m^3s^{-1}$)')
    plt.title('Downstream Discharge Over Time (RBFCM, $\Delta x=500\,m$, $\Delta t=500\,s$)')
    plt.legend()
    plt.grid(True)
    plt.savefig("case1/dx_dt_constant.pdf")
    plt.show()
else:
    print("No data to plot.")
