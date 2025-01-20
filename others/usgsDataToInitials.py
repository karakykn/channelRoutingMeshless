import os
import pandas as pd
from datetime import timedelta
import numpy as np

# Paths for the code and data files
data_file_path = "../lowerMississippi/baton rouge/usgs.txt"
output_file_path_h = "../lowerMississippi/upstreamH.txt"
output_file_path_bc = "../lowerMississippi/upstreamBC.txt"
output_file_path_bcdc = "../lowerMississippi/downstreamBC.txt"
output_file_path_initQ = "../lowerMississippi/initialQ.txt"
output_file_path_initH = "../lowerMississippi/initialH.txt"

# Define the resample time interval (in seconds)
dt = 1000.0  # Example: 3600 seconds (1 hour)
nodeNo = 11

# Conversion factors
feet_to_meters = 0.3048  # 1 foot = 0.3048 meters
cfs_to_cms = 0.0283168  # 1 cubic foot per second = 0.0283168 cubic meters per second

# Read the USGS data file
columns = [
    "agency_cd", "site_no", "datetime", "gage_height_max", "max_cd",
    "gage_height_min", "min_cd", "gage_height_mean", "mean_cd", "discharge", "discharge_cd"
]
data = pd.read_csv(data_file_path, sep='\t', comment='#', names=columns, header=1, parse_dates=["datetime"])

# Filter necessary columns and remove unnecessary rows
data = data[["datetime", "gage_height_mean", "discharge"]]
data.dropna(inplace=True)

# Convert gage height to meters and discharge to cubic meters per second
data["gage_height_mean"] *= feet_to_meters
data["discharge"] *= cfs_to_cms

# Calculate cumulative seconds for the time column
start_time = data["datetime"].iloc[0]
data["cumulative_seconds"] = (data["datetime"] - start_time).dt.total_seconds()

# Interpolate data at the specified interval
new_times = np.arange(0, data["cumulative_seconds"].iloc[-1] + dt, dt)
new_data = pd.DataFrame({"cumulative_seconds": new_times})
new_data = new_data.merge(data, on="cumulative_seconds", how="outer").sort_values("cumulative_seconds")
new_data.interpolate(method='linear', inplace=True)

# Save the results in separate files without headers or time column
np.savetxt(output_file_path_h, new_data["gage_height_mean"].values, fmt='%f')
np.savetxt(output_file_path_bc, new_data["discharge"].values, fmt='%f')
np.savetxt(output_file_path_bcdc, np.zeros(new_data.shape[0]), fmt='%f')
np.savetxt(output_file_path_initQ, np.ones(nodeNo) * new_data["discharge"][0], fmt='%f') #setting first value in the dataset as initial value for all points in space.
np.savetxt(output_file_path_initH, np.ones(nodeNo) * new_data["gage_height_mean"][0], fmt='%f')

print(f"Water surface elevations (in meters) have been saved to {output_file_path_h}")
print(f"Discharge data (in cubic meters per second) have been saved to {output_file_path_bc}")