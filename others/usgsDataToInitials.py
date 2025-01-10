import os
import pandas as pd
from datetime import timedelta
import numpy as np

# Paths for the code and data files
data_file_path = "../lowerMississippi/baton rouge/usgs.txt"
output_file_path_h = "../lowerMississippi/upstreamH.txt"
output_file_path_bc = "../lowerMississippi/upstreamBC.txt"

# Define the resample time interval (in seconds)
dt = 3600.0  # Example: 3600 seconds (1 hour)

# Read the USGS data file
columns = [
    "agency_cd", "site_no", "datetime", "gage_height_max", "max_cd",
    "gage_height_min", "min_cd", "gage_height_mean", "mean_cd", "discharge", "discharge_cd"
]
data = pd.read_csv(data_file_path, sep='\t', comment='#', names=columns, header=1, parse_dates=["datetime"])

# Filter necessary columns and remove unnecessary rows
data = data[["datetime", "gage_height_mean", "discharge"]]
data.dropna(inplace=True)

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

print(f"Water surface elevations have been saved to {output_file_path_h}")
print(f"Discharge data have been saved to {output_file_path_bc}")
